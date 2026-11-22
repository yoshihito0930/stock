"""ルール台帳 (rules/active.json) と、その自動更新ロジック。

週次（research）:
  ファミリーごとに「新しい候補」と「いま台帳にあるルール」を同じ最新データで採点し直し、
  入れ替え・昇格・降格・廃止を決める。変更はすべて更新履歴に残す。

日次（refresh）:
  台帳のルールを最新データで採点し直し、直近の成績が崩れたものを一時停止する。
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .config import ACTIVE_RULES, CHANGELOG, HISTORY_JSON, RESEARCH_JSON, Config
from .portfolio import build_portfolio
from .research import (Context, check_gates, evaluate_rule, health_check, neighbor_pass_for,
                       research_family, rule_objective, tier_of)
from .strategies import FAMILIES, Params, param_key
from .util import clean, read_json, write_json

TIER_RANK = {None: 0, "watch": 1, "active": 2}
STATUS_LABEL = {"active": "採用", "watch": "観察中", "suspended": "一時停止"}

PARAM_LABELS = {
    "trend": "長期トレンド条件", "mom": "半年の上昇率条件", "rsi_n": "RSI の日数", "rsi_th": "RSI のしきい値",
    "limit_pct": "指値の位置", "max_hold": "最大保有日数", "stop_pct": "損切り幅", "market": "相場環境フィルター",
    "index_rsi": "指数 RSI のしきい値", "surge": "5日間の急騰率", "lookback": "高値/安値の参照日数",
    "vol_ratio": "売買代金の増加倍率", "exit_sma": "決済に使う移動平均", "jump": "前日比の上昇率",
}


def _fmt_param(name: str, v) -> str:
    if v is None or v == "none" or v == 0:
        return "なし"
    if name in ("limit_pct", "stop_pct", "mom", "surge", "jump"):
        return f"{v * 100:g}%"
    if name == "market":
        return {"up200": "指数が200日線より上", "down200": "指数が200日線より下"}.get(v, str(v))
    if name in ("trend", "exit_sma", "lookback", "max_hold", "rsi_n"):
        return f"{v}日"
    if name == "vol_ratio":
        return f"{v:g}倍"
    return str(v)


def param_diff(old: Params, new: Params) -> list[str]:
    out = []
    for k in new:
        if old.get(k) != new[k]:
            out.append(f"{PARAM_LABELS.get(k, k)}: {_fmt_param(k, old.get(k))} → {_fmt_param(k, new[k])}")
    return out


def empty_state() -> dict:
    return {"schema": 1, "updated_at": None, "data_through": None, "rules": [], "rejected": []}


def load_state() -> dict:
    return read_json(ACTIVE_RULES, None) or empty_state()


def _evidence_line(ev: dict) -> str:
    f, o = ev["full"], ev["oos"]
    return (f"全期間: 取引 {f['trades']} 回・勝率 {f['win_rate']:.0%}・PF {f['profit_factor']:.2f}・"
            f"月平均 {f['monthly_mean'] / 1e4:+.1f}万円 ／ 未使用データ: PF {o['profit_factor']:.2f}・"
            f"月平均 {o['monthly_mean'] / 1e4:+.1f}万円")


def _failed(gates: list[dict]) -> list[str]:
    return [g["name"] for g in gates if not g["ok"]]


def _make_rule(family: str, params: Params, tier: str, ev: dict, gates: list[dict], wf: dict | None,
               neighbor: float | None, today: str, version: int, since: str | None = None) -> dict:
    return {
        "id": family, "family": family, "version": version, "status": tier, "tier": tier,
        "since": since or today, "adopted_on": today, "params": params,
        "evidence": ev, "gates": gates, "walk_forward": wf, "neighbor_pass": neighbor,
        "health": {"ok": True, "reasons": []},
    }


def config_change_events(cfg: Config, state: dict, today: str) -> list[dict]:
    """前回から資金・目標・建玉上限・保有数の設定が変わっていれば、その旨を履歴に残す。

    設定が変わると同じルールでも採点が変わる。ルールの入れ替えが相場のせいか設定のせいかを後から区別できるようにする。
    """
    prev = state.get("config")
    if not prev:
        return []
    man = lambda v: f"{v / 1e4:,.0f}万円"
    items = [
        ("運用資金", prev.get("capital"), cfg.capital, man),
        ("月間目標", prev.get("monthly_target"), cfg.monthly_target, man),
        ("建玉合計の上限", prev.get("exposure_cap", prev.get("capital", 0) * prev.get("max_leverage", 0)),
         cfg.exposure_cap, man),
        ("1 ルールの同時保有数", prev.get("max_positions", 5), cfg.max_positions, lambda v: f"{v:g} 銘柄"),
    ]
    details = [f"{name}: {fmt(old)} → {fmt(new)}" for name, old, new, fmt in items
               if old is not None and abs(float(old) - float(new)) > 1e-9]
    if not details:
        return []
    details.append(f"1 銘柄あたりの金額は約 {man(cfg.capital * cfg.rule_alloc / cfg.max_positions)}。"
                   "この前提で全ルールを検証し直した。")
    return [{"date": today, "family": "", "type": "設定変更", "title": "売買の前提（config.yaml）を変更", "details": details}]


def update_rules(ctx: Context, state: dict, families: list[str] | None = None,
                 today: str | None = None, workers: int | None = None, log=print) -> tuple[dict, list[dict]]:
    """全ファミリーを探索し直して台帳を更新する（週次）。"""
    cfg = ctx.cfg
    today = today or date.today().isoformat()
    oos_p = pd.Period(cfg.validation["oos_start"][:7], "M")
    current = {r["family"]: r for r in state.get("rules", [])}
    rejected_prev = {r["family"]: r for r in state.get("rejected", [])}
    new_rules, rejected, research_summary = [], [], []
    events = config_change_events(cfg, state, today)
    keys = families or list(FAMILIES)

    for key in FAMILIES:
        fam = FAMILIES[key]
        cur = current.get(key)
        if key not in keys:  # 今回は探索しないファミリーはそのまま引き継ぐ
            if cur:
                new_rules.append(cur)
            elif key in rejected_prev:
                rejected.append(rejected_prev[key])
            continue
        log(f"[{key}] {fam.name}: {len(fam.combos())} 通りを検証中...")
        r = research_family(ctx, fam, workers)
        grid, wf, cand = r.get("_grid"), r["walk_forward"], r["candidate"]
        cand_tier = cand["tier"] if cand else None

        cur_tier, cur_ev, cur_gates, cur_nb = None, None, None, None
        if cur:
            cur_ev = evaluate_rule(ctx, key, cur["params"], since=cur["adopted_on"])
            cur_nb = neighbor_pass_for(fam, grid, cur["params"], oos_p) if grid else cur.get("neighbor_pass")
            cur_gates = check_gates(cur_ev, wf, cur_nb, cfg)
            cur_tier = tier_of(cur_gates)

        same = bool(cur and cand and param_key(cur["params"]) == param_key(cand["params"]))
        replace = False
        if cand_tier and not same:
            if TIER_RANK[cand_tier] > TIER_RANK[cur_tier]:
                replace = True
            elif cur and cand_tier == cur_tier:
                margin = 1.0 + float(cfg.validation["replace_margin"])
                replace = rule_objective(cand["evidence"]) >= rule_objective(cur_ev) * margin

        if replace:
            version = (cur["version"] + 1) if cur else 1
            rule = _make_rule(key, cand["params"], cand_tier, cand["evidence"], cand["gates"], wf,
                              cand["neighbor_pass"], today, version)
            rule["evidence"]["forward"] = None
            if cur:
                title = f"{fam.name}: ルールを更新（v{cur['version']} → v{version}、{STATUS_LABEL[cand_tier]}）"
                details = param_diff(cur["params"], cand["params"])
                details.append("新: " + _evidence_line(cand["evidence"]))
                details.append("旧: " + _evidence_line(cur_ev))
                etype = "更新"
            else:
                title = f"{fam.name}: 新規に{STATUS_LABEL[cand_tier]}（v{version}）"
                details = [_evidence_line(cand["evidence"])]
                if cand_tier == "watch":
                    details.append("未達の基準: " + "、".join(_failed(cand["gates"])))
                etype = "新規"
            events.append({"date": today, "family": key, "type": etype, "title": title, "details": details})
            new_rules.append(rule)
        elif cur and cur_tier:
            rule = dict(cur)
            prev_tier = cur.get("tier", cur["status"] if cur["status"] != "suspended" else "active")
            rule.update({"tier": cur_tier, "evidence": cur_ev, "gates": cur_gates, "walk_forward": wf,
                         "neighbor_pass": cur_nb})
            if cur_tier != prev_tier:
                up = TIER_RANK[cur_tier] > TIER_RANK[prev_tier]
                rule["since"] = today
                events.append({
                    "date": today, "family": key, "type": "昇格" if up else "降格",
                    "title": f"{fam.name}: {STATUS_LABEL[prev_tier]} → {STATUS_LABEL[cur_tier]}",
                    "details": [_evidence_line(cur_ev)] + (
                        [] if up else ["未達の基準: " + "、".join(_failed(cur_gates))]),
                })
            new_rules.append(rule)
        elif cur:
            events.append({
                "date": today, "family": key, "type": "廃止",
                "title": f"{fam.name}: 基準を満たさなくなったため廃止（v{cur['version']}）",
                "details": ["未達の基準: " + "、".join(_failed(cur_gates)), _evidence_line(cur_ev)],
            })
            cur = None

        if not any(x["family"] == key for x in new_rules):
            rejected.append({
                "family": key, "combos_tested": r["combos_tested"],
                "params": cand["params"] if cand else None,
                "gates": cand["gates"] if cand else [],
                "evidence": {k: cand["evidence"][k] for k in ("is", "oos", "full")} if cand else None,
                "walk_forward": {k: v for k, v in wf.items() if k != "folds"},
                "note": r.get("note"),
            })
        research_summary.append({
            "family": key, "name": fam.name, "combos_tested": r["combos_tested"],
            "candidate_params": cand["params"] if cand else None,
            "candidate_tier": cand_tier,
            "gates": cand["gates"] if cand else [],
            "walk_forward": wf,
            "is": cand["evidence"]["is"] if cand else None,
            "oos": cand["evidence"]["oos"] if cand else None,
            "full": cand["evidence"]["full"] if cand else None,
        })

    state = dict(state)
    state["rules"] = new_rules
    state["rejected"] = rejected
    state["last_research"] = today
    write_json(RESEARCH_JSON, {"run_at": today, "data_through": ctx.panel.dates[-1],
                               "combos_tested": sum(x["combos_tested"] for x in research_summary),
                               "families": research_summary})
    state, ev2 = _apply_health(ctx, state, today, reevaluate=False)
    return state, events + ev2


def _apply_health(ctx: Context, state: dict, today: str, reevaluate: bool) -> tuple[dict, list[dict]]:
    """健康診断: 直近の成績が崩れた採用ルールを一時停止、持ち直したら再開。"""
    cfg = ctx.cfg
    events = []
    for rule in state["rules"]:
        fam = FAMILIES[rule["family"]]
        if reevaluate:
            ev = evaluate_rule(ctx, rule["family"], rule["params"], since=rule["adopted_on"])
            rule["evidence"] = ev
            rule["gates"] = check_gates(ev, rule.get("walk_forward"), rule.get("neighbor_pass"), cfg)
        elif rule["evidence"].get("forward") is None:
            rule["evidence"] = evaluate_rule(ctx, rule["family"], rule["params"], since=rule["adopted_on"])
        h = health_check(rule["evidence"], cfg)
        rule["health"] = h
        tier = rule.get("tier", "watch")
        want = tier if h["ok"] else "suspended"
        if want != rule["status"]:
            if want == "suspended":
                events.append({"date": today, "family": rule["family"], "type": "一時停止",
                               "title": f"{fam.name}: 直近の成績悪化により一時停止", "details": h["reasons"]})
            elif rule["status"] == "suspended":
                events.append({"date": today, "family": rule["family"], "type": "再開",
                               "title": f"{fam.name}: 成績が持ち直したため再開（{STATUS_LABEL[want]}）", "details": []})
            rule["status"] = want
            rule["since"] = today
    return state, events


def refresh_rules(ctx: Context, state: dict, today: str | None = None) -> tuple[dict, list[dict]]:
    """日次: 台帳のルールを最新データで採点し直す（探索はしない）。"""
    today = today or date.today().isoformat()
    events = config_change_events(ctx.cfg, state, today)
    state, ev = _apply_health(ctx, state, today, reevaluate=True)
    return state, events + ev


def finalize(ctx: Context, state: dict, events: list[dict], today: str | None = None) -> tuple[dict, dict]:
    """ポートフォリオ成績とシグナルを計算し、台帳・履歴を保存する。"""
    cfg = ctx.cfg
    today = today or date.today().isoformat()
    active = [r for r in state["rules"] if r["status"] == "active"]
    watch = [r for r in state["rules"] if r["status"] == "watch"]
    pf_active, sig_active = build_portfolio(ctx, active, "採用ルール")
    pf_watch, sig_watch = build_portfolio(ctx, watch, "観察中ルール（仮想売買）")
    state["portfolio"] = pf_active
    state["portfolio_watch"] = pf_watch
    state["updated_at"] = today
    state["data_through"] = ctx.panel.dates[-1].strftime("%Y-%m-%d")
    state["config"] = {
        "capital": cfg.capital, "monthly_target": cfg.monthly_target, "max_leverage": cfg.max_leverage,
        "exposure_cap": cfg.exposure_cap, "rule_exposure": cfg.capital * cfg.rule_alloc,
        "max_positions": cfg.max_positions,
        "costs": {"slippage_bps": cfg.costs.slippage_bps, "commission_bps": cfg.costs.commission_bps,
                  "long_interest_annual": cfg.costs.long_interest_annual,
                  "short_annual": cfg.costs.short_fee_annual + cfg.costs.short_extra_annual},
        "validation": cfg.validation, "portfolio": cfg.portfolio,
        "universe_size": len(ctx.panel.codes),
        "backtest_start": cfg.data["backtest_start"],
    }
    write_json(ACTIVE_RULES, state)
    signals = {"generated_at": today, "data_date": state["data_through"],
               "active": sig_active, "watch": sig_watch}
    append_history(events)
    return state, signals


def append_history(events: list[dict]) -> None:
    if not events:
        return
    hist = read_json(HISTORY_JSON, []) or []
    hist = clean(events) + hist
    write_json(HISTORY_JSON, hist)
    lines = ["# ルール更新履歴", "", "新しい順。ルールの採用・更新・停止・廃止はすべてここに記録されます。", ""]
    cur = None
    for e in hist:
        if e["date"] != cur:
            cur = e["date"]
            lines += [f"## {cur}", ""]
        lines.append(f"- **[{e['type']}] {e['title']}**")
        for d in e.get("details", []):
            lines.append(f"  - {d}")
    CHANGELOG.parent.mkdir(parents=True, exist_ok=True)
    CHANGELOG.write_text("\n".join(lines) + "\n", encoding="utf-8")
