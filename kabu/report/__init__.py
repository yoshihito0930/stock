"""人が読む成果物の生成: RULES.md / SIGNALS.md / trades/REPORT.md / docs/index.html。"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from ..config import (ACTIVE_RULES, DOCS_DIR, HISTORY_JSON, RULES_MD, SIGNALS_JSON, SIGNALS_MD,
                      TRADE_ANALYSIS, TRADE_REPORT_MD, load_config)
from ..strategies import FAMILIES, MAX_POSITIONS
from ..util import clean, read_json
from . import texts
from .markdown import (build_rules_md, build_signals_md, build_trades_md, fmt_gate, headline,
                       next_business_day, position_yen)

TEMPLATE = Path(__file__).with_name("template.html")


def _rule_vm(state: dict, r: dict) -> dict:
    fam = FAMILIES[r["family"]]
    d = fam.describe(r["params"])
    ev = r["evidence"]
    table = [dict(label=label, **{k: ev[key][k] for k in ("trades", "win_rate", "profit_factor", "avg_ret",
                                                           "monthly_mean", "max_drawdown")})
             for label, key in (("学習期間", "is"), ("未使用データ", "oos"), ("全期間", "full"))]
    fw = ev.get("forward")
    if fw and fw["trades"] > 0:
        table.append(dict(label=f"台帳掲載後（{fw['since']}〜）",
                          **{k: fw[k] for k in ("trades", "win_rate", "profit_factor", "avg_ret",
                                                "monthly_mean", "max_drawdown")}))
    wf = r.get("walk_forward")
    yen_pos = position_yen(state, r)
    gates = []
    for g in r["gates"]:
        v, need = fmt_gate(g)
        gates.append({"name": g["name"], "value": v, "need": need, "ok": g["ok"]})
    return {
        "id": r["id"], "name": fam.name, "style": fam.style, "side": fam.side, "status": r["status"],
        "version": r["version"], "adopted_on": r["adopted_on"], "summary": d["summary"], "idea": fam.idea,
        "screen": d["screen"], "pick": d["pick"], "entry": d["entry"], "exit": d["exit"],
        "sizing": [f"1 銘柄あたり約 {yen_pos / 1e4:.0f}万円（100株単位で、この金額を超えない最大株数）。",
                   f"このルールで同時に持つのは最大 {MAX_POSITIONS} 銘柄。"],
        "table": table, "gates": gates, "yearly": ev["yearly"], "health": r["health"]["reasons"],
        "wf": (f"ウォークフォワード: PF {wf['profit_factor']:.2f}、{wf['years']} 年中 {wf['positive_years']} 年がプラス、"
               f"月平均 {wf['monthly_mean'] / 1e4:+.1f}万円。") if wf else None,
    }


def build_site_data(state: dict, signals: dict, history: list, analysis: dict | None, include_trades: bool) -> dict:
    cfg = state["config"]
    order = {"active": 0, "watch": 1, "suspended": 2}
    rules = sorted(state["rules"], key=lambda r: order[r["status"]])
    pf, pw = state.get("portfolio") or {}, state.get("portfolio_watch") or {}
    portfolios = []
    for p, title in ((pf, "採用ルールをまとめて運用した場合"), (pw, "観察中ルールをまとめて運用した場合（参考）")):
        if p.get("empty", True):
            continue
        portfolios.append({
            "title": title,
            "note": (f"対象: {'、'.join(FAMILIES[i].name for i in p['rules'])}。過去の最大ドローダウンが資金の "
                     f"{cfg['portfolio']['max_drawdown_pct']:.0%} に収まるよう、1 銘柄あたり約 {p['position_yen'] / 1e4:.0f}万円に調整。"
                     "手数料・スリッページ・金利を引いた後の数字。"),
            "summary": p["summary"], "yearly": p["yearly"], "monthly": p["monthly"], "equity": p["equity"],
            "last12": p["last12"]["monthly_mean"], "days_invested": p["exposure"]["days_invested"],
        })
    rejected = []
    for r in state["rejected"]:
        ev = r.get("evidence")
        rejected.append({
            "name": FAMILIES[r["family"]].name, "combos": r["combos_tested"],
            "pf": f"{ev['is']['profit_factor']:.2f}（学習）→ {ev['oos']['profit_factor']:.2f}（未使用データ）" if ev else "-",
            "failed": "、".join(g["name"] for g in r["gates"] if not g["ok"]) or (r.get("note") or "-"),
        })
    trades = None
    if analysis and include_trades:
        trades = dict(analysis)
        trades["recent"] = [dict(t, rule_name=FAMILIES[t["rule_id"]].name if t["rule_id"] in FAMILIES else "ルール外")
                            for t in analysis["recent"]]
    gen, dd = date.fromisoformat(signals["generated_at"]), date.fromisoformat(signals["data_date"])
    return clean({
        "meta": {"updated_at": state["updated_at"], "data_through": state["data_through"],
                 "last_research": state.get("last_research"), "next_day": next_business_day(signals["data_date"]),
                 "stale": (gen - dd).days > 4},
        "goal": {
            "target": cfg["monthly_target"],
            "expected": None if pf.get("empty", True) else pf["summary"]["monthly_mean"],
            "watch_expected": None if pw.get("empty", True) else pw["summary"]["monthly_mean"],
            "n_active": sum(1 for r in rules if r["status"] == "active"),
            "n_watch": sum(1 for r in rules if r["status"] == "watch"),
        },
        "headline": [x[2:] if x.startswith("- ") else x for x in headline(state)],
        "signals": {"active": signals["active"], "watch": signals["watch"]},
        "suspended": [f"{FAMILIES[r['family']].name}: " + " / ".join(r["health"]["reasons"])
                      for r in rules if r["status"] == "suspended"],
        "routine": texts.routine(), "common": texts.common(cfg), "method": texts.method(cfg),
        "import_howto": texts.import_howto(),
        "rules": [_rule_vm(state, r) for r in rules],
        "portfolios": portfolios, "rejected": rejected,
        "trades": trades, "trades_hidden": bool(analysis) and not include_trades,
        "history": history[:60],
    })


def build_site(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload)


def build_all() -> None:
    state = read_json(ACTIVE_RULES)
    if not state or not state.get("config"):
        raise SystemExit("台帳がありません。先に `python -m kabu research` を実行してください。")
    signals = read_json(SIGNALS_JSON)
    history = read_json(HISTORY_JSON, []) or []
    analysis = read_json(TRADE_ANALYSIS)
    cfg = load_config()
    RULES_MD.write_text(build_rules_md(state, history), encoding="utf-8")
    SIGNALS_MD.write_text(build_signals_md(state, signals), encoding="utf-8")
    TRADE_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    TRADE_REPORT_MD.write_text(build_trades_md(analysis), encoding="utf-8")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    data = build_site_data(state, signals, history, analysis, bool(cfg.site.get("include_real_trades", True)))
    (DOCS_DIR / "index.html").write_text(build_site(data), encoding="utf-8")
    (DOCS_DIR / ".nojekyll").write_text("", encoding="utf-8")
