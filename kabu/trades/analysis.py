"""実トレードの分析。

見るもの:
  - 月ごとの実現損益と目標との差
  - 勝率・損益比・保有日数（勝ち/負け別）
  - ルール通りの取引か、ルール外（裁量）の取引か。それぞれの成績
  - ルール通りに建てた取引を「ルール通りに決済していたら」いくらだったか（裁量決済の損得）
  - 寄成で入るルールの実際の約定が、始値からどれだけ滑ったか → バックテストのコスト想定に反映
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import (LEDGER_CSV, ROUNDTRIPS_CSV, SLIPPAGE_OVERRIDE, TRADE_ANALYSIS)
from ..data.universe import load_universe
from ..engine import simulate_single
from ..research import Context
from ..rulebook import load_state
from ..strategies import FAMILIES
from ..util import write_json
from .importer import load_ledger
from .roundtrip import build_roundtrips

MIN_SLIPPAGE_SAMPLES = 20


def _stats(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "total_pnl": 0.0, "avg_win": 0.0,
                "avg_loss": 0.0, "payoff": 0.0, "avg_days_win": 0.0, "avg_days_loss": 0.0, "avg_pnl": 0.0}
    w, l = df[df["pnl"] > 0], df[df["pnl"] <= 0]
    gp, gl = float(w["pnl"].sum()), float(-l["pnl"].sum())
    avg_w = float(w["pnl"].mean()) if len(w) else 0.0
    avg_l = float(l["pnl"].mean()) if len(l) else 0.0
    return {
        "trades": int(n), "win_rate": len(w) / n,
        "profit_factor": gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0),
        "total_pnl": float(df["pnl"].sum()), "avg_pnl": float(df["pnl"].mean()),
        "avg_win": avg_w, "avg_loss": avg_l, "payoff": (avg_w / -avg_l) if avg_l < 0 else 0.0,
        "avg_days_win": float(w["days"].mean()) if len(w) else 0.0,
        "avg_days_loss": float(l["days"].mean()) if len(l) else 0.0,
    }


def match_rules(ctx: Context, rt: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """各取引が台帳のどのルールのシグナルに沿っていたかを調べる。"""
    arr = ctx.arr
    code_idx = {c: i for i, c in enumerate(arr.codes)}
    setups = [(r, ctx.setup(r["family"], r["params"], rule_id=r["id"])) for r in rules]
    out = rt.copy()
    out["rule_id"] = ""
    out["rule_pnl"] = np.nan
    out["entry_slip_bps"] = np.nan
    out["limit_missed"] = False
    for n, tr in rt.iterrows():
        i = code_idx.get(tr["code"])
        d = pd.Timestamp(tr["entry_date"])
        t = int(arr.dates.searchsorted(d))
        if i is None or t <= 0 or t >= len(arr.dates) or arr.dates[t] != d:
            continue
        side = 1 if tr["direction"] == "long" else -1
        for rule, s in setups:
            if s.side != side or not s.entry_mask[t - 1, i]:
                continue
            out.at[n, "rule_id"] = rule["id"]
            # 実際の円価格 → 調整済み価格 の換算
            scale = arr.RAW[t, i] / arr.PC[t, i] if arr.PC[t, i] > 0 else 1.0
            px_adj = tr["entry_price"] / scale
            op = arr.O[t, i]
            if s.entry_type == "open" and op > 0:
                out.at[n, "entry_slip_bps"] = side * (px_adj - op) / op * 1e4
            elif s.entry_type == "limit":
                lim = arr.PC[t - 1, i] * (1.0 - side * s.limit_pct)
                # 指値より不利な値段で建てている = ルールの指値を待てなかった
                out.at[n, "limit_missed"] = bool(side * (px_adj - lim) > 0.002 * lim)
            sim = simulate_single(arr, s, tr["code"], d, tr["entry_price"], int(tr["qty"]), ctx.cfg.costs)
            if sim and not sim.get("open"):
                out.at[n, "rule_pnl"] = sim["pnl"]
            break
    return out


def _insights(rt: pd.DataFrame, st: dict, monthly: pd.Series, target: float, by_rule: dict) -> list[str]:
    tips = []
    if len(monthly):
        mean = float(monthly.mean())
        tips.append(f"月平均の実現損益は {mean / 1e4:+.1f}万円（目標 {target / 1e4:.0f}万円の {max(mean, 0) / target:.0%}）。"
                    f"プラスの月は {int((monthly > 0).sum())}/{len(monthly)}。")
    if st["trades"] >= 10:
        if st["payoff"] > 0:
            need = 1.0 / (1.0 + st["payoff"])
            verdict = "上回っている" if st["win_rate"] > need else "下回っている"
            tips.append(f"平均利益 {st['avg_win'] / 1e4:.1f}万円 に対し平均損失 {-st['avg_loss'] / 1e4:.1f}万円"
                        f"（損益比 {st['payoff']:.2f}）。この損益比だと勝率 {need:.0%} が損益分岐点で、"
                        f"実際の勝率 {st['win_rate']:.0%} はそれを{verdict}。")
        if st["avg_days_loss"] > 1.5 * max(st["avg_days_win"], 0.5) and st["avg_days_loss"] >= 2:
            tips.append(f"負けた取引の平均保有 {st['avg_days_loss']:.1f}日 は、勝った取引の {st['avg_days_win']:.1f}日 より"
                        "明らかに長い。損切りが遅れて含み損を抱え続けている可能性が高い。")
        worst = float(rt["pnl"].min())
        if st["total_pnl"] != 0 and worst < 0 and -worst > 3 * abs(st["avg_win"]) > 0:
            tips.append(f"最大の負け 1 回（{worst / 1e4:+.1f}万円）が、平均的な勝ち {-worst / st['avg_win']:.1f} 回分を消している。"
                        "1 銘柄の金額か損切り位置を見直す余地がある。")
    on, off = by_rule.get("on"), by_rule.get("off")
    if on and off and (on["trades"] + off["trades"]) >= 10:
        share = on["trades"] / (on["trades"] + off["trades"])
        tips.append(f"ルール通りの取引は全体の {share:.0%}（{on['trades']} 回、損益 {on['total_pnl'] / 1e4:+.1f}万円）。"
                    f"ルール外の取引は {off['trades']} 回で損益 {off['total_pnl'] / 1e4:+.1f}万円。")
        if off["trades"] >= 5 and off["total_pnl"] < 0:
            tips.append("ルール外の取引が合計でマイナス。まずルール外の取引をやめるだけで成績が改善する。")
    return tips


def analyze(ctx: Context) -> dict | None:
    cfg = ctx.cfg
    ledger = load_ledger()
    if ledger.empty:
        return None
    rt, open_lots, unmatched = build_roundtrips(ledger)
    names = load_universe(cfg).set_index("code")["name"].to_dict()
    for df in (rt, open_lots):   # 手入力などで銘柄名が空のものは銘柄マスタから補う
        if len(df):
            df["name"] = [n if str(n).strip() else names.get(c, "") for c, n in zip(df["code"], df["name"])]
    state = load_state()
    rules = state.get("rules", [])
    if len(rt):
        rt = match_rules(ctx, rt, rules)
    else:
        rt = rt.assign(rule_id="", rule_pnl=np.nan, entry_slip_bps=np.nan, limit_missed=False)
    ROUNDTRIPS_CSV.parent.mkdir(parents=True, exist_ok=True)
    rt.to_csv(ROUNDTRIPS_CSV, index=False, encoding="utf-8")

    st = _stats(rt)
    monthly = pd.Series(dtype="float64")
    if len(rt):
        per = pd.to_datetime(rt["exit_date"]).dt.to_period("M")
        monthly = rt.groupby(per)["pnl"].sum()
        monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"), fill_value=0.0)

    on = rt[rt["rule_id"] != ""]
    off = rt[rt["rule_id"] == ""]
    by_rule = {"on": _stats(on), "off": _stats(off)}
    per_rule = []
    for rid, grp in on.groupby("rule_id"):
        s = _stats(grp)
        done = grp.dropna(subset=["rule_pnl"])
        s.update({
            "id": rid, "name": FAMILIES[rid].name if rid in FAMILIES else rid,
            "compared": int(len(done)),
            "actual_pnl_compared": float(done["pnl"].sum()) if len(done) else 0.0,
            "rule_pnl_compared": float(done["rule_pnl"].sum()) if len(done) else 0.0,
            "limit_missed": int(grp["limit_missed"].sum()),
        })
        per_rule.append(s)

    # 裁量決済の損得（ルール通りに決済していた場合との差）
    done = on.dropna(subset=["rule_pnl"])
    discretion = None
    if len(done):
        discretion = {"trades": int(len(done)), "actual": float(done["pnl"].sum()),
                      "by_rule": float(done["rule_pnl"].sum()),
                      "diff": float(done["pnl"].sum() - done["rule_pnl"].sum())}

    # スリッページ実測
    slips = on["entry_slip_bps"].dropna()
    slippage = None
    if len(slips):
        slippage = {"samples": int(len(slips)), "median_bps": float(slips.median()), "mean_bps": float(slips.mean())}
        if len(slips) >= MIN_SLIPPAGE_SAMPLES:
            write_json(SLIPPAGE_OVERRIDE, {"slippage_bps": max(0.0, float(slips.median())), "samples": int(len(slips)),
                                           "note": "実トレードの約定から測った片道スリッページ（中央値）。設定値より大きければバックテストに使われる。"})

    tips = _insights(rt, st, monthly, cfg.monthly_target, by_rule)
    if discretion and discretion["trades"] >= 5:
        d = discretion["diff"]
        tips.append(f"ルール通りに建てた {discretion['trades']} 回について、決済もルール通りにしていた場合との差は "
                    f"{d / 1e4:+.1f}万円（実際 {discretion['actual'] / 1e4:+.1f}万円 / ルール通り {discretion['by_rule'] / 1e4:+.1f}万円）。"
                    + ("自己判断の決済が成績を下げている。" if d < 0 else "自己判断の決済は今のところプラスに働いている。"))
    if slippage and slippage["samples"] >= 5:
        tips.append(f"寄成の約定は始値から中央値で {slippage['median_bps']:+.0f}bps ずれている（{slippage['samples']} 回）。"
                    f"バックテストの想定は片道 {cfg.costs.slippage_bps:.0f}bps。")

    result = {
        "generated_at": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "period": {"from": ledger["date"].min(), "to": ledger["date"].max()},
        "executions": int(len(ledger)), "unmatched_closes": int(unmatched),
        "overall": st,
        "monthly": [{"month": str(k), "pnl": float(v)} for k, v in monthly.items()],
        "monthly_mean": float(monthly.mean()) if len(monthly) else 0.0,
        "target": cfg.monthly_target,
        "by_direction": {d: _stats(rt[rt["direction"] == d]) for d in ("long", "short")},
        "by_holding": {
            "デイトレ（当日決済）": _stats(rt[rt["days"] == 0]),
            "1〜5日": _stats(rt[(rt["days"] >= 1) & (rt["days"] <= 5)]),
            "6日以上": _stats(rt[rt["days"] >= 6]),
        },
        "by_rule": by_rule, "per_rule": per_rule, "discretion": discretion, "slippage": slippage,
        "insights": tips,
        "open_lots": open_lots.to_dict("records"),
        "recent": rt.tail(30).iloc[::-1][["code", "name", "direction", "qty", "entry_date", "entry_price",
                                           "exit_date", "exit_price", "pnl", "days", "rule_id"]].to_dict("records"),
    }
    write_json(TRADE_ANALYSIS, result)
    return result


def analyze_if_any(ctx: Context) -> bool:
    if not LEDGER_CSV.exists():
        return False
    return analyze(ctx) is not None
