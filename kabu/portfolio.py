"""複数ルールをまとめて運用したときの成績と、次の営業日に出す注文（シグナル）。"""
from __future__ import annotations

import pandas as pd

from .data.universe import load_universe
from .engine import Result
from .metrics import max_drawdown, monthly_pnl, summarize, trade_stats, yearly_table
from .research import Context, rule_objective
from .strategies import FAMILIES, MAX_POSITIONS
from .util import round_to_tick


def _run(ctx: Context, rules: list[dict], scale: float) -> Result:
    setups = []
    for r in rules:
        s = ctx.setup(r["family"], r["params"], rule_id=r["id"])
        s.alloc = scale
        setups.append(s)
    return ctx.backtest(setups)


def choose_scale(ctx: Context, rules: list[dict]) -> tuple[float, Result]:
    """過去検証の最大ドローダウンが上限に収まる範囲で、最も大きい建玉倍率を選ぶ。"""
    cfg = ctx.cfg
    limit = float(cfg.portfolio["max_drawdown_pct"])
    top = cfg.rule_alloc   # 1 ルールの枠が建玉上限を超えても意味がない
    scales = sorted({top} | {float(x) for x in cfg.portfolio["scales"] if float(x) < top}, reverse=True)
    res = None
    for k in scales:
        res = _run(ctx, rules, k)
        if max_drawdown(res.equity) / cfg.capital <= limit:
            return k, res
    return scales[-1], res


def build_portfolio(ctx: Context, rules: list[dict], label: str) -> tuple[dict, dict]:
    """ルール群をまとめて検証し、(成績の要約, シグナル) を返す。"""
    cfg = ctx.cfg
    if not rules:
        return ({"label": label, "rules": [], "empty": True},
                {"label": label, "orders": [], "positions": []})
    rules = sorted(rules, key=lambda r: -rule_objective(r["evidence"]))
    scale, res = choose_scale(ctx, rules)
    cap, tgt = cfg.capital, cfg.monthly_target
    summary = summarize(res.trades, res.equity, cap, tgt)
    m = monthly_pnl(res.equity)
    weekly = res.equity.resample("W-FRI").last().dropna()
    gross = res.gross / cap
    per_rule = []
    for r in rules:
        tr = res.trades[res.trades["rule_id"] == r["id"]]
        st = trade_stats(tr)
        st.update({"id": r["id"], "name": FAMILIES[r["family"]].name,
                   "monthly_mean": st["total_pnl"] / max(1, len(m))})
        per_rule.append(st)
    oos_start = cfg.validation["oos_start"]
    recent = m[m.index >= pd.Period(oos_start[:7], "M")]
    last12 = m.iloc[-12:]
    position_yen = cap * scale / MAX_POSITIONS
    portfolio = {
        "label": label,
        "empty": False,
        "rules": [r["id"] for r in rules],
        "scale": scale,
        "position_yen": position_yen,
        "summary": summary,
        "per_rule": per_rule,
        "yearly": yearly_table(res.trades, res.equity),
        "monthly": [{"month": str(k), "pnl": float(v)} for k, v in m.items()],
        "equity": [{"date": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in weekly.items()],
        "since_oos": {"start": oos_start, "monthly_mean": float(recent.mean()) if len(recent) else 0.0,
                      "positive_months": float((recent > 0).mean()) if len(recent) else 0.0},
        "last12": {"monthly_mean": float(last12.mean()) if len(last12) else 0.0,
                   "total": float(last12.sum()) if len(last12) else 0.0},
        "exposure": {"avg": float(gross.mean()), "max": float(gross.max()),
                     "days_invested": float((gross > 0).mean())},
        "required_capital": (cap * tgt / summary["monthly_mean"]) if summary["monthly_mean"] > 0 else None,
    }
    signals = make_signals(ctx, rules, res, scale, label)
    return portfolio, signals


def make_signals(ctx: Context, rules: list[dict], res: Result, scale: float, label: str) -> dict:
    """最終日の終値をもとに、次の営業日に出す注文と、保有中（モデル上）の建玉の扱いを一覧にする。"""
    arr = ctx.arr
    names = load_universe(ctx.cfg).set_index("code")["name"].to_dict()
    by_id = {r["id"]: r for r in rules}
    ids = [r["id"] for r in rules]
    last = len(arr.dates) - 1
    orders = []
    for od in res.next_orders:
        rule = by_id[ids[od.setup_idx]]
        p = rule["params"]
        code = arr.codes[od.i]
        close = float(arr.RAW[last, od.i])   # 最新日は調整済み価格 = 実際の株価
        item = {
            "rule_id": rule["id"], "rule_name": FAMILIES[rule["family"]].name, "tier": rule["status"],
            "code": code, "name": names.get(code, ""), "side": "買い" if od.side > 0 else "空売り",
            "shares": od.shares, "ref_close": close,
        }
        if od.entry_type == "limit":
            lim = round_to_tick(close * (1.0 - od.side * p["limit_pct"]), down=od.side > 0)
            item.update({"order": "指値", "limit_price": lim, "amount": lim * od.shares})
            base = lim
        else:
            item.update({"order": "寄成", "limit_price": None, "amount": close * od.shares})
            base = close
        stop = p.get("stop_pct")
        if stop:
            sp = round_to_tick(base * (1.0 - od.side * stop), down=od.side < 0)
            item["stop_pct"] = stop
            item["stop_price_est"] = sp
        item["max_hold"] = p.get("max_hold", 0)
        orders.append(item)
    positions = []
    for pos in res.open_positions:
        rule = by_id[ids[pos.setup_idx]]
        p = rule["params"]
        code = arr.codes[pos.i]
        held = last - pos.t_in
        left = p.get("max_hold", 0) - held
        now = float(arr.RAW[last, pos.i])
        entry_yen = pos.entry_px * pos.scale
        positions.append({
            "rule_id": rule["id"], "rule_name": FAMILIES[rule["family"]].name, "tier": rule["status"],
            "code": code, "name": names.get(code, ""), "side": "買い" if pos.side > 0 else "空売り",
            "entry_date": arr.dates[pos.t_in].strftime("%Y-%m-%d"), "entry_price": entry_yen,
            "shares": pos.shares, "days_held": held, "last_close": now,
            "unrealized": pos.side * pos.shares * pos.scale * (pos.last_px - pos.entry_px),
            "exit_next_open": pos.exit_next_open,
            "days_left": left,
            "stop_price": round_to_tick(pos.stop * pos.scale, down=pos.side < 0) if pos.stop else None,
        })
    return {
        "label": label,
        "data_date": arr.dates[last].strftime("%Y-%m-%d"),
        "scale": scale,
        "orders": orders,
        "positions": positions,
    }
