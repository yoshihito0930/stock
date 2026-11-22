"""成績の指標。"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def monthly_pnl(equity: pd.Series) -> pd.Series:
    """累積損益（日次）→ 月ごとの損益（円）。"""
    if equity.empty:
        return pd.Series(dtype="float64")
    month_end = equity.groupby(equity.index.to_period("M")).last()
    return month_end.diff().fillna(month_end.iloc[0])


def max_drawdown(equity: pd.Series) -> float:
    """最大ドローダウン（円、正の値）。"""
    if equity.empty:
        return 0.0
    peak = np.maximum.accumulate(np.maximum(equity.to_numpy(), 0.0))
    return float((peak - equity.to_numpy()).max())


def trade_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_pnl": 0.0, "avg_ret": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "avg_days": 0.0, "total_pnl": 0.0}
    pnl = trades["pnl"].to_numpy()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gp, gl = float(wins.sum()), float(-losses.sum())
    return {
        "trades": int(n),
        "win_rate": float(len(wins) / n),
        "profit_factor": float(gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
        "avg_pnl": float(pnl.mean()),
        "avg_ret": float(trades["ret"].mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "avg_days": float(trades["days"].mean()),
        "total_pnl": float(pnl.sum()),
    }


def summarize(trades: pd.DataFrame, equity: pd.Series, capital: float, target: float) -> dict:
    """1 つの期間についての成績サマリー。"""
    m = monthly_pnl(equity)
    out = trade_stats(trades)
    n = len(m)
    mean = float(m.mean()) if n else 0.0
    std = float(m.std(ddof=1)) if n > 1 else 0.0
    dd = max_drawdown(equity) if len(equity) else 0.0
    out.update({
        "months": int(n),
        "monthly_mean": mean,
        "monthly_median": float(m.median()) if n else 0.0,
        "monthly_std": std,
        "monthly_worst": float(m.min()) if n else 0.0,
        "monthly_best": float(m.max()) if n else 0.0,
        "positive_months": float((m > 0).mean()) if n else 0.0,
        # 取引のなかった月（損益ゼロ）を除いた「動いた月」の中での勝率
        "active_months": int((m != 0).sum()) if n else 0,
        "positive_active_months": float((m > 0).sum() / max(1, (m != 0).sum())) if n else 0.0,
        "target_months": float((m >= target).mean()) if n else 0.0,
        "tstat": float(mean / (std / math.sqrt(n))) if std > 0 and n > 1 else 0.0,
        "max_drawdown": dd,
        "max_drawdown_pct": dd / capital if capital else 0.0,
        "annual_return_pct": mean * 12 / capital if capital else 0.0,
        "target_ratio": mean / target if target else 0.0,
    })
    return out


def window(trades: pd.DataFrame, equity: pd.Series, start: str | None, end: str | None):
    """期間で切り出す（取引は建てた日、損益曲線は日付で）。end は含まない。

    損益曲線は「期間開始時点を 0」に引き直す（それ以前の累積損益を混ぜない）。
    """
    t, e = trades, equity
    if start is not None:
        s = pd.Timestamp(start)
        t = t[t["entry_date"] >= s]
        before = e[e.index < s]
        base = float(before.iloc[-1]) if len(before) else 0.0
        e = e[e.index >= s] - base
    if end is not None:
        x = pd.Timestamp(end)
        t = t[t["entry_date"] < x]
        e = e[e.index < x]
    return t, e


def yearly_table(trades: pd.DataFrame, equity: pd.Series) -> list[dict]:
    rows = []
    if equity.empty:
        return rows
    m = monthly_pnl(equity)
    for year, grp in m.groupby(m.index.year):
        tr = trades[trades["entry_date"].dt.year == year]
        st = trade_stats(tr)
        rows.append({
            "year": int(year), "pnl": float(grp.sum()), "monthly_mean": float(grp.mean()),
            "positive_months": int((grp > 0).sum()), "months": int(len(grp)),
            "trades": st["trades"], "win_rate": st["win_rate"], "profit_factor": st["profit_factor"],
        })
    return rows
