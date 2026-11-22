"""約定の列を「建てて → 決済した」1 往復の取引にまとめる（先入先出）。"""
from __future__ import annotations

from collections import defaultdict, deque

import pandas as pd

RT_COLUMNS = ["code", "name", "direction", "qty", "entry_date", "entry_price", "exit_date", "exit_price",
              "fee", "pnl", "ret", "days", "notional"]


def build_roundtrips(ledger: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(決済済みの往復取引, 未決済の建玉, 建てが見つからなかった決済の件数) を返す。"""
    lots: dict[tuple[str, str], deque] = defaultdict(deque)
    rows, unmatched = [], 0
    order = ledger.assign(_a=ledger["action"].map({"open": 0, "close": 1})).sort_values(
        ["date", "_a"], kind="stable")
    for r in order.itertuples(index=False):
        key = (r.code, r.direction)
        if r.action == "open":
            lots[key].append({"date": r.date, "price": r.price, "qty": r.qty,
                              "fee_per": (r.fee / r.qty) if r.qty else 0.0, "name": r.name})
            continue
        remaining = r.qty
        fee_per_close = (r.fee / r.qty) if r.qty else 0.0
        while remaining > 1e-9 and lots[key]:
            lot = lots[key][0]
            q = min(remaining, lot["qty"])
            sign = 1.0 if r.direction == "long" else -1.0
            fee = q * (lot["fee_per"] + fee_per_close)
            pnl = sign * q * (r.price - lot["price"]) - fee
            notional = q * lot["price"]
            rows.append({
                "code": r.code, "name": r.name or lot["name"], "direction": r.direction, "qty": q,
                "entry_date": lot["date"], "entry_price": lot["price"], "exit_date": r.date,
                "exit_price": r.price, "fee": fee, "pnl": pnl, "ret": pnl / notional if notional else 0.0,
                "days": int((pd.Timestamp(r.date) - pd.Timestamp(lot["date"])).days), "notional": notional,
            })
            lot["qty"] -= q
            remaining -= q
            if lot["qty"] <= 1e-9:
                lots[key].popleft()
        if remaining > 1e-9:
            unmatched += 1
    open_rows = [
        {"code": code, "name": lot["name"], "direction": direction, "qty": lot["qty"],
         "entry_date": lot["date"], "entry_price": lot["price"]}
        for (code, direction), dq in lots.items() for lot in dq
    ]
    rt = pd.DataFrame(rows, columns=RT_COLUMNS)
    if len(rt):
        # 同じ日に同じ値段で分割約定したものは 1 取引にまとめる
        keys = ["code", "name", "direction", "entry_date", "entry_price", "exit_date", "exit_price", "days"]
        rt = rt.groupby(keys, as_index=False, sort=False).agg(
            qty=("qty", "sum"), fee=("fee", "sum"), pnl=("pnl", "sum"), notional=("notional", "sum"))
        rt["ret"] = rt["pnl"] / rt["notional"]
        rt = rt[RT_COLUMNS].sort_values(["exit_date", "entry_date"], kind="stable").reset_index(drop=True)
    return rt, pd.DataFrame(open_rows, columns=["code", "name", "direction", "qty", "entry_date", "entry_price"]), unmatched
