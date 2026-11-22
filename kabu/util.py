"""小さな共通関数。"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def clean(obj: Any) -> Any:
    """JSON に書ける形へ変換する（numpy 型・無限大・NaN・日付を処理）。"""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x):
            return None
        if math.isinf(x):
            return 999.0 if x > 0 else -999.0
        return round(x, 6)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return clean(obj.tolist())
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def tick_size(price: float) -> float:
    """呼値の単位（通常銘柄の表。TOPIX500 銘柄はこれより細かいので、この刻みなら必ず発注できる）。"""
    for limit, tick in ((3000, 1), (5000, 5), (30000, 10), (50000, 50), (300000, 100), (500000, 500),
                        (3000000, 1000), (5000000, 5000), (30000000, 10000), (50000000, 50000)):
        if price <= limit:
            return float(tick)
    return 100000.0


def round_to_tick(price: float, down: bool) -> float:
    t = tick_size(price)
    q = price / t
    return (math.floor(q + 1e-9) if down else math.ceil(q - 1e-9)) * t


def yen(x: float | None, sign: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x:+,.0f}円" if sign else f"{x:,.0f}円"


def man(x: float | None, sign: bool = True, digits: int = 1) -> str:
    """円 → 「+3.2万円」の形。"""
    if x is None:
        return "-"
    v = x / 10000.0
    return f"{v:+,.{digits}f}万円" if sign else f"{v:,.{digits}f}万円"


def pct(x: float | None, digits: int = 0, sign: bool = False) -> str:
    if x is None:
        return "-"
    return f"{x * 100:+.{digits}f}%" if sign else f"{x * 100:.{digits}f}%"
