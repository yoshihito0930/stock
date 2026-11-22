"""テクニカル指標（日付 × 銘柄 の横持ち DataFrame をまとめて計算する）。

すべて「その日の終値までの情報」だけで計算する。翌日の注文判断に使うので、
未来の値を参照する処理（shift(-1) など）は絶対に入れないこと。
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .data.prices import Panel


def sma(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).mean()


def rsi(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """ワイルダー式 RSI。"""
    diff = close.diff()
    up = diff.clip(lower=0.0)
    down = (-diff).clip(lower=0.0)
    avg_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    avg_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    total = avg_up + avg_down
    out = 100.0 * avg_up / total.where(total > 0)
    return out.where(close.notna())


def pct_change(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return close / close.shift(n) - 1.0


def rolling_high(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).max()


def rolling_low(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n, min_periods=n).min()


def atr_pct(panel: Panel, n: int = 14) -> pd.DataFrame:
    """ATR ÷ 終値（1 日の値動きの大きさ、比率）。"""
    prev = panel.close.shift(1)
    tr = np.maximum(panel.high - panel.low, np.maximum((panel.high - prev).abs(), (panel.low - prev).abs()))
    return tr.rolling(n, min_periods=n).mean() / panel.close


class Features:
    """指標のキャッシュ。同じ指標をパラメータ探索のたびに再計算しないためのもの。"""

    def __init__(self, panel: Panel, min_turnover: float, min_price: float):
        self.panel = panel
        self._cache: dict[tuple, object] = {}
        self.min_turnover = min_turnover
        self.min_price = min_price

    def _get(self, key: tuple, fn: Callable[[], object]):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ---- 個別銘柄 ----
    def sma(self, n: int) -> pd.DataFrame:
        return self._get(("sma", n), lambda: sma(self.panel.close, n))

    def rsi(self, n: int) -> pd.DataFrame:
        return self._get(("rsi", n), lambda: rsi(self.panel.close, n))

    def ret(self, n: int) -> pd.DataFrame:
        return self._get(("ret", n), lambda: pct_change(self.panel.close, n))

    def high_close(self, n: int) -> pd.DataFrame:
        """過去 n 日の終値の最高値（当日を含む）。"""
        return self._get(("hc", n), lambda: rolling_high(self.panel.close, n))

    def low_close(self, n: int) -> pd.DataFrame:
        return self._get(("lc", n), lambda: rolling_low(self.panel.close, n))

    def atr_pct(self, n: int = 14) -> pd.DataFrame:
        return self._get(("atr", n), lambda: atr_pct(self.panel, n))

    def turnover(self) -> pd.DataFrame:
        """売買代金（円）。調整済み価格×調整済み出来高 = 実際の売買代金。"""
        return self._get(("turnover",), lambda: self.panel.close * self.panel.volume)

    def volume_ratio(self, n: int = 20) -> pd.DataFrame:
        """当日の売買代金 ÷ 前日までの n 日平均。"""
        def f():
            tv = self.turnover()
            return tv / tv.shift(1).rolling(n, min_periods=n).mean()
        return self._get(("vr", n), f)

    def close_location(self) -> pd.DataFrame:
        """終値がその日の値幅のどこにあるか（1 = 高値引け、0 = 安値引け）。"""
        def f():
            rng = (self.panel.high - self.panel.low)
            return ((self.panel.close - self.panel.low) / rng.where(rng > 0)).fillna(0.5)
        return self._get(("cl",), f)

    def tradable(self) -> pd.DataFrame:
        """流動性と株価の条件を満たすか（直近 20 日の売買代金中央値で判定）。"""
        def f():
            med = self.turnover().rolling(20, min_periods=15).median()
            return (med >= self.min_turnover) & (self.panel.raw_close >= self.min_price) & self.panel.close.notna()
        return self._get(("tradable",), f)

    # ---- 相場全体 ----
    def index_above_sma(self, n: int) -> pd.Series:
        def f():
            ic = self.panel.index_close
            return (ic > ic.rolling(n, min_periods=n).mean()).reindex(self.panel.dates).fillna(False)
        return self._get(("idx_above", n), f)

    def index_rsi(self, n: int) -> pd.Series:
        def f():
            ic = self.panel.index_close.to_frame("i")
            return rsi(ic, n)["i"].reindex(self.panel.dates)
        return self._get(("idx_rsi", n), f)

    def market_mask(self, mode: str) -> np.ndarray:
        """相場環境フィルター。T×1 の bool 配列を返す。

        none        : 常に可
        up{n}       : 指数が n 日移動平均より上
        down{n}     : 指数が n 日移動平均より下
        """
        T = len(self.panel.dates)
        if mode in (None, "none"):
            return np.ones((T, 1), dtype=bool)
        if mode.startswith("up"):
            return self.index_above_sma(int(mode[2:])).to_numpy(dtype=bool).reshape(T, 1)
        if mode.startswith("down"):
            n = int(mode[4:])
            ic = self.panel.index_close
            below = (ic < ic.rolling(n, min_periods=n).mean()).reindex(self.panel.dates).fillna(False)
            return below.to_numpy(dtype=bool).reshape(T, 1)
        raise ValueError(f"unknown market mode: {mode}")
