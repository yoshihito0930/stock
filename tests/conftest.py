import numpy as np
import pandas as pd
import pytest

from kabu.config import Costs
from kabu.data.prices import Panel


def make_panel(bars: dict[str, list[tuple]], raw_scale: float = 1.0) -> Panel:
    """bars = {銘柄: [(始値, 高値, 安値, 終値), ...]} から Panel を作る。"""
    n = len(next(iter(bars.values())))
    idx = pd.bdate_range("2024-01-01", periods=n)
    def frame(k):
        return pd.DataFrame({c: [b[k] for b in v] for c, v in bars.items()}, index=idx, dtype="float64")
    close = frame(3)
    vol = pd.DataFrame(1e6, index=idx, columns=close.columns)
    return Panel(frame(0), frame(1), frame(2), close, vol, close * raw_scale,
                 pd.Series(np.linspace(100, 110, n), index=idx))


@pytest.fixture
def zero_costs():
    return Costs(0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.fixture
def costs():
    return Costs(commission_bps=0.0, slippage_bps=10.0, long_interest_annual=0.0365,
                 short_fee_annual=0.0365, short_extra_annual=0.0)
