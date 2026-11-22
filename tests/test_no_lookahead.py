"""すべてのルールの型について、未来のデータを足しても過去のシグナルが変わらないことを確認する。"""
import numpy as np
import pandas as pd
import pytest

from kabu.data.prices import Panel
from kabu.indicators import Features
from kabu.strategies import FAMILIES, build_setup


def random_panel(T=420, N=12, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    cols = [f"S{i}" for i in range(N)]
    ret = rng.normal(0.0003, 0.03, (T, N))
    close = 1000 * np.exp(np.cumsum(ret, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.01, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, (T, N))))
    vol = rng.uniform(1e6, 5e6, (T, N))
    f = lambda a: pd.DataFrame(a, index=idx, columns=cols)
    index = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.012, T))), index=idx)
    return Panel(f(open_), f(high), f(low), f(close), f(vol), f(close), index)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_signals_do_not_change_when_future_data_is_added(family):
    full = random_panel()
    cut = 330
    part = full.slice(None, str(full.dates[cut - 1].date()))
    fam = FAMILIES[family]
    shortable = np.ones(len(full.codes), dtype=bool)
    combos = fam.combos()
    step = max(1, len(combos) // 12)
    hits = 0
    for params in combos[::step]:
        a = build_setup(family, params, Features(full, 1e8, 100), shortable)
        b = build_setup(family, params, Features(part, 1e8, 100), shortable)
        assert np.array_equal(a.entry_mask[:cut], b.entry_mask), params
        if a.exit_mask is not None:
            assert np.array_equal(a.exit_mask[:cut], b.exit_mask), params
        sel = b.entry_mask
        assert np.allclose(a.score[:cut][sel], b.score[sel]), params
        hits += int(sel.sum())
    assert hits > 0, "テストデータでシグナルが 1 回も出ていない（テストが空振り）"


def test_every_family_describes_every_combo():
    for fam in FAMILIES.values():
        for params in fam.combos():
            d = fam.describe(params)
            assert d["summary"] and d["screen"] and d["entry"] and d["exit"] and d["pick"]
            assert fam.neighbors(params)
