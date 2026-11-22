import numpy as np
import pandas as pd
import pytest

from kabu.data.prices import drop_broken_history
from kabu.metrics import max_drawdown, monthly_pnl, summarize, window
from kabu.research import select_params, tier_of
from kabu.strategies import FAMILIES, param_key
from kabu.util import round_to_tick


def test_window_rebases_equity_so_prior_profit_is_not_counted():
    idx = pd.bdate_range("2021-11-01", "2022-03-31")
    eq = pd.Series(np.linspace(0, 1_000_000, len(idx)), index=idx)
    tr = pd.DataFrame({"entry_date": pd.to_datetime(["2021-12-01", "2022-02-01"]), "pnl": [1.0, 2.0],
                       "ret": [0.01, 0.01], "days": [1, 1]})
    t, e = window(tr, eq, "2022-01-01", None)
    assert len(t) == 1
    m = monthly_pnl(e)
    # 2022 年 1 月の損益に、2021 年までの累積が混ざらない
    assert m.iloc[0] < 300_000
    assert m.sum() == pytest.approx(eq.iloc[-1] - eq[eq.index < "2022-01-01"].iloc[-1])


def test_max_drawdown():
    eq = pd.Series([0, 10, 4, 12, -3, 5.0])
    assert max_drawdown(eq) == 15


def test_summarize_counts_flat_months_separately():
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    eq = pd.Series(0.0, index=idx)
    eq[eq.index >= "2024-02-01"] = 100.0     # 2 月だけ +100、他の月は取引なし
    tr = pd.DataFrame({"entry_date": pd.to_datetime(["2024-02-01"]), "pnl": [100.0], "ret": [0.01], "days": [1]})
    s = summarize(tr, eq, 1000, 50)
    assert s["positive_months"] == 0.25 and s["positive_active_months"] == 1.0 and s["active_months"] == 1


def test_tier_requires_core_gates():
    g = lambda core, ok: {"core": core, "ok": ok}
    assert tier_of([g(True, True), g(False, True)]) == "active"
    assert tier_of([g(True, True), g(False, False)]) == "watch"
    assert tier_of([g(True, False), g(False, True)]) is None


def test_selection_prefers_broadly_good_values_over_one_lucky_spike():
    fam = FAMILIES["breakdown_short"]
    months = pd.period_range("2015-01", "2018-12", freq="M")
    rng = np.random.default_rng(1)
    grid = {}
    for p in fam.combos():
        # lookback=60 の組み合わせは全体に少し良い。lookback=20 のうち 1 点だけ極端に良い（偶然の当たり）
        mean = 3000 if p["lookback"] == 60 else -1000
        pnl = rng.normal(mean, 4000, len(months))
        grid[param_key(p)] = {"params": p, "pnl": pnl, "gp": np.maximum(pnl, 0), "gl": np.maximum(-pnl, 0),
                              "n": np.full(len(months), 5), "months": months}
    lucky = next(k for k, r in grid.items() if r["params"]["lookback"] == 20)
    grid[lucky]["pnl"] = rng.normal(30000, 1000, len(months))
    sel = select_params(fam, grid, None, None)
    assert sel["params"]["lookback"] == 60


def test_selection_ignores_combos_with_too_few_trades():
    fam = FAMILIES["breakdown_short"]
    months = pd.period_range("2015-01", "2018-12", freq="M")
    grid = {param_key(p): {"params": p, "pnl": np.full(len(months), 1000.0) + np.arange(len(months)),
                           "gp": np.full(len(months), 1000.0), "gl": np.zeros(len(months)),
                           "n": np.zeros(len(months), dtype=int), "months": months} for p in fam.combos()}
    assert select_params(fam, grid, None, None) is None


def test_broken_history_is_dropped():
    idx = pd.bdate_range("2020-01-01", periods=600)
    px = pd.Series(100.0, index=idx)
    px.iloc[100:] = 10.0      # 未調整の分割のような -90% の段差
    df = pd.DataFrame({"open": px, "high": px * 1.01, "low": px * 0.99, "close": px, "volume": 1.0, "raw_close": px})
    out = drop_broken_history(df)
    assert out.index[0] == idx[100 + 260] and (out["close"] == 10.0).all()
    clean = df.iloc[:100]
    assert len(drop_broken_history(clean)) == 100


def test_tick_rounding():
    assert round_to_tick(1483.9, down=True) == 1483
    assert round_to_tick(3001, down=True) == 3000 and round_to_tick(3001, down=False) == 3005
    assert round_to_tick(12344, down=False) == 12350


def test_unfinished_intraday_bar_is_dropped():
    from kabu.data.prices import drop_unfinished_bar
    idx = pd.to_datetime(["2026-11-19", "2026-11-20"])
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=idx)
    assert len(drop_unfinished_bar(df, pd.Timestamp("2026-11-20 10:30"))) == 1   # 場中 → 当日足は使わない
    assert len(drop_unfinished_bar(df, pd.Timestamp("2026-11-20 16:45"))) == 2   # 大引け後 → 使う
    assert len(drop_unfinished_bar(df, pd.Timestamp("2026-11-21 09:00"))) == 2   # 翌日 → 前日足は確定済み
