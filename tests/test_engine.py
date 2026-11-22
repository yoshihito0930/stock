"""エンジンが「現実に実行できる手順」だけを再現していることの確認。"""
import numpy as np
import pytest

from kabu.engine import Arrays, Setup, run_backtest
from tests.conftest import make_panel


def setup_for(panel, signal_days, **kw):
    T, N = panel.close.shape
    mask = np.zeros((T, N), dtype=bool)
    for t, i in signal_days:
        mask[t, i] = True
    base = dict(rule_id="t", side=1, entry_mask=mask, score=np.zeros((T, N)), max_hold=5, max_positions=5, alloc=1.0)
    base.update(kw)
    return Setup(**base)


FLAT = (100, 101, 99, 100)


def run(panel, setup, costs, capital=1_000_000, lev=2.0):
    return run_backtest(Arrays.from_panel(panel), [setup], capital, lev, costs)


def test_signal_is_executed_next_day_at_open(zero_costs):
    # 1 日目の終値でシグナル → 2 日目の始値で約定（シグナル当日の値段では買えない）
    p = make_panel({"A": [FLAT, FLAT, (110, 112, 109, 111), (111, 113, 110, 112), (112, 114, 111, 113)]})
    s = setup_for(p, [(1, 0)], max_hold=1)
    tr = run(p, s, zero_costs).trades
    assert len(tr) == 1
    assert tr.loc[0, "entry_px"] == 110            # 2 日目（index 2）の始値
    assert tr.loc[0, "exit_px"] == 112             # max_hold=1 → 翌日の引け
    assert str(tr.loc[0, "entry_date"].date()) == str(p.dates[2].date())


def test_no_trade_on_first_bar_and_no_fill_without_next_day(zero_costs):
    p = make_panel({"A": [FLAT, FLAT, FLAT]})
    s = setup_for(p, [(2, 0)])  # 最終日のシグナルは翌日がないので約定しない
    res = run(p, s, zero_costs)
    assert len(res.trades) == 0 and not res.open_positions
    assert len(res.next_orders) == 1  # 代わりに「次の営業日の注文」として出てくる


def test_limit_fills_only_when_price_trades_through(zero_costs):
    bars = [FLAT, FLAT, (100, 101, 97.0, 99), FLAT, FLAT]
    p = make_panel({"A": bars})
    # 指値 = 前日終値 100 の 3% 下 = 97。安値がちょうど 97 では約定しない（割り込んでいない）
    s = setup_for(p, [(1, 0)], entry_type="limit", limit_pct=0.03, max_hold=1)
    assert len(run(p, s, zero_costs).trades) == 0
    bars[2] = (100, 101, 96.5, 99)
    p = make_panel({"A": bars})
    s = setup_for(p, [(1, 0)], entry_type="limit", limit_pct=0.03, max_hold=1)
    tr = run(p, s, zero_costs).trades
    assert len(tr) == 1 and tr.loc[0, "entry_px"] == pytest.approx(97.0)


def test_limit_gap_down_fills_at_open_not_limit(zero_costs):
    # 寄付が指値より安い → 始値で約定（指値より有利な値段になる）
    p = make_panel({"A": [FLAT, FLAT, (95, 98, 94, 97), FLAT, FLAT]})
    s = setup_for(p, [(1, 0)], entry_type="limit", limit_pct=0.03, max_hold=1)
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "entry_px"] == pytest.approx(95.0)


def test_stop_loss_gap_fills_at_open(zero_costs):
    # 買値 100、損切り 95。翌日 90 で寄り付いたら 95 ではなく 90 で約定する
    p = make_panel({"A": [FLAT, FLAT, (100, 101, 99, 100), (90, 92, 88, 91), FLAT]})
    s = setup_for(p, [(1, 0)], stop_pct=0.05)
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "exit_px"] == pytest.approx(90.0)
    assert tr.loc[0, "reason"] == "損切り(寄付)"


def test_stop_wins_when_stop_and_target_hit_same_day(zero_costs):
    p = make_panel({"A": [FLAT, FLAT, (100, 100.5, 99.5, 100), (100, 112, 90, 105), FLAT]})
    s = setup_for(p, [(1, 0)], stop_pct=0.05, target_pct=0.10)
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "reason"] == "損切り" and tr.loc[0, "exit_px"] == pytest.approx(95.0)


def test_intraday_limit_entry_cannot_take_profit_on_earlier_high(zero_costs):
    # 場中に指値で拾った日は、高値が約定の前か後か分からない → 終値が利確値以上のときだけ利確
    p = make_panel({"A": [FLAT, FLAT, (100, 110, 96, 98), FLAT, FLAT]})
    s = setup_for(p, [(1, 0)], entry_type="limit", limit_pct=0.03, target_pct=0.05, max_hold=1)
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "reason"] == "期限(引け)"


def test_day_trade_exits_same_day_close(zero_costs):
    p = make_panel({"A": [FLAT, FLAT, (100, 104, 96, 103), FLAT]})
    s = setup_for(p, [(1, 0)], entry_type="limit", limit_pct=0.03, max_hold=0)
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "days"] == 0 and tr.loc[0, "exit_px"] == 103 and tr.loc[0, "entry_px"] == pytest.approx(97)


def test_exit_signal_executes_next_open(zero_costs):
    p = make_panel({"A": [FLAT, FLAT, (100, 101, 99, 100), (102, 103, 101, 102), (105, 106, 104, 105), FLAT]})
    s = setup_for(p, [(1, 0)], max_hold=10)
    s.exit_mask = np.zeros_like(s.entry_mask)
    s.exit_mask[3, 0] = True      # 4 日目の終値で決済条件 → 5 日目の始値で決済
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "exit_px"] == 105 and tr.loc[0, "reason"] == "決済シグナル"


def test_short_pnl_and_stop(zero_costs):
    p = make_panel({"A": [FLAT, FLAT, (100, 101, 99, 100), (100, 108, 99, 107), FLAT]})
    s = setup_for(p, [(1, 0)], side=-1, stop_pct=0.05)
    tr = run(p, s, zero_costs).trades
    assert tr.loc[0, "exit_px"] == pytest.approx(105.0) and tr.loc[0, "pnl"] < 0


def test_lot_rounding_and_leverage_cap(zero_costs):
    # 資金 100万・枠 1 倍・5 銘柄 → 1 銘柄 20万円。株価 1,500 円なら 100 株（15万円）
    p = make_panel({c: [(1500, 1510, 1490, 1500)] * 5 for c in "ABCDEF"})
    s = setup_for(p, [(1, i) for i in range(6)], max_hold=10)
    res = run(p, s, zero_costs)
    assert len(res.open_positions) == 5                       # 同時保有の上限
    assert all(pos.shares == 100 for pos in res.open_positions)
    # 株価が高すぎて 1 単元も買えない銘柄は見送る
    p = make_panel({"A": [(3000, 3010, 2990, 3000)] * 5})
    s = setup_for(p, [(1, 0)])
    assert len(run(p, s, zero_costs).open_positions) == 0
    # レバレッジ上限: 資金 100万 × 0.2 倍 = 20万円 → 15万円の建玉は 1 つしか持てない
    p = make_panel({c: [(1500, 1510, 1490, 1500)] * 5 for c in "AB"})
    s = setup_for(p, [(1, 0), (1, 1)], max_hold=10)
    assert len(run(p, s, zero_costs, lev=0.2).open_positions) == 1


def test_costs_reduce_pnl(costs, zero_costs):
    p = make_panel({"A": [FLAT, FLAT, (100, 101, 99, 100), (100, 101, 99, 100), FLAT]})
    s = setup_for(p, [(1, 0)], max_hold=1)
    free = run(p, s, zero_costs).trades.loc[0, "pnl"]
    paid = run(p, s, costs).trades.loc[0, "pnl"]
    notional = run(p, s, costs).trades.loc[0, "notional"]
    assert free == 0
    # 往復スリッページ 0.2% + 金利 1 日分 0.01%
    assert paid == pytest.approx(-notional * (0.002 + 0.0001), rel=0.02)


def test_untraded_day_blocks_fills(zero_costs):
    p = make_panel({"A": [FLAT, FLAT, FLAT, FLAT, FLAT]})
    p.volume.iloc[2, 0] = 0   # 値が付かなかった日
    s = setup_for(p, [(1, 0)])
    assert len(run(p, s, zero_costs).open_positions) == 0


def test_same_stock_is_not_bought_twice(zero_costs):
    p = make_panel({"A": [FLAT] * 6})
    s = setup_for(p, [(1, 0), (2, 0), (3, 0)], max_hold=10)
    assert len(run(p, s, zero_costs).open_positions) == 1


def test_exposure_cap_in_yen(zero_costs):
    # 建玉合計の上限 30万円: 15万円の建玉は 2 つまで。3 つ目の注文は出ない
    p = make_panel({c: [(1500, 1510, 1490, 1500)] * 5 for c in "ABC"})
    s = setup_for(p, [(1, 0), (1, 1), (1, 2)], max_hold=10)
    res = run_backtest(Arrays.from_panel(p), [s], 1_000_000, 2.0, zero_costs, max_exposure=300_000)
    assert len(res.open_positions) == 2
    assert sum(pos.notional for pos in res.open_positions) <= 300_000
    # 上限なしなら 3 つとも建つ
    res = run_backtest(Arrays.from_panel(p), [s], 1_000_000, 2.0, zero_costs)
    assert len(res.open_positions) == 3
