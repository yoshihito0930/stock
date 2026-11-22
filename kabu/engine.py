"""バックテストエンジン（日足・ポートフォリオ単位）。

現実に実行できる手順だけを再現する:
  - シグナルは「当日の終値」で判定し、注文は「翌営業日」に出す（先読みなし）。
  - 成行は翌日の始値で約定。指値は翌日の安値(高値)が指値を「割り込んだ(超えた)」ときだけ約定。
  - 損切りは逆指値。寄付で既に割り込んでいれば始値で約定（ギャップは不利な側に倒す）。
  - 同じ日に損切りと利確の両方に届いた場合は、損切りが先に付いたとみなす（保守的）。
  - 株数は単元（100株）単位に切り捨て。建玉合計は 資金×レバレッジ上限 を超えない。
  - 手数料・スリッページ・信用金利/貸株料を差し引く。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Costs
from .data.prices import Panel


@dataclass
class Arrays:
    """エンジン用の numpy 配列（T 日 × N 銘柄）。出来高ゼロの日は売買不可として NaN。"""

    dates: pd.DatetimeIndex
    codes: list[str]
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    RAW: np.ndarray  # 未調整終値（株数計算用）
    PC: np.ndarray   # その日時点で最後に付いた終値（調整済み。売買がない日は前の値を引き継ぐ）

    @classmethod
    def from_panel(cls, panel: Panel) -> "Arrays":
        traded = (panel.volume.to_numpy() > 0) & panel.close.notna().to_numpy()

        def tradable(df: pd.DataFrame) -> np.ndarray:
            a = df.to_numpy(dtype="float64", copy=True)
            a[~traded] = np.nan
            return a

        close_ff = panel.close.ffill()
        raw_ff = panel.raw_close.ffill()
        return cls(
            dates=panel.dates,
            codes=panel.codes,
            O=tradable(panel.open),
            H=tradable(panel.high),
            L=tradable(panel.low),
            C=tradable(panel.close),
            RAW=raw_ff.to_numpy(dtype="float64"),
            PC=close_ff.to_numpy(dtype="float64"),
        )


@dataclass
class Setup:
    """1 つの売買ルールをエンジンが実行できる形にしたもの。"""

    rule_id: str
    side: int                      # +1 = 買い、-1 = 空売り
    entry_mask: np.ndarray         # T×N bool。t 日の終値で条件成立 → t+1 日に注文
    score: np.ndarray              # T×N float。候補が多いとき大きい順に採用
    entry_type: str = "open"       # "open" = 寄成 / "limit" = 指値
    limit_pct: float = 0.0         # 指値: 前日終値からどれだけ有利な価格に置くか（比率）
    exit_mask: np.ndarray | None = None  # T×N bool。t 日の終値で成立 → t+1 日の寄付で決済
    stop_pct: float | None = None        # 損切り幅（約定値からの比率）
    target_pct: float | None = None      # 利確幅（約定値からの比率）
    max_hold: int = 5              # 最大保有日数（0 = 当日の引けで決済 = デイトレ）
    max_positions: int = 5         # 同時に持つ最大銘柄数
    alloc: float = 1.0             # このルールに割り当てる建玉枠（資金に対する倍率）
    allowed: np.ndarray | None = None    # N bool。売買してよい銘柄（空売り可能銘柄など）


@dataclass
class Position:
    setup_idx: int
    i: int               # 銘柄の列番号
    side: int
    t_in: int
    entry_px: float      # 調整済み価格
    shares: int
    scale: float         # 調整済み価格 → 実際の円価格 への倍率
    stop: float | None
    target: float | None
    intraday_entry: bool  # 寄付ではなく場中に約定した（その日の高値/安値の前後関係が不明）
    exit_next_open: bool = False
    exit_reason: str = ""
    last_px: float = 0.0

    @property
    def notional(self) -> float:
        return self.shares * self.entry_px * self.scale


@dataclass
class Order:
    """翌営業日に出す注文（シグナル表示にも使う）。"""

    setup_idx: int
    i: int
    side: int
    entry_type: str
    shares: int
    ref_close: float          # 前日終値（調整済み。最新日では実際の株価と同じ）
    limit_px: float | None    # 指値（調整済み）
    score: float


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series          # 日次の累積損益（円、含み損益込み）
    open_positions: list[Position] = field(default_factory=list)
    next_orders: list[Order] = field(default_factory=list)
    gross: pd.Series | None = None  # 日次の建玉合計（円）


TRADE_COLUMNS = [
    "rule_id", "code", "side", "entry_date", "exit_date", "entry_px", "exit_px",
    "shares", "notional", "pnl", "ret", "days", "reason",
]


class Simulator:
    def __init__(self, arr: Arrays, setups: list[Setup], capital: float, max_leverage: float,
                 costs: Costs, lot: int = 100):
        self.a = arr
        self.setups = setups
        self.capital = capital
        self.max_leverage = max_leverage
        self.costs = costs
        self.lot = lot
        self.positions: list[Position] = []
        self.realized = 0.0
        self.allow_entries = True
        self._trades: list[tuple] = []

    # ------------------------------------------------------------------
    def _close(self, p: Position, t: int, px: float, reason: str, slip: bool) -> None:
        a = self.a
        if slip:
            px = px * (1.0 - p.side * self.costs.side_cost)
        notional = p.notional
        gross = p.side * p.shares * p.scale * (px - p.entry_px)
        cal_days = max(1, (a.dates[t] - a.dates[p.t_in]).days)
        daily = self.costs.long_daily if p.side > 0 else self.costs.short_daily
        carry = notional * daily * cal_days
        pnl = gross - carry
        self.realized += pnl
        self._trades.append((
            self.setups[p.setup_idx].rule_id, a.codes[p.i], p.side, a.dates[p.t_in], a.dates[t],
            p.entry_px * p.scale, px * p.scale, p.shares, notional, pnl, pnl / notional,
            t - p.t_in, reason,
        ))

    def _base(self) -> float:
        """建玉計算の元になる資金。利益は再投資せず（単利）、損失が出たら縮小する。"""
        return max(0.0, min(self.capital, self.capital + self.realized))

    def plan_orders(self, t: int) -> list[Order]:
        """t 日に出す注文を決める（t-1 日の終値までの情報だけを使う）。

        exit_next_open が立っている建玉は t 日の寄付で決済される前提で枠を数える。
        """
        a = self.a
        if t <= 0:
            return []
        staying = [p for p in self.positions if not p.exit_next_open]
        held = {p.i for p in self.positions}
        base = self._base()
        gross = sum(p.notional for p in staying)
        cap = base * self.max_leverage
        orders: list[Order] = []
        for si, s in enumerate(self.setups):
            free = s.max_positions - sum(1 for p in staying if p.setup_idx == si)
            if free <= 0:
                continue
            row = s.entry_mask[t - 1]
            cand = np.flatnonzero(row)
            if cand.size == 0:
                continue
            sc = s.score[t - 1, cand]
            cand = cand[np.argsort(-sc, kind="stable")]
            budget = base * s.alloc / s.max_positions
            for i in cand:
                if free <= 0:
                    break
                i = int(i)
                if i in held or (s.allowed is not None and not s.allowed[i]):
                    continue
                raw = a.RAW[t - 1, i]
                pc = a.PC[t - 1, i]
                if not (raw > 0 and pc > 0):
                    continue
                shares = int(math.floor(budget / (raw * self.lot))) * self.lot
                if shares <= 0:
                    continue
                if gross + shares * raw > cap:
                    continue
                lim = pc * (1.0 - s.side * s.limit_pct) if s.entry_type == "limit" else None
                orders.append(Order(si, i, s.side, s.entry_type, shares, pc, lim, float(s.score[t - 1, i])))
                held.add(i)
                gross += shares * raw
                free -= 1
        return orders

    # ------------------------------------------------------------------
    def step(self, t: int) -> None:
        a, cost = self.a, self.costs.side_cost
        o, h, l, c = a.O[t], a.H[t], a.L[t], a.C[t]

        orders = self.plan_orders(t) if self.allow_entries else []

        # 1) 寄付での決済（前日の終値で決済条件が成立していた建玉）
        keep: list[Position] = []
        for p in self.positions:
            if p.exit_next_open and not np.isnan(o[p.i]):
                self._close(p, t, o[p.i], p.exit_reason, slip=True)
            else:
                keep.append(p)
        self.positions = keep

        # 2) 新規
        for od in orders:
            i, s = od.i, self.setups[od.setup_idx]
            op = o[i]
            if np.isnan(op):
                continue
            intraday = False
            if od.entry_type == "open":
                px = op * (1.0 + od.side * cost)
            elif od.side > 0:
                if not (l[i] < od.limit_px):   # 指値を「割り込んだ」ときだけ約定とみなす
                    continue
                px = min(op, od.limit_px)
                intraday = op > od.limit_px
            else:
                if not (h[i] > od.limit_px):
                    continue
                px = max(op, od.limit_px)
                intraday = op < od.limit_px
            scale = a.RAW[t - 1, i] / od.ref_close
            stop = px * (1.0 - od.side * s.stop_pct) if s.stop_pct else None
            target = px * (1.0 + od.side * s.target_pct) if s.target_pct else None
            self.positions.append(Position(od.setup_idx, i, od.side, t, px, od.shares, scale, stop, target,
                                           intraday, last_px=px))

        # 3) 場中の管理（損切り → 利確 → 期限 → 終値での決済判定）
        keep = []
        for p in self.positions:
            i, s = p.i, self.setups[p.setup_idx]
            ci = c[i]
            if np.isnan(ci):          # 売買が成立しなかった日は何もできない
                keep.append(p)
                continue
            new = p.t_in == t
            done = False
            if p.stop is not None:
                if p.side > 0:
                    if not new and o[i] <= p.stop:
                        self._close(p, t, o[i], "損切り(寄付)", True); done = True
                    elif l[i] <= p.stop:
                        self._close(p, t, p.stop, "損切り", True); done = True
                else:
                    if not new and o[i] >= p.stop:
                        self._close(p, t, o[i], "損切り(寄付)", True); done = True
                    elif h[i] >= p.stop:
                        self._close(p, t, p.stop, "損切り", True); done = True
            if not done and p.target is not None:
                if p.side > 0:
                    if not new and o[i] >= p.target:
                        self._close(p, t, o[i], "利確(寄付)", False); done = True
                    elif (ci >= p.target) if (new and p.intraday_entry) else (h[i] >= p.target):
                        self._close(p, t, p.target, "利確", False); done = True
                else:
                    if not new and o[i] <= p.target:
                        self._close(p, t, o[i], "利確(寄付)", False); done = True
                    elif (ci <= p.target) if (new and p.intraday_entry) else (l[i] <= p.target):
                        self._close(p, t, p.target, "利確", False); done = True
            if not done and (t - p.t_in) >= s.max_hold:
                self._close(p, t, ci, "期限(引け)", True); done = True
            if not done:
                p.last_px = ci
                if s.exit_mask is not None and s.exit_mask[t, i]:
                    p.exit_next_open = True
                    p.exit_reason = "決済シグナル"
                keep.append(p)
        self.positions = keep

    def unrealized(self) -> float:
        return sum(p.side * p.shares * p.scale * (p.last_px - p.entry_px) for p in self.positions)

    def run(self, start: int, end: int | None = None) -> Result:
        a = self.a
        end = len(a.dates) if end is None else end
        eq = np.zeros(end - start)
        gross = np.zeros(end - start)
        for t in range(start, end):
            self.step(t)
            eq[t - start] = self.realized + self.unrealized()
            gross[t - start] = sum(p.notional for p in self.positions)
        trades = pd.DataFrame(self._trades, columns=TRADE_COLUMNS)
        idx = a.dates[start:end]
        # 最終日の翌営業日に出す注文（= 今日のシグナル）
        next_orders = self.plan_orders(end) if end == len(a.dates) else []
        return Result(trades, pd.Series(eq, index=idx), list(self.positions), next_orders,
                      pd.Series(gross, index=idx))


def run_backtest(arr: Arrays, setups: list[Setup], capital: float, max_leverage: float, costs: Costs,
                 lot: int = 100, start: str | None = None, end: str | None = None) -> Result:
    t0 = int(arr.dates.searchsorted(pd.Timestamp(start))) if start else 1
    t0 = max(t0, 1)
    t1 = int(arr.dates.searchsorted(pd.Timestamp(end))) if end else None
    return Simulator(arr, setups, capital, max_leverage, costs, lot).run(t0, t1)


def simulate_single(arr: Arrays, setup: Setup, code: str, entry_date: pd.Timestamp, entry_px_yen: float,
                    shares: int, costs: Costs) -> dict | None:
    """実トレードの検証用: ある日に実際の値段で建てた 1 建玉を「ルール通りに」決済したらどうなったか。"""
    if code not in arr.codes:
        return None
    i = arr.codes.index(code)
    t = int(arr.dates.searchsorted(entry_date))
    if t >= len(arr.dates) or arr.dates[t] != entry_date or t == 0:
        return None
    ref = arr.PC[t, i] if not np.isnan(arr.C[t, i]) else np.nan
    raw = arr.RAW[t, i]
    if not (ref > 0 and raw > 0):
        return None
    scale = raw / ref
    px = entry_px_yen / scale
    sim = Simulator(arr, [setup], 1e12, 1e6, costs)
    sim.allow_entries = False
    stop = px * (1.0 - setup.side * setup.stop_pct) if setup.stop_pct else None
    target = px * (1.0 + setup.side * setup.target_pct) if setup.target_pct else None
    sim.positions = [Position(0, i, setup.side, t, px, shares, scale, stop, target, True, last_px=px)]
    for tt in range(t, len(arr.dates)):
        sim.step(tt)
        if sim._trades:
            return dict(zip(TRADE_COLUMNS, sim._trades[0]))
    return {"open": True, "pnl": sim.unrealized()}
