"""売買ルールの「型」（ファミリー）の定義。

各ファミリーは 3 つを提供する:
  - grid()      : 探索するパラメータの候補
  - build()     : パラメータ → エンジンが実行できる Setup
  - describe()  : パラメータ → 人が読んで実行できる日本語のルール文

新しい型を足すときは Family を 1 つ書いて FAMILIES に登録するだけでよい。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .engine import Setup
from .indicators import Features

Params = dict[str, Any]

MAX_POSITIONS = 5   # 1 ルールあたりの同時保有銘柄数
ALLOC = 1.0         # 1 ルールあたりの建玉枠（資金の何倍まで）
INDEX_NAME = "日経平均"


@dataclass(frozen=True)
class Family:
    key: str
    name: str            # 表示名
    style: str           # "スイング" / "デイトレ"
    side: int
    grid: dict[str, list]
    build: Callable[[Features, Params, np.ndarray | None], Setup]
    describe: Callable[[Params], dict]
    idea: str            # なぜ利益になると考えるか（仮説）

    def combos(self) -> list[Params]:
        keys = list(self.grid)
        return [dict(zip(keys, vals)) for vals in itertools.product(*(self.grid[k] for k in keys))]

    def neighbors(self, params: Params) -> list[Params]:
        """1 つのパラメータだけを隣の値にずらした組み合わせ。"""
        out = []
        for k, values in self.grid.items():
            idx = values.index(params[k])
            for j in (idx - 1, idx + 1):
                if 0 <= j < len(values):
                    q = dict(params)
                    q[k] = values[j]
                    out.append(q)
        return out


def param_key(params: Params) -> str:
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def _np(df) -> np.ndarray:
    return df.to_numpy()


def _pct(x: float) -> str:
    v = x * 100
    return f"{v:.0f}%" if abs(v - round(v)) < 1e-9 else f"{v:.1f}%"


def _market_text(mode: str) -> str | None:
    if mode in (None, "none"):
        return None
    if mode.startswith("up"):
        return f"相場全体が上向きのときだけ売買する（{INDEX_NAME}が {mode[2:]} 日移動平均より上）。下回っている間は新規の買いを休む。"
    if mode.startswith("down"):
        return f"相場全体が下向きのときだけ売買する（{INDEX_NAME}が {mode[4:]} 日移動平均より下）。上回っている間は新規の売りを休む。"
    return None


def _common_universe() -> list[str]:
    return [
        "対象は TOPIX500 と TOPIX Small 1 の構成銘柄（約1,000銘柄）。",
        "直近 20 日の売買代金（中央値）が 5 億円以上で、株価 200 円以上の銘柄だけ。",
    ]


# ----------------------------------------------------------------------
# 1) 押し目買い（スイング）
# ----------------------------------------------------------------------
def _build_dip_swing(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    rsi = _np(f.rsi(p["rsi_n"]))
    mask = _np(f.tradable()) & (rsi < p["rsi_th"])
    if p["trend"]:
        mask &= close > _np(f.sma(p["trend"]))
    if p["mom"] > 0:
        mask &= _np(f.ret(120)) >= p["mom"]
    mask &= f.market_mask(p["market"])
    exit_mask = close > _np(f.sma(5))
    return Setup(
        rule_id="dip_swing", side=1, entry_mask=mask, score=np.nan_to_num(-rsi, nan=-1e9),
        entry_type="limit" if p["limit_pct"] > 0 else "open", limit_pct=p["limit_pct"],
        exit_mask=exit_mask, stop_pct=p["stop_pct"], target_pct=None, max_hold=p["max_hold"],
        max_positions=MAX_POSITIONS, alloc=ALLOC,
    )


def _describe_dip_swing(p: Params) -> dict:
    cond = [f"RSI({p['rsi_n']}日) が {p['rsi_th']} 未満まで売り込まれている（短期的な売られすぎ）。"]
    if p["mom"] > 0:
        cond.insert(0, f"直近 120 営業日（約半年）で {_pct(p['mom'])} 以上値上がりしている（中期で強い銘柄）。")
    if p["trend"]:
        cond.insert(0, f"終値が {p['trend']} 日移動平均より上（長期では上昇トレンド）。")
    m = _market_text(p["market"])
    if p["limit_pct"] > 0:
        entry = [
            f"翌営業日の朝、寄付前に「前日終値より {_pct(p['limit_pct'])} 安い価格」で買いの指値注文（当日限り）を出す。",
            "その日に約定しなければ注文は失効。追いかけて買わない。",
        ]
    else:
        entry = ["翌営業日の寄付に成行（寄成）で買う。"]
    exits = [
        "終値が 5 日移動平均を上回ったら、翌営業日の寄付に成行で売る（反発を確認して降りる）。",
        f"買ってから {p['max_hold']} 営業日目の引けまでに上の条件にならなければ、引け成行で売る（時間切れ）。",
    ]
    if p["stop_pct"]:
        exits.insert(0, f"約定したらすぐ「買値より {_pct(p['stop_pct'])} 下」に逆指値の売り（損切り）を入れる。")
    return {
        "summary": "上昇基調の銘柄が短期的に売られすぎたところを買い、数日の反発で売る。",
        "screen": _common_universe() + cond + ([m] if m else []),
        "pick": f"条件に合う銘柄が多い日は、RSI が低い（より売られている）順に最大 {MAX_POSITIONS} 銘柄まで。",
        "entry": entry,
        "exit": exits,
    }


# ----------------------------------------------------------------------
# 2) 急落拾いデイトレ（指値で拾って当日引けで手仕舞い）
# ----------------------------------------------------------------------
def _build_dip_day(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    rsi = _np(f.rsi(p["rsi_n"]))
    mask = _np(f.tradable()) & (rsi < p["rsi_th"])
    if p["trend"]:
        mask &= close > _np(f.sma(p["trend"]))
    mask &= f.market_mask(p["market"])
    return Setup(
        rule_id="dip_day", side=1, entry_mask=mask, score=np.nan_to_num(-rsi, nan=-1e9),
        entry_type="limit", limit_pct=p["limit_pct"], exit_mask=None, stop_pct=None, target_pct=None,
        max_hold=0, max_positions=MAX_POSITIONS, alloc=ALLOC,
    )


def _describe_dip_day(p: Params) -> dict:
    cond = []
    if p["trend"]:
        cond.append(f"終値が {p['trend']} 日移動平均より上（長期では上昇トレンド）。")
    if p["rsi_th"] < 100:
        cond.append(f"RSI({p['rsi_n']}日) が {p['rsi_th']} 未満（すでに短期的に売られている）。")
    m = _market_text(p["market"])
    return {
        "summary": "売られている銘柄が翌日さらに大きく下げた瞬間を指値で拾い、その日の引けで必ず手仕舞う。持ち越さない。",
        "screen": _common_universe() + cond + ([m] if m else []),
        "pick": f"条件に合う銘柄のうち RSI が低い順に最大 {MAX_POSITIONS} 銘柄に注文を出す。",
        "entry": [
            f"翌営業日の朝、寄付前に「前日終値より {_pct(p['limit_pct'])} 安い価格」で買いの指値注文（当日限り）を出す。",
            "約定しなければ何もしない（大半の日は約定しない）。",
        ],
        "exit": [
            "約定した日の大引けで成行（引成）で売る。昼休みか 15 時頃に約定を確認して引成の売り注文を入れる。",
            "損切りの逆指値は置かない（当日中に必ず決済するため）。",
        ],
    }


# ----------------------------------------------------------------------
# 3) 急騰の逆張り空売り（スイング）
# ----------------------------------------------------------------------
def _build_rip_short(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    rsi = _np(f.rsi(2))
    mask = _np(f.tradable()) & (rsi > p["rsi_th"])
    if p["surge"] > 0:
        mask &= _np(f.ret(5)) >= p["surge"]
    if p["trend"]:
        mask &= close < _np(f.sma(200))
    mask &= f.market_mask(p["market"])
    exit_mask = close < _np(f.sma(5))
    return Setup(
        rule_id="rip_short", side=-1, entry_mask=mask, score=np.nan_to_num(_np(f.ret(5)), nan=-1e9),
        entry_type="limit" if p["limit_pct"] > 0 else "open", limit_pct=p["limit_pct"],
        exit_mask=exit_mask, stop_pct=p["stop_pct"], target_pct=None, max_hold=p["max_hold"],
        max_positions=MAX_POSITIONS, alloc=ALLOC, allowed=shortable,
    )


def _describe_rip_short(p: Params) -> dict:
    cond = [f"RSI(2日) が {p['rsi_th']} を超えている（短期的な買われすぎ）。"]
    if p["surge"] > 0:
        cond.append(f"直近 5 営業日で {_pct(p['surge'])} 以上急騰している。")
    if p["trend"]:
        cond.append("終値が 200 日移動平均より下（長期では下降トレンドの中の急騰）。")
    m = _market_text(p["market"])
    if p["limit_pct"] > 0:
        entry = [
            f"翌営業日の朝、寄付前に「前日終値より {_pct(p['limit_pct'])} 高い価格」で信用新規売りの指値注文（当日限り）を出す。",
            "その日に約定しなければ注文は失効。",
        ]
    else:
        entry = ["翌営業日の寄付に成行（寄成）で信用新規売り。"]
    return {
        "summary": "短期間で急騰しすぎた銘柄を空売りし、数日の反落で買い戻す。",
        "screen": _common_universe() + ["空売りできる銘柄（貸借銘柄）だけ。発注画面で「売建可」を確認する。"] + cond + ([m] if m else []),
        "pick": f"条件に合う銘柄が多い日は、5 日間の上昇率が大きい順に最大 {MAX_POSITIONS} 銘柄まで。",
        "entry": entry,
        "exit": [
            f"約定したらすぐ「売値より {_pct(p['stop_pct'])} 上」に逆指値の買い戻し（損切り）を入れる。空売りは損失が青天井なので必須。",
            "終値が 5 日移動平均を下回ったら、翌営業日の寄付に成行で買い戻す。",
            f"売ってから {p['max_hold']} 営業日目の引けまでに上の条件にならなければ、引け成行で買い戻す（時間切れ）。",
        ],
    }


# ----------------------------------------------------------------------
# 4) 高値ブレイク順張り（スイング）
# ----------------------------------------------------------------------
def _build_breakout_long(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    vr = _np(f.volume_ratio(20))
    mask = _np(f.tradable()) & (close >= _np(f.high_close(p["lookback"]))) & (vr >= p["vol_ratio"])
    mask &= _np(f.close_location()) >= 0.5
    mask &= f.market_mask(p["market"])
    exit_mask = close < _np(f.sma(p["exit_sma"]))
    return Setup(
        rule_id="breakout_long", side=1, entry_mask=mask, score=np.nan_to_num(vr, nan=-1e9),
        entry_type="open", exit_mask=exit_mask, stop_pct=p["stop_pct"], target_pct=None,
        max_hold=p["max_hold"], max_positions=MAX_POSITIONS, alloc=ALLOC,
    )


def _describe_breakout_long(p: Params) -> dict:
    m = _market_text(p["market"])
    cond = [
        f"終値が過去 {p['lookback']} 営業日の終値の最高値を更新した。",
        f"その日の売買代金が、直前 20 日平均の {p['vol_ratio']:g} 倍以上（出来高を伴ったブレイク）。",
        "終値がその日の値幅の上半分で引けている（上ヒゲで終わっていない）。",
    ]
    exits = [
        f"終値が {p['exit_sma']} 日移動平均を下回ったら、翌営業日の寄付に成行で売る。",
        f"買ってから {p['max_hold']} 営業日目の引けで、まだ持っていれば引け成行で売る（時間切れ）。",
    ]
    if p["stop_pct"]:
        exits.insert(0, f"約定したらすぐ「買値より {_pct(p['stop_pct'])} 下」に逆指値の売り（損切り）を入れる。")
    return {
        "summary": "出来高を伴って高値を更新した銘柄に乗り、勢いが続く間だけ持つ。",
        "screen": _common_universe() + cond + ([m] if m else []),
        "pick": f"条件に合う銘柄が多い日は、売買代金の増加倍率が大きい順に最大 {MAX_POSITIONS} 銘柄まで。",
        "entry": ["翌営業日の寄付に成行（寄成）で買う。"],
        "exit": exits,
    }


# ----------------------------------------------------------------------
# 5) 安値ブレイク順張り空売り（スイング）
# ----------------------------------------------------------------------
def _build_breakdown_short(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    vr = _np(f.volume_ratio(20))
    mask = _np(f.tradable()) & (close <= _np(f.low_close(p["lookback"]))) & (vr >= p["vol_ratio"])
    mask &= _np(f.close_location()) <= 0.5
    mask &= f.market_mask(p["market"])
    exit_mask = close > _np(f.sma(p["exit_sma"]))
    return Setup(
        rule_id="breakdown_short", side=-1, entry_mask=mask, score=np.nan_to_num(vr, nan=-1e9),
        entry_type="open", exit_mask=exit_mask, stop_pct=p["stop_pct"], target_pct=None,
        max_hold=p["max_hold"], max_positions=MAX_POSITIONS, alloc=ALLOC, allowed=shortable,
    )


def _describe_breakdown_short(p: Params) -> dict:
    m = _market_text(p["market"])
    return {
        "summary": "出来高を伴って安値を割り込んだ銘柄を空売りし、下げの勢いが続く間だけ持つ。",
        "screen": _common_universe() + [
            "空売りできる銘柄（貸借銘柄）だけ。発注画面で「売建可」を確認する。",
            f"終値が過去 {p['lookback']} 営業日の終値の最安値を更新した。",
            f"その日の売買代金が、直前 20 日平均の {p['vol_ratio']:g} 倍以上。",
            "終値がその日の値幅の下半分で引けている。",
        ] + ([m] if m else []),
        "pick": f"条件に合う銘柄が多い日は、売買代金の増加倍率が大きい順に最大 {MAX_POSITIONS} 銘柄まで。",
        "entry": ["翌営業日の寄付に成行（寄成）で信用新規売り。"],
        "exit": [
            f"約定したらすぐ「売値より {_pct(p['stop_pct'])} 上」に逆指値の買い戻し（損切り）を入れる。",
            f"終値が {p['exit_sma']} 日移動平均を上回ったら、翌営業日の寄付に成行で買い戻す。",
            f"売ってから {p['max_hold']} 営業日目の引けで、まだ持っていれば引け成行で買い戻す（時間切れ）。",
        ],
    }


# ----------------------------------------------------------------------
# 6) 好材料の急伸に順張り（スイング）
# ----------------------------------------------------------------------
def _build_news_long(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    r1 = _np(f.ret(1))
    vr = _np(f.volume_ratio(20))
    mask = _np(f.tradable()) & (r1 >= p["jump"]) & (vr >= p["vol_ratio"])
    mask &= _np(f.close_location()) >= 0.5
    if p["trend"]:
        mask &= close > _np(f.sma(200))
    mask &= f.market_mask(p["market"])
    return Setup(
        rule_id="news_long", side=1, entry_mask=mask, score=np.nan_to_num(vr, nan=-1e9),
        entry_type="open", exit_mask=None, stop_pct=p["stop_pct"], target_pct=None,
        max_hold=p["max_hold"], max_positions=MAX_POSITIONS, alloc=ALLOC,
    )


def _describe_news_long(p: Params) -> dict:
    m = _market_text(p["market"])
    cond = [
        f"前日比 {_pct(p['jump'])} 以上の急伸で引けた（決算・上方修正などの好材料が出た日を想定）。",
        f"その日の売買代金が、直前 20 日平均の {p['vol_ratio']:g} 倍以上。",
        "終値がその日の値幅の上半分で引けている（寄り天で終わっていない）。",
    ]
    if p["trend"]:
        cond.append("終値が 200 日移動平均より上。")
    exits = [f"買ってから {p['max_hold']} 営業日目の引けに成行（引成）で売る。途中の値動きでは動かない。"]
    if p["stop_pct"]:
        exits.insert(0, f"約定したらすぐ「買値より {_pct(p['stop_pct'])} 下」に逆指値の売り（損切り）を入れる。")
    return {
        "summary": "好材料で出来高を伴って急伸した銘柄を翌朝に買い、評価が織り込まれるまでの数日〜数週間を取る。",
        "screen": _common_universe() + cond + ([m] if m else []),
        "pick": f"条件に合う銘柄が多い日は、売買代金の増加倍率が大きい順に最大 {MAX_POSITIONS} 銘柄まで。",
        "entry": ["翌営業日の寄付に成行（寄成）で買う。"],
        "exit": exits,
    }


# ----------------------------------------------------------------------
# 7) 全体急落時のリバウンド狙い（スイング）
# ----------------------------------------------------------------------
def _build_panic_dip(f: Features, p: Params, shortable: np.ndarray | None) -> Setup:
    close = _np(f.panel.close)
    rsi = _np(f.rsi(2))
    T = close.shape[0]
    idx_rsi = f.index_rsi(2).to_numpy().reshape(T, 1)
    mask = _np(f.tradable()) & (rsi < p["rsi_th"]) & (idx_rsi < p["index_rsi"])
    if p["trend"]:
        mask &= close > _np(f.sma(200))
    exit_mask = close > _np(f.sma(5))
    return Setup(
        rule_id="panic_dip", side=1, entry_mask=mask, score=np.nan_to_num(-rsi, nan=-1e9),
        entry_type="limit" if p["limit_pct"] > 0 else "open", limit_pct=p["limit_pct"],
        exit_mask=exit_mask, stop_pct=None, target_pct=None, max_hold=p["max_hold"],
        max_positions=MAX_POSITIONS, alloc=ALLOC,
    )


def _describe_panic_dip(p: Params) -> dict:
    cond = [
        f"{INDEX_NAME}の RSI(2日) が {p['index_rsi']} 未満（相場全体が短期的に売り込まれている日）。",
        f"個別銘柄の RSI(2日) が {p['rsi_th']} 未満。",
    ]
    if p["trend"]:
        cond.append("終値が 200 日移動平均より上（長期では上昇トレンドの銘柄だけ）。")
    if p["limit_pct"] > 0:
        entry = [
            f"翌営業日の朝、寄付前に「前日終値より {_pct(p['limit_pct'])} 安い価格」で買いの指値注文（当日限り）を出す。",
            "その日に約定しなければ注文は失効。",
        ]
    else:
        entry = ["翌営業日の寄付に成行（寄成）で買う。"]
    return {
        "summary": "相場全体が投げ売りになった日に、売られすぎた銘柄を拾って数日のリバウンドを取る。普段は何もしない。",
        "screen": _common_universe() + cond,
        "pick": f"条件に合う銘柄のうち RSI が低い順に最大 {MAX_POSITIONS} 銘柄まで。",
        "entry": entry,
        "exit": [
            "終値が 5 日移動平均を上回ったら、翌営業日の寄付に成行で売る。",
            f"買ってから {p['max_hold']} 営業日目の引けまでに上の条件にならなければ、引け成行で売る（時間切れ）。",
            "損切りの逆指値は置かない（急落局面は値が飛びやすく、逆指値は底で刈られやすい）。その代わり 1 銘柄の金額を守る。",
        ],
    }


FAMILIES: dict[str, Family] = {
    fam.key: fam
    for fam in [
        Family(
            key="dip_swing", name="押し目買い", style="スイング", side=1,
            grid={
                "trend": [0, 200],
                "mom": [0.0, 0.2],
                "rsi_n": [2, 3],
                "rsi_th": [5, 10, 20],
                "limit_pct": [0.0, 0.015, 0.03],
                "max_hold": [2, 5, 10],
                "stop_pct": [None, 0.07],
                "market": ["none", "up200"],
            },
            build=_build_dip_swing, describe=_describe_dip_swing,
            idea="短期の投げ売りは行き過ぎやすく、数日で平均に戻る傾向がある（短期リバーサル）。",
        ),
        Family(
            key="panic_dip", name="全体急落時のリバウンド狙い", style="スイング", side=1,
            grid={
                "index_rsi": [5, 10, 20],
                "rsi_th": [5, 10, 20],
                "trend": [0, 200],
                "limit_pct": [0.0, 0.015, 0.03],
                "max_hold": [2, 5, 10],
            },
            build=_build_panic_dip, describe=_describe_panic_dip,
            idea="相場全体の急落時は買い手が消えて値段が行き過ぎる。そこで買う人は、数日の戻りという形で報酬を受け取りやすい。",
        ),
        Family(
            key="dip_day", name="急落拾いデイトレ", style="デイトレ", side=1,
            grid={
                "trend": [0, 200],
                "rsi_n": [2, 3],
                "rsi_th": [10, 20, 35, 100],
                "limit_pct": [0.02, 0.03, 0.05, 0.07],
                "market": ["none", "up200"],
            },
            build=_build_dip_day, describe=_describe_dip_day,
            idea="場中のパニック的な下げは、当日中に一部戻すことが多い。",
        ),
        Family(
            key="rip_short", name="急騰の逆張り空売り", style="スイング", side=-1,
            grid={
                "rsi_th": [90, 95],
                "surge": [0.0, 0.10, 0.20],
                "trend": [0, 1],
                "limit_pct": [0.0, 0.02, 0.04],
                "max_hold": [2, 5],
                "stop_pct": [0.07, 0.12],
                "market": ["none", "down200"],
            },
            build=_build_rip_short, describe=_describe_rip_short,
            idea="材料や仕手化で短期に買われすぎた銘柄は、数日のうちに反落しやすい。",
        ),
        Family(
            key="breakout_long", name="高値ブレイク順張り", style="スイング", side=1,
            grid={
                "lookback": [20, 60, 250],
                "vol_ratio": [1.0, 2.0, 3.0],
                "exit_sma": [5, 10, 20],
                "max_hold": [5, 10, 20],
                "stop_pct": [None, 0.07],
                "market": ["none", "up200"],
            },
            build=_build_breakout_long, describe=_describe_breakout_long,
            idea="出来高を伴う新高値は、新しい買い手の参入を示し、しばらく上昇が続きやすい（モメンタム）。",
        ),
        Family(
            key="news_long", name="好材料の急伸に順張り", style="スイング", side=1,
            grid={
                "jump": [0.05, 0.08, 0.12],
                "vol_ratio": [2.0, 4.0],
                "trend": [0, 200],
                "max_hold": [3, 10, 20],
                "stop_pct": [None, 0.07],
                "market": ["none", "up200"],
            },
            build=_build_news_long, describe=_describe_news_long,
            idea="決算などの好材料は 1 日では織り込まれず、その後もしばらく同じ方向に動きやすい（決算後ドリフト）。",
        ),
        Family(
            key="breakdown_short", name="安値ブレイク順張り空売り", style="スイング", side=-1,
            grid={
                "lookback": [20, 60],
                "vol_ratio": [1.0, 2.0],
                "exit_sma": [5, 10],
                "max_hold": [3, 5, 10],
                "stop_pct": [0.07],
                "market": ["none", "down200"],
            },
            build=_build_breakdown_short, describe=_describe_breakdown_short,
            idea="下落相場で出来高を伴って安値を割った銘柄は、投げ売りが続きやすい。",
        ),
    ]
}


def build_setup(family_key: str, params: Params, features: Features, shortable: np.ndarray | None,
                rule_id: str | None = None) -> Setup:
    fam = FAMILIES[family_key]
    s = fam.build(features, params, shortable)
    s.rule_id = rule_id or family_key
    return s
