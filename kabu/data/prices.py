"""株価データの取得とキャッシュ。

Yahoo Finance のチャート API から日足を取得する。
- 価格は「分割・配当調整済み」に直して保存する（買いは配当込み、空売りは配当落ち分を支払う形になる）。
- 発注株数の計算には当時の実際の株価が要るので、未調整終値 (raw_close) も保存する。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from ..config import CACHE_DIR, Config

_HOSTS = ("query2.finance.yahoo.com", "query1.finance.yahoo.com")
_UA = {"User-Agent": "Mozilla/5.0 (compatible; kabu-research/0.1)"}
FIELDS = ("open", "high", "low", "close", "volume", "raw_close")
PANEL_PATH = CACHE_DIR / "prices.parquet"


def to_symbol(code: str) -> str:
    return code if ("." in code or code.startswith("^")) else f"{code}.T"


def _fetch_json(symbol: str, start: str, session: requests.Session, retries: int = 4) -> dict | None:
    p1 = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    p2 = int(time.time()) + 86400
    last_err: Exception | None = None
    for attempt in range(retries):
        host = _HOSTS[attempt % len(_HOSTS)]
        url = f"https://{host}/v8/finance/chart/{symbol}"
        try:
            r = session.get(
                url,
                params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"},
                headers=_UA,
                timeout=30,
            )
            if r.status_code == 404:
                return None
            if r.status_code == 200:
                res = r.json().get("chart", {}).get("result")
                return res[0] if res else None
            last_err = RuntimeError(f"HTTP {r.status_code}")
        except (requests.RequestException, ValueError) as e:  # ネットワーク/JSON エラーは再試行
            last_err = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{symbol}: 取得失敗 ({last_err})")


MARKET_CLOSE_JST = (15, 45)  # 大引け 15:30 の後、日足が確定したとみなす時刻


def drop_unfinished_bar(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """場中に取得すると当日の足は途中経過になる。確定前の足でシグナルを出さないよう捨てる。"""
    if df.empty:
        return df
    now = now or pd.Timestamp.now(tz="Asia/Tokyo").tz_localize(None)
    if df.index[-1] == now.normalize() and (now.hour, now.minute) < MARKET_CLOSE_JST:
        return df.iloc[:-1]
    return df


def parse_chart(result: dict, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """チャート API の 1 銘柄分の結果を日足 DataFrame にする。"""
    ts = result.get("timestamp")
    if not ts:
        return pd.DataFrame(columns=list(FIELDS))
    q = result["indicators"]["quote"][0]
    adj = result["indicators"].get("adjclose", [{}])[0].get("adjclose")
    offset = result.get("meta", {}).get("gmtoffset", 32400)
    idx = pd.to_datetime(np.asarray(ts, dtype="int64") + offset, unit="s").normalize()
    df = pd.DataFrame(
        {
            "open": q.get("open"),
            "high": q.get("high"),
            "low": q.get("low"),
            "close": q.get("close"),
            "volume": q.get("volume"),
            "adjclose": adj if adj is not None else q.get("close"),
        },
        index=idx,
        dtype="float64",
    )
    df = df[~df.index.duplicated(keep="last")]
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = drop_unfinished_bar(df, now)
    df = df[(df["close"] > 0) & (df["open"] > 0)]
    if df.empty:
        return pd.DataFrame(columns=list(FIELDS))

    # 未調整終値: API の close は分割調整済みなので、その後に起きた分割の比率を掛け戻す
    raw_factor = pd.Series(1.0, index=df.index)
    splits = (result.get("events") or {}).get("splits") or {}
    for ev in splits.values():
        try:
            num, den = float(ev["numerator"]), float(ev["denominator"])
            d = pd.to_datetime(int(ev["date"]) + offset, unit="s").normalize()
        except (KeyError, TypeError, ValueError):
            continue
        if num > 0 and den > 0:
            raw_factor[df.index < d] *= num / den
    raw_close = df["close"] * raw_factor

    # 配当も含めた調整係数を OHLC に掛ける。
    # 提供元の調整後終値が壊れている銘柄がある（負の値・不連続な係数）。その場合は分割調整のみにする。
    ratio = (df["adjclose"] / df["close"]).replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(1.0)
    step = ratio / ratio.shift(1)
    ratio_ok = (
        (ratio.min() > 0.2)
        and (ratio.max() < 1.001)
        and (step.dropna().between(0.999, 1.15).all())  # 係数は時間とともに 1 に近づくだけのはず
    )
    if not ratio_ok:
        ratio = pd.Series(1.0, index=df.index)
    out = pd.DataFrame(
        {
            "open": df["open"] * ratio,
            "high": df["high"] * ratio,
            "low": df["low"] * ratio,
            "close": df["close"] * ratio,
            "volume": df["volume"].fillna(0.0) / ratio,  # 売買代金 (= 価格×出来高) が保存されるように
            "raw_close": raw_close,
        }
    )
    # 高値・安値の整合性（データ不備対策）
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    return drop_broken_history(out)


# 値幅制限（最大でも前日比 +-30〜40% 程度）では起こり得ない動き = データ不良とみなす
_MAX_DAILY_MOVE = 0.5
_MAX_DAY_RANGE = 2.0
_WARMUP_AFTER_BAD = 260  # 不良データの影響が指標（200日線など）から抜けるまでの日数


def drop_broken_history(df: pd.DataFrame) -> pd.DataFrame:
    """あり得ない値動きを含む期間を、その影響が消えるまで丸ごと捨てる。

    未調整の分割・誤った値・再上場前の別会社のデータなどが混じると、
    バックテストに架空の大勝ち/大負けが入り込む。疑わしい足が見つかったら
    「最後の不良足 + 260 営業日」より前の履歴を使わない。
    """
    if df.empty:
        return df
    ret = df["close"] / df["close"].shift(1) - 1.0
    bad = (ret.abs() > _MAX_DAILY_MOVE) | (df["high"] / df["low"] > _MAX_DAY_RANGE) | (df["close"] <= 0)
    if not bad.any():
        return df
    last_bad = int(np.flatnonzero(bad.to_numpy())[-1])
    return df.iloc[last_bad + _WARMUP_AFTER_BAD:]


def fetch_one(code: str, start: str, session: requests.Session | None = None) -> pd.DataFrame:
    s = session or requests.Session()
    res = _fetch_json(to_symbol(code), start, s)
    if res is None:
        return pd.DataFrame(columns=list(FIELDS))
    return parse_chart(res)


def download_all(codes: list[str], start: str, workers: int = 6, progress: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """全銘柄を並列取得して縦持ち DataFrame (date, code, ...) にまとめる。"""
    session = requests.Session()
    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    done = 0

    def job(code: str) -> tuple[str, pd.DataFrame | None]:
        try:
            return code, fetch_one(code, start, session)
        except RuntimeError:
            return code, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(job, c) for c in codes]
        for fut in as_completed(futs):
            code, df = fut.result()
            done += 1
            if df is None or df.empty:
                failed.append(code)
            else:
                df = df.copy()
                df["code"] = code
                df.index.name = "date"
                frames.append(df.reset_index())
            if progress and done % 100 == 0:
                print(f"  {done}/{len(codes)} 銘柄 取得済み（失敗 {len(failed)}）", flush=True)
    if not frames:
        raise RuntimeError("株価を 1 銘柄も取得できませんでした。")
    return pd.concat(frames, ignore_index=True), failed


def update_prices(cfg: Config, codes: list[str], force: bool = False) -> pd.DataFrame:
    """株価キャッシュを更新する。調整係数は過去に遡って変わるため、毎回全期間を取り直す。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index_code = cfg.universe["index_symbol"]
    if PANEL_PATH.exists() and not force:
        age_h = (time.time() - PANEL_PATH.stat().st_mtime) / 3600
        if age_h < 6:
            print(f"株価キャッシュは {age_h:.1f} 時間前に更新済みのため再取得しません（--force で強制）。")
            return pd.read_parquet(PANEL_PATH)
    all_codes = sorted(set(codes) | {index_code})
    print(f"{len(all_codes)} 銘柄の日足を取得します（{cfg.data['start']} 〜）...")
    long_df, failed = download_all(all_codes, cfg.data["start"])
    if failed:
        print(f"取得できなかった銘柄: {len(failed)} 件 {failed[:15]}{' ...' if len(failed) > 15 else ''}")
    if len(failed) > 0.2 * len(all_codes):
        raise RuntimeError("取得失敗が 20% を超えました。データ提供元の障害の可能性があるため中止します。")
    long_df.to_parquet(PANEL_PATH, index=False)
    return long_df


@dataclass
class Panel:
    """日付 × 銘柄 の横持ちデータ一式。"""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    raw_close: pd.DataFrame
    index_close: pd.Series  # 相場環境判定用の指数

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def codes(self) -> list[str]:
        return list(self.close.columns)

    def slice(self, start: str | None = None, end: str | None = None) -> "Panel":
        sl = slice(start, end)
        return Panel(
            self.open.loc[sl], self.high.loc[sl], self.low.loc[sl], self.close.loc[sl],
            self.volume.loc[sl], self.raw_close.loc[sl], self.index_close.loc[sl],
        )


def build_panel(long_df: pd.DataFrame, index_code: str) -> Panel:
    long_df = long_df.copy()
    long_df["date"] = pd.to_datetime(long_df["date"])
    wide = {f: long_df.pivot(index="date", columns="code", values=f).sort_index() for f in FIELDS}
    if index_code not in wide["close"].columns:
        raise RuntimeError(f"指数 {index_code} のデータがありません。")
    stocks = [c for c in wide["close"].columns if c != index_code]
    # 営業日 = 半数以上の銘柄に値が付いた日（指数データ側の欠損日に左右されないようにする）
    counts = wide["close"][stocks].notna().sum(axis=1)
    days = counts.index[counts >= 0.5 * counts.max()]
    index_close = wide["close"][index_code].reindex(days).ffill()
    w = {f: wide[f].reindex(index=days, columns=stocks) for f in FIELDS}
    return Panel(w["open"], w["high"], w["low"], w["close"], w["volume"].fillna(0.0), w["raw_close"], index_close)


def load_panel(cfg: Config) -> Panel:
    if not PANEL_PATH.exists():
        raise FileNotFoundError("株価キャッシュがありません。先に `python -m kabu update-data` を実行してください。")
    return build_panel(pd.read_parquet(PANEL_PATH), cfg.universe["index_symbol"])
