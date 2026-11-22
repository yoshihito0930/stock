"""銘柄マスタ: JPX が公表する上場銘柄一覧から売買対象の母集団を作る。"""
from __future__ import annotations

import io

import pandas as pd
import requests

from ..config import UNIVERSE_CSV, Config

JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xlsx"
)
_UA = {"User-Agent": "Mozilla/5.0 (compatible; kabu-research/0.1)"}


def fetch_universe(cfg: Config) -> pd.DataFrame:
    """JPX の一覧をダウンロードし、設定の規模区分に該当する内国株だけ残す。"""
    resp = requests.get(JPX_LIST_URL, headers=_UA, timeout=60)
    resp.raise_for_status()
    raw = pd.read_excel(io.BytesIO(resp.content), dtype=str)
    raw = raw.rename(
        columns={
            "コード": "code",
            "銘柄名": "name",
            "市場・商品区分": "market",
            "33業種区分": "sector",
            "規模区分": "segment",
        }
    )
    keep = raw["segment"].isin(cfg.universe["segments"]) & raw["market"].str.contains("内国株式", na=False)
    df = raw.loc[keep, ["code", "name", "market", "sector", "segment"]].copy()
    df["code"] = df["code"].str.strip()
    df["name"] = df["name"].str.replace("　", " ").str.strip()
    df["shortable"] = df["segment"].isin(cfg.universe["short_segments"])
    return df.sort_values("code").reset_index(drop=True)


def update_universe(cfg: Config) -> pd.DataFrame:
    df = fetch_universe(cfg)
    if len(df) < 100:
        raise RuntimeError(f"銘柄一覧が異常に少ない ({len(df)} 件)。JPX のファイル形式が変わった可能性があります。")
    UNIVERSE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(UNIVERSE_CSV, index=False, encoding="utf-8")
    return df


def load_universe(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    if refresh or not UNIVERSE_CSV.exists():
        return update_universe(cfg)
    df = pd.read_csv(UNIVERSE_CSV, dtype={"code": str})
    df["shortable"] = df["shortable"].astype(bool)
    return df
