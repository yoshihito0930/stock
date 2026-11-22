"""設定ファイル (config.yaml) とリポジトリ内パスの読み込み。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"          # 価格データ（git 管理外）
UNIVERSE_CSV = DATA_DIR / "universe.csv"  # 銘柄マスタ（git 管理）
RULES_DIR = ROOT / "rules"
ACTIVE_RULES = RULES_DIR / "active.json"
RESEARCH_JSON = RULES_DIR / "research.json"
HISTORY_JSON = RULES_DIR / "history.json"
CHANGELOG = RULES_DIR / "CHANGELOG.md"
TRADES_DIR = ROOT / "trades"
INBOX_DIR = TRADES_DIR / "inbox"
LEDGER_CSV = TRADES_DIR / "ledger.csv"
MANUAL_CSV = TRADES_DIR / "manual.csv"
ROUNDTRIPS_CSV = TRADES_DIR / "roundtrips.csv"
TRADE_ANALYSIS = TRADES_DIR / "analysis.json"
TRADE_REPORT_MD = TRADES_DIR / "REPORT.md"
SLIPPAGE_OVERRIDE = TRADES_DIR / "measured_costs.json"
DOCS_DIR = ROOT / "docs"
RULES_MD = ROOT / "RULES.md"
SIGNALS_MD = ROOT / "SIGNALS.md"
SIGNALS_JSON = RULES_DIR / "signals.json"


@dataclass(frozen=True)
class Costs:
    commission_bps: float
    slippage_bps: float
    long_interest_annual: float
    short_fee_annual: float
    short_extra_annual: float

    @property
    def side_cost(self) -> float:
        """片道コスト（比率）。"""
        return (self.commission_bps + self.slippage_bps) / 10000.0

    @property
    def long_daily(self) -> float:
        return self.long_interest_annual / 365.0

    @property
    def short_daily(self) -> float:
        return (self.short_fee_annual + self.short_extra_annual) / 365.0


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    @property
    def capital(self) -> float:
        return float(self.raw["account"]["capital"])

    @property
    def monthly_target(self) -> float:
        return float(self.raw["account"]["monthly_target"])

    @property
    def max_leverage(self) -> float:
        return float(self.raw["account"]["max_leverage"])

    @property
    def lot_size(self) -> int:
        return int(self.raw["account"]["lot_size"])

    @property
    def costs(self) -> Costs:
        c = dict(self.raw["costs"])
        # 実トレードから測ったスリッページが想定より大きければ、そちらを使う
        if SLIPPAGE_OVERRIDE.exists():
            try:
                measured = json.loads(SLIPPAGE_OVERRIDE.read_text(encoding="utf-8"))
                c["slippage_bps"] = max(float(c["slippage_bps"]), float(measured.get("slippage_bps", 0.0)))
            except (ValueError, OSError):
                pass
        return Costs(**{k: float(v) for k, v in c.items()})

    @property
    def universe(self) -> dict[str, Any]:
        return self.raw["universe"]

    @property
    def data(self) -> dict[str, Any]:
        return self.raw["data"]

    @property
    def validation(self) -> dict[str, Any]:
        return self.raw["validation"]

    @property
    def portfolio(self) -> dict[str, Any]:
        return self.raw["portfolio"]

    @property
    def site(self) -> dict[str, Any]:
        return self.raw.get("site", {})


def load_config(path: Path | None = None) -> Config:
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
