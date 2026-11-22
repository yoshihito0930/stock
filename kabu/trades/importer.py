"""実トレード履歴の取り込み。

対応形式:
  1. 楽天証券の「取引履歴（国内株式）」CSV（Shift-JIS）。現物・信用新規・信用返済を読む。
  2. 汎用 CSV（手入力用）: date,code,side,qty,price[,fee,name,note]
       side = buy(買い) / sell(売り) / short(空売り) / cover(買い戻し)

取り込んだ約定は trades/ledger.csv に正規化して蓄積する。同じ CSV を何度入れても重複しない。
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from ..config import INBOX_DIR, LEDGER_CSV, MANUAL_CSV

LEDGER_COLUMNS = ["date", "code", "name", "action", "direction", "account", "qty", "price", "fee", "seq", "source"]
_KEY = ["date", "code", "action", "direction", "qty", "price"]


def _norm(s: str) -> str:
    """列名の表記ゆれを吸収する（全角/半角、単位だけの括弧、空白を除去）。

    「金利（支払）」「金利（受取）」のように意味のある括弧は残す。
    """
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[\[\(〔［（]\s*(円|株|税抜)\s*[\]\)〕］）]", "", s)
    return re.sub(r"\s+", "", s)


def _num(x) -> float:
    if x is None:
        return 0.0
    s = unicodedata.normalize("NFKC", str(x)).replace(",", "").replace("円", "").strip()
    if s in ("", "-", "--", "nan", "None"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _read_csv(path: Path) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, keep_default_na=False)
        except (UnicodeDecodeError, pd.errors.ParserError) as e:
            last = e
    raise ValueError(f"{path.name}: CSV として読めませんでした ({last})")


def _find(cols: dict[str, str], *keywords: str, exclude: tuple[str, ...] = ()) -> str | None:
    for norm, orig in cols.items():
        if all(k in norm for k in keywords) and not any(x in norm for x in exclude):
            return orig
    return None


def parse_rakuten(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    cols = {_norm(c): c for c in df.columns}
    c_date = _find(cols, "約定日", exclude=("建",))
    c_code = _find(cols, "銘柄コード")
    c_name = _find(cols, "銘柄名")
    c_kind = _find(cols, "取引区分")
    c_side = _find(cols, "売買区分")
    c_qty = _find(cols, "数量")
    c_price = _find(cols, "単価", exclude=("建",))
    missing = [n for n, c in (("約定日", c_date), ("銘柄コード", c_code), ("取引区分", c_kind),
                              ("売買区分", c_side), ("数量", c_qty), ("単価", c_price)) if c is None]
    if missing:
        raise ValueError(f"{source}: 楽天証券の取引履歴に必要な列がありません: {missing}")
    # 支払いコストの列（受取は差し引く）
    pay = [c for n, c in cols.items()
           if any(k in n for k in ("手数料", "税金等", "諸費用", "金利", "逆日歩", "貸株料", "事務管理費", "名義書換料"))
           and "受取" not in n]
    recv = [c for n, c in cols.items() if "受取" in n and any(k in n for k in ("金利", "逆日歩"))]

    rows, warnings, skipped = [], [], {}
    for _, r in df.iterrows():
        kind, side = _norm(r[c_kind]), _norm(r[c_side])
        if "新規" in kind:
            action, direction, account = "open", ("long" if "買" in side else "short"), "信用"
        elif "返済" in kind:
            # 売埋 = 買建の返済（買いポジションの決済）、買埋 = 売建の返済
            action, direction, account = "close", ("long" if "売" in side else "short"), "信用"
        elif "現物" in kind:
            action, direction, account = ("open" if "買" in side else "close"), "long", "現物"
        else:
            skipped[kind] = skipped.get(kind, 0) + 1
            continue
        qty, price = _num(r[c_qty]), _num(r[c_price])
        if qty <= 0 or price <= 0:
            continue
        fee = sum(_num(r[c]) for c in pay) - sum(_num(r[c]) for c in recv)
        rows.append({
            "date": pd.to_datetime(unicodedata.normalize("NFKC", r[c_date])).strftime("%Y-%m-%d"),
            "code": _norm(r[c_code]).upper(), "name": str(r[c_name]).strip() if c_name else "",
            "action": action, "direction": direction, "account": account,
            "qty": qty, "price": price, "fee": fee, "source": source,
        })
    for k, n in skipped.items():
        warnings.append(f"{source}: 取引区分「{k}」の {n} 行は対象外として読み飛ばしました。")
    return pd.DataFrame(rows), warnings


_SIDE_MAP = {
    "buy": ("open", "long"), "買": ("open", "long"), "買い": ("open", "long"), "買付": ("open", "long"),
    "買建": ("open", "long"),
    "sell": ("close", "long"), "売": ("close", "long"), "売り": ("close", "long"), "売付": ("close", "long"),
    "売埋": ("close", "long"),
    "short": ("open", "short"), "空売り": ("open", "short"), "売建": ("open", "short"),
    "cover": ("close", "short"), "買戻し": ("close", "short"), "買い戻し": ("close", "short"),
    "買埋": ("close", "short"),
}


def parse_generic(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    cols = {_norm(c).lower(): c for c in df.columns}
    need = ["date", "code", "side", "qty", "price"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"{source}: 汎用形式に必要な列がありません: {missing}（必要: {need}）")
    rows, warnings = [], []
    for n, r in df.iterrows():
        side = _norm(r[cols["side"]]).lower()
        if not side and not str(r[cols["date"]]).strip():
            continue
        if side not in _SIDE_MAP:
            warnings.append(f"{source} {n + 2} 行目: side「{r[cols['side']]}」が解釈できません（buy/sell/short/cover）。")
            continue
        action, direction = _SIDE_MAP[side]
        qty, price = _num(r[cols["qty"]]), _num(r[cols["price"]])
        if qty <= 0 or price <= 0:
            continue
        rows.append({
            "date": pd.to_datetime(str(r[cols["date"]]).strip()).strftime("%Y-%m-%d"),
            "code": _norm(r[cols["code"]]).upper(), "name": str(r[cols["name"]]).strip() if "name" in cols else "",
            "action": action, "direction": direction, "account": "手入力",
            "qty": qty, "price": price, "fee": _num(r[cols["fee"]]) if "fee" in cols else 0.0, "source": source,
        })
    return pd.DataFrame(rows), warnings


def parse_file(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = _read_csv(path)
    normed = {_norm(c) for c in df.columns}
    if any("約定日" in c for c in normed) and any("売買区分" in c for c in normed):
        out, warn = parse_rakuten(df, path.name)
    else:
        out, warn = parse_generic(df, path.name)
    if out.empty:
        return pd.DataFrame(columns=LEDGER_COLUMNS), warn
    out["seq"] = out.groupby(_KEY).cumcount()
    return out[LEDGER_COLUMNS], warn


def load_ledger() -> pd.DataFrame:
    if not LEDGER_CSV.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    df = pd.read_csv(LEDGER_CSV, dtype={"code": str, "name": str, "source": str}, keep_default_na=False)
    for c in ("qty", "price", "fee"):
        df[c] = df[c].astype(float)
    df["seq"] = df["seq"].astype(int)
    return df


def merge_ledger(ledger: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """同じ約定（日付・銘柄・売買・数量・単価・同一内容の何件目か）は 1 回だけ残す。"""
    if new.empty:
        return ledger
    both = new if ledger.empty else pd.concat([ledger, new], ignore_index=True)
    both = both.drop_duplicates(subset=_KEY + ["seq"], keep="first")
    return both.sort_values(["date", "code", "action"], kind="stable",
                            key=lambda s: s.map({"open": 0, "close": 1}) if s.name == "action" else s
                            ).reset_index(drop=True)


def import_files(files: list[str] | None = None, log=print) -> tuple[int, int]:
    """CSV を台帳に取り込む。files 省略時は trades/inbox/*.csv と trades/manual.csv。"""
    if files:
        paths = [Path(f) for f in files]
    else:
        paths = sorted(INBOX_DIR.glob("*.csv")) if INBOX_DIR.exists() else []
        if MANUAL_CSV.exists():
            paths.append(MANUAL_CSV)
    ledger = load_ledger()
    before = len(ledger)
    for p in paths:
        try:
            new, warn = parse_file(p)
        except ValueError as e:
            log(f"読み込み失敗: {e}")
            continue
        for w in warn:
            log("注意: " + w)
        ledger = merge_ledger(ledger, new)
        log(f"{p.name}: {len(new)} 件の約定を読みました。")
    if len(ledger):
        LEDGER_CSV.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(LEDGER_CSV, index=False, encoding="utf-8")
    return len(ledger) - before, len(ledger)
