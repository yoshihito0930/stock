import pandas as pd
import pytest

from kabu.trades.importer import merge_ledger, parse_file
from kabu.trades.roundtrip import build_roundtrips

RAKUTEN_HEADER = ('"約定日","受渡日","銘柄コード","銘柄名","市場名称","口座区分","取引区分","売買区分","信用区分","弁済期限",'
                  '"数量［株］","単価［円］","手数料［円］","税金等［円］","諸費用［円］","税区分","受渡金額［円］","建約定日","建単価［円］",'
                  '"建手数料［円］","建手数料消費税［円］","金利（支払）〔円〕","金利（受取）〔円〕","逆日歩／特別空売り料（支払）〔円〕",'
                  '"逆日歩（受取）〔円〕","貸株料","事務管理費〔円〕（税抜）","名義書換料〔円〕（税抜）"')
RAKUTEN_ROWS = [
    '"2026/10/1","2026/10/3","7203","トヨタ自動車","東証","特定","現物","買付","-","-","100","3,000.0","0","0","0","源泉徴収あり","300,000","-","-","-","-","-","-","-","-","-","-","-"',
    '"2026/10/5","2026/10/7","7203","トヨタ自動車","東証","特定","現物","売付","-","-","100","3,100.0","0","0","0","源泉徴収あり","310,000","-","-","-","-","-","-","-","-","-","-","-"',
    '"2026/10/6","2026/10/8","9984","ソフトバンクグループ","東証","特定","信用新規","売建","制度","6ヶ月","200","10,000.0","0","0","0","-","-","-","-","-","-","-","-","-","-","-","-","-"',
    '"2026/10/8","2026/10/10","9984","ソフトバンクグループ","東証","特定","信用返済","買埋","制度","6ヶ月","200","9,500.0","0","0","0","源泉徴収あり","99,000","2026/10/6","10,000.0","0","0","0","0","300","0","700","0","0"',
    '"2026/10/9","2026/10/13","6758","ソニーグループ","東証","特定","信用新規","買建","制度","6ヶ月","100","4,000.0","0","0","0","-","-","-","-","-","-","-","-","-","-","-","-","-"',
    '"2026/10/9","2026/10/13","6758","ソニーグループ","東証","特定","信用返済","売埋","制度","6ヶ月","100","3,900.0","0","0","0","源泉徴収あり","-10,100","2026/10/9","4,000.0","0","0","100","0","0","0","0","0","0"',
    '"2026/10/12","2026/10/14","8306","三菱ＵＦＪ","東証","特定","現引","-","制度","6ヶ月","100","2,000.0","0","0","0","-","-","-","-","-","-","-","-","-","-","-","-","-"',
]


@pytest.fixture
def rakuten_csv(tmp_path):
    p = tmp_path / "tradehistory(JP)_20261015.csv"
    p.write_bytes(("\n".join([RAKUTEN_HEADER] + RAKUTEN_ROWS) + "\n").encode("cp932"))
    return p


def test_rakuten_csv_is_parsed(rakuten_csv):
    df, warn = parse_file(rakuten_csv)
    assert len(df) == 6
    assert any("現引" in w for w in warn)           # 対象外の行は黙って捨てず、警告を出す
    first = df.iloc[0]
    assert (first["date"], first["code"], first["action"], first["direction"], first["price"]) == \
        ("2026-10-01", "7203", "open", "long", 3000.0)
    short_open = df[(df["code"] == "9984") & (df["action"] == "open")].iloc[0]
    assert short_open["direction"] == "short"
    short_close = df[(df["code"] == "9984") & (df["action"] == "close")].iloc[0]
    assert short_close["direction"] == "short" and short_close["fee"] == 1000.0   # 逆日歩 300 + 貸株料 700


def test_roundtrips_and_pnl(rakuten_csv):
    df, _ = parse_file(rakuten_csv)
    rt, open_lots, unmatched = build_roundtrips(df)
    assert len(rt) == 3 and open_lots.empty and unmatched == 0
    by = rt.set_index("code")
    assert by.loc["7203", "pnl"] == pytest.approx(10_000)             # (3100-3000)×100
    assert by.loc["9984", "pnl"] == pytest.approx(100_000 - 1000)     # 空売り: (10000-9500)×200 − コスト
    assert by.loc["6758", "pnl"] == pytest.approx(-10_000 - 100)      # デイトレの損失 + 金利
    assert by.loc["6758", "days"] == 0 and by.loc["7203", "days"] == 4


def test_reimport_does_not_duplicate(rakuten_csv):
    df, _ = parse_file(rakuten_csv)
    ledger = merge_ledger(pd.DataFrame(columns=df.columns), df)
    again = merge_ledger(ledger, df)
    assert len(again) == len(ledger) == 6


def test_identical_fills_same_day_are_kept(tmp_path):
    # 同じ日に同じ値段で 2 回約定したものは、重複ではなく 2 件として残す
    p = tmp_path / "manual.csv"
    p.write_text("date,code,side,qty,price\n2026-10-01,7203,buy,100,3000\n2026-10-01,7203,buy,100,3000\n"
                 "2026-10-02,7203,sell,200,3050\n", encoding="utf-8")
    df, _ = parse_file(p)
    assert len(df) == 3
    rt, _, _ = build_roundtrips(merge_ledger(pd.DataFrame(columns=df.columns), df))
    assert len(rt) == 1 and rt.loc[0, "qty"] == 200 and rt.loc[0, "pnl"] == pytest.approx(10_000)


def test_fifo_partial_close_and_unmatched(tmp_path):
    p = tmp_path / "manual.csv"
    p.write_text("date,code,side,qty,price,fee\n"
                 "2026-10-01,7203,buy,100,3000,0\n2026-10-02,7203,buy,100,3200,0\n"
                 "2026-10-03,7203,sell,150,3300,0\n2026-10-04,6758,sell,100,4000,0\n"
                 "2026-10-05,1111,short,100,500,0\n2026-10-06,1111,cover,100,450,0\n", encoding="utf-8")
    df, _ = parse_file(p)
    rt, open_lots, unmatched = build_roundtrips(df)
    assert unmatched == 1                                   # 建てた記録のない 6758 の売り
    t = rt[rt["code"] == "7203"]
    assert t["pnl"].sum() == pytest.approx(100 * 300 + 50 * 100)   # 古い建玉から先に決済
    assert open_lots.iloc[0]["qty"] == 50 and open_lots.iloc[0]["entry_price"] == 3200
    assert rt[rt["code"] == "1111"]["pnl"].iloc[0] == pytest.approx(5000)


def test_bad_side_is_reported(tmp_path):
    p = tmp_path / "manual.csv"
    p.write_text("date,code,side,qty,price\n2026-10-01,7203,hold,100,3000\n", encoding="utf-8")
    df, warn = parse_file(p)
    assert df.empty and warn
