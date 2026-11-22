"""コマンドライン。`python -m kabu <command>` で実行する。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

from .config import SIGNALS_JSON, load_config
from .util import write_json


def _context(cfg, refresh_universe: bool = False):
    from .data.prices import load_panel
    from .research import Context

    return Context.create(cfg, load_panel(cfg))


def cmd_update_data(args) -> None:
    from .data.prices import update_prices
    from .data.universe import load_universe

    cfg = load_config()
    uni = load_universe(cfg, refresh=args.refresh_universe)
    df = update_prices(cfg, uni["code"].tolist(), force=args.force)
    print(f"株価データ: {df['code'].nunique()} 銘柄、最終日 {str(df['date'].max())[:10]}")


def _finish(ctx, state, events, today: str) -> None:
    from .report import build_all
    from .rulebook import finalize

    state, signals = finalize(ctx, state, events, today)
    write_json(SIGNALS_JSON, signals)
    build_all()
    for e in events:
        print(f"  [{e['type']}] {e['title']}")
    n_act = sum(1 for r in state["rules"] if r["status"] == "active")
    n_watch = sum(1 for r in state["rules"] if r["status"] == "watch")
    print(f"台帳を更新しました: 採用 {n_act} / 観察中 {n_watch} / 不採用 {len(state['rejected'])}"
          f"（データ基準日 {state['data_through']}）")


def cmd_research(args) -> None:
    """全ルールを探索・検証し直して台帳を更新する（週次）。"""
    from .rulebook import load_state, update_rules

    cfg = load_config()
    if not args.no_update:
        cmd_update_data(argparse.Namespace(force=args.force, refresh_universe=True))
    t0 = time.time()
    ctx = _context(cfg)
    today = args.today or date.today().isoformat()
    state, events = update_rules(ctx, load_state(), families=args.families or None, today=today,
                                 workers=args.workers)
    _finish(ctx, state, events, today)
    print(f"所要時間 {time.time() - t0:.0f} 秒")


def cmd_daily(args) -> None:
    """株価を更新し、台帳のルールを採点し直して、次の営業日のシグナルを出す（日次）。"""
    from .rulebook import load_state, refresh_rules

    cfg = load_config()
    if not args.no_update:
        cmd_update_data(argparse.Namespace(force=args.force, refresh_universe=False))
    ctx = _context(cfg)
    today = args.today or date.today().isoformat()
    state = load_state()
    if not state.get("rules") and not state.get("rejected"):
        print("台帳が空です。先に `python -m kabu research` を実行してください。")
        sys.exit(1)
    state, events = refresh_rules(ctx, state, today)
    # 実トレードが取り込まれていれば、最新のルール・株価で分析もやり直す
    from .trades.analysis import analyze_if_any

    analyze_if_any(ctx)
    _finish(ctx, state, events, today)


def cmd_import_trades(args) -> None:
    """実トレードの履歴 CSV を取り込み、分析する。"""
    from .report import build_all
    from .trades.analysis import analyze_if_any
    from .trades.importer import import_files

    cfg = load_config()
    n_new, n_total = import_files(args.files or None)
    print(f"取り込み: 新規 {n_new} 件（台帳合計 {n_total} 件）")
    ctx = _context(cfg)
    if analyze_if_any(ctx):
        print("実トレード分析を更新しました: trades/REPORT.md")
    build_all()


def cmd_report(args) -> None:
    from .report import build_all

    build_all()
    print("RULES.md / SIGNALS.md / docs/index.html を再生成しました。")


def cmd_backtest(args) -> None:
    """自分のルール案（ファミリー + パラメータ）を過去データで試す。"""
    from .research import check_gates, evaluate_rule
    from .strategies import FAMILIES

    cfg = load_config()
    if args.family not in FAMILIES:
        print("ファミリー名: " + ", ".join(FAMILIES))
        sys.exit(1)
    fam = FAMILIES[args.family]
    params = {k: v[0] for k, v in fam.grid.items()}
    if args.params:
        params.update(json.loads(args.params))
    ctx = _context(cfg)
    ev = evaluate_rule(ctx, args.family, params)
    print(f"{fam.name}  {params}")
    for label, key in (("学習期間", "is"), ("未使用データ", "oos"), ("全期間", "full")):
        m = ev[key]
        print(f"  {label:6s} 取引 {m['trades']:5d} 回  勝率 {m['win_rate']:.0%}  PF {m['profit_factor']:.2f}  "
              f"月平均 {m['monthly_mean'] / 1e4:+.1f}万円  最大DD {m['max_drawdown'] / 1e4:.0f}万円  t値 {m['tstat']:.1f}")
    for g in check_gates(ev, None, None, cfg):
        print(f"  {'○' if g['ok'] else '×'} {g['name']}: {g['value']:.2f}（基準 {g['threshold']}）")
    print("  ※ ウォークフォワードと近傍パラメータの検査は `research` でのみ行います。")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="kabu", description="過去データで検証した売買ルールを作り、育てる。")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("update-data", help="株価データを更新する")
    p.add_argument("--force", action="store_true")
    p.add_argument("--refresh-universe", action="store_true", help="銘柄一覧も取り直す")
    p.set_defaults(fn=cmd_update_data)

    p = sub.add_parser("research", help="[週次] ルールを探索・検証し直して台帳を更新する")
    p.add_argument("--families", nargs="*", help="対象ファミリー（省略時は全部）")
    p.add_argument("--no-update", action="store_true", help="株価を取り直さない")
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int)
    p.add_argument("--today", help="記録に使う日付（テスト用）")
    p.set_defaults(fn=cmd_research)

    p = sub.add_parser("daily", help="[日次] 株価更新 → ルールの健康診断 → 次の営業日のシグナル")
    p.add_argument("--no-update", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--today")
    p.set_defaults(fn=cmd_daily)

    p = sub.add_parser("import-trades", help="実トレードの CSV を取り込んで分析する")
    p.add_argument("files", nargs="*", help="CSV ファイル（省略時は trades/inbox/ の中身）")
    p.set_defaults(fn=cmd_import_trades)

    p = sub.add_parser("report", help="RULES.md などを再生成する")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("backtest", help="ルール案を過去データで試す")
    p.add_argument("family")
    p.add_argument("--params", help='JSON 例: \'{"rsi_th": 10, "max_hold": 5}\'')
    p.set_defaults(fn=cmd_backtest)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
