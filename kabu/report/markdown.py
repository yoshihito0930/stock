"""RULES.md / SIGNALS.md / trades/REPORT.md の生成（GitHub のスマホアプリでそのまま読める形）。"""
from __future__ import annotations

from datetime import date, timedelta

from ..strategies import FAMILIES, MAX_POSITIONS
from ..util import man, pct
from . import texts

STATUS_LABEL = {"active": "採用", "watch": "観察中（仮想売買）", "suspended": "一時停止"}
STATUS_EMOJI = {"active": "🟢", "watch": "🟡", "suspended": "⏸"}


def next_business_day(d: str) -> str:
    x = date.fromisoformat(d) + timedelta(days=1)
    while x.weekday() >= 5:
        x += timedelta(days=1)
    return x.isoformat()


def fmt_gate(g: dict) -> tuple[str, str]:
    f = g.get("fmt")
    def one(v):
        if f == "int":
            return f"{v:.0f}"
        if f == "pct":
            return f"{v * 100:.0f}%"
        return f"{v:.2f}"
    op = "以下" if g.get("upper") else "以上"
    return one(g["value"]), f"{one(g['threshold'])} {op}"


def position_yen(state: dict, rule: dict) -> float:
    cfg = state["config"]
    pf = state.get("portfolio") if rule["status"] == "active" else state.get("portfolio_watch")
    scale = (pf or {}).get("scale") or 0.5
    return cfg["capital"] * scale / MAX_POSITIONS


def headline(state: dict) -> list[str]:
    """いまの結論（目標に対してどこまで届いているか）を正直に書く。"""
    cfg = state["config"]
    tgt, cap = cfg["monthly_target"], cfg["capital"]
    rules = state["rules"]
    n_act = sum(1 for r in rules if r["status"] == "active")
    n_watch = sum(1 for r in rules if r["status"] == "watch")
    n_sus = sum(1 for r in rules if r["status"] == "suspended")
    out = [f"- **目標**: 月 {tgt / 1e4:.0f}万円（運用資金 {cap / 1e4:.0f}万円 → 月利 {tgt / cap:.1%}）",
           f"- **ルールの状態**: 採用 {n_act} 本 / 観察中 {n_watch} 本 / 一時停止 {n_sus} 本 / 不採用 {len(state['rejected'])} 本"]
    pf = state.get("portfolio") or {}
    pw = state.get("portfolio_watch") or {}
    if not pf.get("empty", True):
        s = pf["summary"]
        out.append(f"- **採用ルールの過去検証**: 月平均 **{man(s['monthly_mean'])}**（目標の {s['target_ratio']:.0%}）、"
                   f"プラスの月 {pct(s['positive_months'])}、最悪の月 {man(s['monthly_worst'])}、最大ドローダウン {man(s['max_drawdown'], sign=False)}")
        if pf.get("required_capital"):
            out.append(f"- 同じルール・同じリスクで月 {tgt / 1e4:.0f}万円に届くための資金の目安: 約 {pf['required_capital'] / 1e4:,.0f}万円")
    else:
        out.append("- **採用ルール（実際の資金で運用してよいルール）は、現時点でありません。** すべての採用基準を通ったルールがまだ無いためです。")
    if not pw.get("empty", True):
        s = pw["summary"]
        out.append(f"- **観察中ルールを仮に運用した場合の過去検証**: 月平均 {man(s['monthly_mean'])}（目標の {s['target_ratio']:.0%}）、"
                   f"プラスの月 {pct(s['positive_months'])}、最悪の月 {man(s['monthly_worst'])}、最大ドローダウン {man(s['max_drawdown'], sign=False)}。"
                   "統計的な優位性は確認できたが、リスクに対する見返りが基準未満のため、まだ実弾は推奨しません。")
    best = pf if not pf.get("empty", True) else pw
    if not best.get("empty", True) and best["summary"]["target_ratio"] < 1:
        out.append(f"- **正直な現状**: 検証を通過した範囲では月 {tgt / 1e4:.0f}万円には届いていません。"
                   "届いていないものを届くとは書きません。毎週の自動探索で基準を通るルールが増えれば、この数字が更新されます。")
    return out


def rule_section(state: dict, rule: dict, n: int) -> list[str]:
    fam = FAMILIES[rule["family"]]
    d = fam.describe(rule["params"])
    ev = rule["evidence"]
    st = rule["status"]
    yen_pos = position_yen(state, rule)
    L = [f"### ルール{n}: {fam.name}（{fam.style}） {STATUS_EMOJI[st]} {STATUS_LABEL[st]}",
         "",
         f"v{rule['version']}・{rule['adopted_on']} から現在の内容・ID `{rule['id']}`",
         "",
         f"**ひとことで**: {d['summary']}",
         "",
         f"**なぜ効くと考えるか**: {fam.idea}",
         ""]
    if st == "watch":
        L += ["> 🟡 **観察中**: 統計的な優位性の基準は通過しましたが、下の「採用基準チェック」に未達があります。"
              "シグナルは毎日出ますが、仮想売買（記録だけ）で様子を見る段階です。", ""]
    if st == "suspended":
        L += ["> ⏸ **一時停止中**: " + " / ".join(rule["health"]["reasons"]) + "。新規の売買はしません。", ""]
    L += ["**① 銘柄を選ぶ条件**（前日の終値で判定。毎日の該当銘柄は SIGNALS.md に自動で出ます）", ""]
    L += [f"- {x}" for x in d["screen"]]
    L += [f"- {d['pick']}", "", "**② 買い方（エントリー）**" if fam.side > 0 else "**② 売り方（エントリー）**", ""]
    L += [f"- {x}" for x in d["entry"]]
    L += ["", "**③ 手仕舞い（上から順に、先に当てはまったものを実行）**", ""]
    L += [f"{i}. {x}" for i, x in enumerate(d["exit"], 1)]
    L += ["", "**④ 金額**", "",
          f"- 1 銘柄あたり約 {yen_pos / 1e4:.0f}万円（100株単位で、この金額を超えない最大株数）。",
          f"- このルールで同時に持つのは最大 {MAX_POSITIONS} 銘柄。同じ銘柄の買い増し（ナンピン）はしない。",
          "",
          "**検証結果**（手数料・スリッページ・金利を引いた後。建玉枠 = 資金と同額で計算）", "",
          "| 期間 | 取引回数 | 勝率 | PF | 1回の平均 | 月平均 | 最大DD |",
          "|---|--:|--:|--:|--:|--:|--:|"]
    for label, key in (("学習期間", "is"), ("未使用データ", "oos"), ("全期間", "full")):
        m = ev[key]
        L.append(f"| {label} | {m['trades']} | {pct(m['win_rate'])} | {m['profit_factor']:.2f} | "
                 f"{pct(m['avg_ret'], 2, True)} | {man(m['monthly_mean'])} | {man(m['max_drawdown'], sign=False)} |")
    fw = ev.get("forward")
    if fw and fw["trades"] > 0:
        L.append(f"| **台帳掲載後**（{fw['since']}〜） | {fw['trades']} | {pct(fw['win_rate'])} | {fw['profit_factor']:.2f} | "
                 f"{pct(fw['avg_ret'], 2, True)} | {man(fw['monthly_mean'])} | {man(fw['max_drawdown'], sign=False)} |")
    wf = rule.get("walk_forward")
    if wf:
        L += ["", f"ウォークフォワード（毎年それまでのデータで選び直して翌年に試す）: PF {wf['profit_factor']:.2f}、"
                  f"{wf['years']} 年中 {wf['positive_years']} 年がプラス、月平均 {man(wf['monthly_mean'])}。"]
    L += ["", "**採用基準チェック**", "", "| 基準 | 結果 | 必要な水準 | 判定 |", "|---|--:|--:|:-:|"]
    for g in rule["gates"]:
        v, th = fmt_gate(g)
        L.append(f"| {g['name']} | {v} | {th} | {'○' if g['ok'] else '**×**'} |")
    L += ["", "<details><summary>年ごとの成績</summary>", "", "| 年 | 損益 | 取引回数 | 勝率 | プラスの月 |", "|---|--:|--:|--:|--:|"]
    for y in ev["yearly"]:
        L.append(f"| {y['year']} | {man(y['pnl'])} | {y['trades']} | {pct(y['win_rate'])} | {y['positive_months']}/{y['months']} |")
    L += ["", "</details>", ""]
    return L


def build_rules_md(state: dict, history: list[dict]) -> str:
    cfg = state["config"]
    L = ["# 売買ルールブック", "",
         f"> 自動生成（手で編集しないでください）。最終更新 {state['updated_at']}・株価データ {state['data_through']} まで・"
         f"前回のルール探索 {state.get('last_research', '-')}", "",
         "📈 [今日のシグナル](SIGNALS.md)｜🧾 [実トレード分析](trades/REPORT.md)｜🕘 [更新履歴](rules/CHANGELOG.md)", "",
         "## いまの結論", ""]
    L += headline(state)
    L += ["", "## 毎日の手順（スマホだけで完結）", ""]
    L += [f"{i}. {x}" for i, x in enumerate(texts.routine(), 1)]
    L += ["", "## 共通ルール（すべてのルールに優先）", ""]
    L += [f"- {x}" for x in texts.common(cfg)]
    L += ["", "## ルール一覧", ""]
    order = {"active": 0, "watch": 1, "suspended": 2}
    rules = sorted(state["rules"], key=lambda r: order[r["status"]])
    if not rules:
        L += ["現在、台帳に載っているルールはありません。", ""]
    for n, r in enumerate(rules, 1):
        L += rule_section(state, r, n)

    for key, title in (("portfolio", "採用ルールをまとめて運用した場合"), ("portfolio_watch", "観察中ルールをまとめて運用した場合（参考）")):
        pf = state.get(key) or {}
        if pf.get("empty", True):
            continue
        s = pf["summary"]
        L += [f"## {title}", "",
              f"対象: {'、'.join(FAMILIES[i].name for i in pf['rules'])}。"
              f"過去の最大ドローダウンが資金の {cfg['portfolio']['max_drawdown_pct']:.0%} に収まるよう、"
              f"1 銘柄あたり約 {pf['position_yen'] / 1e4:.0f}万円に調整した結果です。", "",
              f"- 月平均 **{man(s['monthly_mean'])}**（中央値 {man(s['monthly_median'])}）／目標比 {s['target_ratio']:.0%}",
              f"- プラスの月 {pct(s['positive_months'])}、月 {cfg['monthly_target'] / 1e4:.0f}万円以上だった月 {pct(s['target_months'])}",
              f"- 最悪の月 {man(s['monthly_worst'])}、最良の月 {man(s['monthly_best'])}、最大ドローダウン {man(s['max_drawdown'], sign=False)}",
              f"- 取引 {s['trades']} 回、勝率 {pct(s['win_rate'])}、PF {s['profit_factor']:.2f}",
              f"- 直近 12 か月の月平均 {man(pf['last12']['monthly_mean'])}",
              f"- 資金を使っている日は全体の {pct(pf['exposure']['days_invested'])}（平均建玉は資金の {pct(pf['exposure']['avg'])}）",
              "", "| 年 | 損益 | 月平均 | プラスの月 | 取引回数 |", "|---|--:|--:|--:|--:|"]
        for y in pf["yearly"]:
            L.append(f"| {y['year']} | {man(y['pnl'])} | {man(y['monthly_mean'])} | {y['positive_months']}/{y['months']} | {y['trades']} |")
        L.append("")

    if state["rejected"]:
        L += ["## 不採用になった型", "",
              "検証して基準に届かなかった型です。「試したが使えなかった」ことも情報なので残します。毎週、最新データで再検証されます。", "",
              "| 型 | 試した組み合わせ | 最良候補の PF（学習 → 未使用） | 未達だった基準 |", "|---|--:|--:|---|"]
        for r in state["rejected"]:
            fam = FAMILIES[r["family"]]
            ev = r.get("evidence")
            pf_txt = f"{ev['is']['profit_factor']:.2f} → {ev['oos']['profit_factor']:.2f}" if ev else "-"
            failed = "、".join(g["name"] for g in r["gates"] if not g["ok"]) or (r.get("note") or "-")
            L.append(f"| {fam.name} | {r['combos_tested']} | {pf_txt} | {failed} |")
        L.append("")

    L += ["## 検証方法と、その限界", ""]
    L += [f"- {x}" for x in texts.method(cfg)]
    L += ["", "## 最近の更新", ""]
    if history:
        for e in history[:10]:
            L.append(f"- {e['date']} **[{e['type']}]** {e['title']}")
        L += ["", "全履歴は [rules/CHANGELOG.md](rules/CHANGELOG.md)。"]
    else:
        L.append("まだ更新履歴はありません。")
    return "\n".join(L) + "\n"


def _order_lines(orders: list[dict]) -> list[str]:
    L = []
    for o in orders:
        L.append(f"#### {o['code']} {o['name']} — {o['side']}（{o['rule_name']}）")
        if o["order"] == "指値":
            L.append(f"- 注文: **指値 {o['limit_price']:,.0f}円 × {o['shares']:,}株**（本日中）約 {o['amount'] / 1e4:.0f}万円")
        else:
            L.append(f"- 注文: **成行（寄付） × {o['shares']:,}株** 約 {o['amount'] / 1e4:.0f}万円")
        L.append(f"- 前日終値 {o['ref_close']:,.1f}円")
        if o.get("stop_pct"):
            L.append(f"- 約定したら損切りの逆指値: 約定値から {o['stop_pct']:.0%}（目安 {o['stop_price_est']:,.0f}円）")
        if o.get("max_hold", 0) == 0:
            L.append("- 約定したら当日の引けで決済（引成）")
        else:
            L.append(f"- 期限: 約定から {o['max_hold']} 営業日目の引け（それまでにルールの決済条件が出れば、その翌朝の寄付で決済）")
        L.append("")
    return L


def _position_lines(positions: list[dict]) -> list[str]:
    L = []
    for p in positions:
        if p["exit_next_open"]:
            action = "**次の営業日の寄付で成行決済**（決済シグナル点灯）"
        elif p["days_left"] <= 1:
            action = "**次の営業日の引けで時間切れ決済**（引成）"
        else:
            action = f"保有継続（時間切れまであと {p['days_left']} 営業日）"
        L.append(f"- {p['code']} {p['name']} {p['side']} {p['shares']:,}株 @ {p['entry_price']:,.0f}円（{p['entry_date']}）"
                 f" 含み損益 {man(p['unrealized'])} → {action}"
                 + (f"／逆指値 {p['stop_price']:,.0f}円" if p.get("stop_price") else ""))
    return L


def build_signals_md(state: dict, signals: dict) -> str:
    d = signals["data_date"]
    nxt = next_business_day(d)
    L = [f"# 今日のシグナル（{nxt} の注文）", "",
         f"> {d} の終値で判定。注文日は次の営業日（{nxt}。祝日ならその次の営業日）。生成 {signals['generated_at']}", "",
         "📘 [ルールブック](RULES.md)", ""]
    gen, dd = date.fromisoformat(signals["generated_at"]), date.fromisoformat(d)
    if (gen - dd).days > 4:
        L += [f"> ⚠️ 株価データが {d} で止まっています。データ更新が失敗している可能性があります。この一覧で発注しないでください。", ""]
    for key, title, note in (
        ("active", "🟢 採用ルール — 実際に発注する", None),
        ("watch", "🟡 観察中ルール — 仮想売買（発注しない。記録だけ）",
         "統計面の基準は通過、リスク面の基準が未達のルールです。実際の資金は入れず、結果だけ追います。"),
    ):
        sig = signals[key]
        L += [f"## {title}", ""]
        if note and (sig["orders"] or sig["positions"]):
            L += [note, ""]
        L += ["### 次の営業日に出す注文", ""]
        if sig["orders"]:
            L += _order_lines(sig["orders"])
        else:
            n_rules = sum(1 for r in state["rules"] if r["status"] == key)
            L += ["該当するルールがありません。" if n_rules == 0 else "今日は条件に合う銘柄がありません。何もしない日です。", ""]
        L += ["### 保有中の建玉の扱い（ルール通りに売買していた場合）", ""]
        L += _position_lines(sig["positions"]) if sig["positions"] else ["保有なし。"]
        L.append("")
    sus = [r for r in state["rules"] if r["status"] == "suspended"]
    if sus:
        L += ["## ⏸ 一時停止中のルール", ""] + [f"- {FAMILIES[r['family']].name}: " + " / ".join(r["health"]["reasons"]) for r in sus] + [""]
    return "\n".join(L) + "\n"


def _stats_row(label: str, s: dict) -> str:
    return (f"| {label} | {s['trades']} | {pct(s['win_rate'])} | {s['profit_factor']:.2f} | {man(s['total_pnl'])} | "
            f"{man(s['avg_win'])} | {man(s['avg_loss'])} |")


def build_trades_md(a: dict | None) -> str:
    L = ["# 実トレード分析", ""]
    if not a:
        L += ["まだ取引履歴が取り込まれていません。", "",
              "## 取り込み方", ""]
        L += [f"{i}. {x}" for i, x in enumerate(texts.import_howto(), 1)]
        L += ["",
              "手入力する場合は `trades/manual.csv` に 1 行ずつ追記（列: `date,code,side,qty,price,fee,name,note`、side は buy / sell / short / cover）。", ""]
        return "\n".join(L) + "\n"
    o = a["overall"]
    L += [f"> 対象期間 {a['period']['from']} 〜 {a['period']['to']}・約定 {a['executions']} 件・決済済み {o['trades']} 取引・生成 {a['generated_at']}", "",
          "## 気づき", ""]
    L += [f"- {t}" for t in a["insights"]] or ["- まだ十分な取引数がありません（10 取引以上で分析コメントが出ます）。"]
    L += ["", "## 月ごとの実現損益", "", f"目標は月 {a['target'] / 1e4:.0f}万円。月平均 **{man(a['monthly_mean'])}**。", "",
          "| 月 | 実現損益 | 目標比 |", "|---|--:|--:|"]
    for m in a["monthly"][-24:]:
        L.append(f"| {m['month']} | {man(m['pnl'])} | {m['pnl'] / a['target']:.0%} |")
    L += ["", "## 成績の内訳", "", "| 区分 | 取引 | 勝率 | PF | 損益 | 平均利益 | 平均損失 |", "|---|--:|--:|--:|--:|--:|--:|",
          _stats_row("全体", o),
          _stats_row("ルール通り", a["by_rule"]["on"]), _stats_row("ルール外", a["by_rule"]["off"]),
          _stats_row("買い", a["by_direction"]["long"]), _stats_row("空売り", a["by_direction"]["short"])]
    L += [_stats_row(k, s) for k, s in a["by_holding"].items()]
    if a["per_rule"]:
        L += ["", "## ルール別（ルール通りに建てた取引）", "",
              "| ルール | 取引 | 勝率 | 実際の損益 | ルール通りに決済した場合 | 指値を待てなかった回数 |", "|---|--:|--:|--:|--:|--:|"]
        for r in a["per_rule"]:
            L.append(f"| {r['name']} | {r['trades']} | {pct(r['win_rate'])} | {man(r['actual_pnl_compared'])} | "
                     f"{man(r['rule_pnl_compared'])} | {r['limit_missed']} |")
    if a["open_lots"]:
        L += ["", "## 未決済の建玉", ""]
        L += [f"- {x['code']} {x['name']} {'買い' if x['direction'] == 'long' else '空売り'} {x['qty']:,.0f}株 @ {x['entry_price']:,.0f}円（{x['entry_date']}）"
              for x in a["open_lots"]]
    L += ["", "## 直近の取引", "", "| 決済日 | 銘柄 | 売買 | 株数 | 建値 → 決済値 | 損益 | 日数 | ルール |", "|---|---|---|--:|--:|--:|--:|---|"]
    for t in a["recent"]:
        rule = FAMILIES[t["rule_id"]].name if t["rule_id"] in FAMILIES else "ルール外"
        L.append(f"| {t['exit_date']} | {t['code']} {t['name']} | {'買い' if t['direction'] == 'long' else '空売り'} | {t['qty']:,.0f} | "
                 f"{t['entry_price']:,.0f} → {t['exit_price']:,.0f} | {man(t['pnl'])} | {t['days']} | {rule} |")
    if a.get("unmatched_closes"):
        L += ["", f"※ 建てた記録が見つからない決済が {a['unmatched_closes']} 件ありました（取り込み期間より前に建てた分）。集計から除いています。"]
    return "\n".join(L) + "\n"
