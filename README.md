# kabu — 検証済みの売買ルールを作り、育てる

過去データで検証を通ったルールだけを「売買ルールブック」に載せ、毎日・毎週の自動実行で最新の状態に保つためのツールです。

- 📘 **[RULES.md](RULES.md)** — いまのルール（言葉で書かれた、そのまま実行できる手順）と検証結果
- 📈 **[SIGNALS.md](SIGNALS.md)** — 次の営業日に出す注文の一覧（毎日更新）
- 🧾 **[trades/REPORT.md](trades/REPORT.md)** — 実トレードの分析
- 🕘 **[rules/CHANGELOG.md](rules/CHANGELOG.md)** — ルールの採用・更新・停止・廃止の履歴
- 📱 `docs/index.html` — 上記を 1 ページにまとめたスマホ用ダッシュボード

前提（`config.yaml` で変更可）: 運用資金 300万円・目標 月20万円・楽天証券・信用買い/空売りあり・日足ベースのデイ〜スイング。

## 大事な前提

**「月◯万円を最低保証するルール」は存在しません。** このツールがやるのは、
目標に対して検証済みルールが過去データ上どこまで届いているかを、割り引かずに示すことです。
届いていなければ「届いていない」と表示します（RULES.md の「いまの結論」）。

## 仕組み

```
株価取得 ─→ ルール探索 ─→ 検証（採点）─→ 台帳 ─→ シグナル ─→ 実トレード ─→ 分析
 (日次)      (週次)        (週次)      (自動更新)   (日次)     (あなた)     (自動)
                 ↑                                                          │
                 └──────────── 実測コストを検証に反映 ←──────────────────────┘
```

### ルールの 3 つの状態

| 状態 | 意味 | 扱い |
|---|---|---|
| 🟢 採用 | すべての採用基準を通過 | 実際の資金で売買する |
| 🟡 観察中 | 統計的な優位性の基準は通過、リスク面などが未達 | 仮想売買（記録だけ） |
| 不採用 | 基準未達 | 売買しない（毎週再検証は続ける） |

採用中のルールも、直近の成績が崩れると自動で ⏸ 一時停止 になります。

### 検証のしかた（要点）

1. **先読みなし**: 判定は前日終値まで、注文は翌営業日。指値は「割り込んだ日だけ約定」。
2. **学習と採点を分ける**: パラメータは 2022 年より前のデータだけで選び、2022 年以降の未使用データで採点。
3. **偶然の当たりを避ける**: 全組み合わせの 1 位ではなく、パラメータごとの中央値で選ぶ。隣のパラメータも黒字か確認。
4. **ウォークフォワード**: 毎年「それまでのデータで選ぶ → 翌年で試す」を繰り返して、選び方自体を検証。
5. **コスト込み**: スリッページ・信用金利・貸株料を差し引く。実トレードの実測値が想定より悪ければそちらを使う。

採用基準の数値は `config.yaml` の `validation:` にあります。

## 使い方（スマホ）

1. 夕方以降に **SIGNALS.md**（または `docs/index.html`）を開く
2. 翌朝、楽天証券 iSPEED で一覧どおりに注文
3. 週末に取引履歴 CSV を `trades/inbox/` に入れる（GitHub のブラウザ画面から「Add file → Upload files」）

GitHub のスマホアプリなら RULES.md / SIGNALS.md がそのまま読めます。
ダッシュボードを使う場合は、リポジトリの Settings → Pages で「Deploy from a branch / main / `/docs`」を選びます
（非公開リポジトリの Pages は有料プランが必要。公開する場合は `config.yaml` の `site.include_real_trades: false` を推奨）。

## 自動実行（GitHub Actions）

| ワークフロー | いつ | 何をする |
|---|---|---|
| `daily.yml` | 平日 16:45 JST | 株価更新 → ルールの健康診断 → シグナル生成 → コミット |
| `weekly.yml` | 土曜 06:00 JST | 全ルールを再探索・再検証 → 台帳を更新 → コミット |
| `trades.yml` | `trades/inbox/` か `trades/manual.csv` への push | 取引履歴の取り込み・分析 |
| `ci.yml` | コード変更時 | テスト |

リポジトリの Settings → Actions → General → Workflow permissions を **Read and write** にしてください。

## 手元で動かす

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python -m kabu update-data          # 株価を取得（約1,000銘柄、数分）
python -m kabu research             # [週次] 探索・検証・台帳更新（数分〜十数分）
python -m kabu daily                # [日次] 株価更新・健康診断・シグナル
python -m kabu import-trades        # trades/inbox/*.csv と trades/manual.csv を取り込んで分析
python -m kabu report               # RULES.md などを再生成
python -m pytest -q                 # テスト
```

### 自分のルール案を試す

思いついた条件は、売買する前に過去データで確認できます。

```bash
python -m kabu backtest dip_swing --params '{"rsi_th": 10, "max_hold": 5, "limit_pct": 0.015}'
```

型（ファミリー）の一覧: `dip_swing`（押し目買い）/ `panic_dip`（全体急落時のリバウンド）/ `dip_day`（急落拾いデイトレ）/
`rip_short`（急騰の逆張り空売り）/ `breakout_long`（高値ブレイク）/ `news_long`（好材料の急伸）/ `breakdown_short`（安値ブレイク空売り）。

新しい型を足すには `kabu/strategies.py` に `Family` を 1 つ追加します（条件の作り方・探索するパラメータ・日本語の説明文の 3 点）。
追加した型は次回の `research` から自動で検証対象になります。

## 実トレードの記録

- **楽天証券**: Web の「取引履歴 → 国内株式」から CSV を保存し、`trades/inbox/` に置く。現物・信用新規・信用返済に対応。
- **手入力**: `trades/manual.csv` に追記（`date,code,side,qty,price,fee,name,note`、side は `buy` / `sell` / `short` / `cover`）。

同じ CSV を何度入れても重複しません。分析では、各取引がルール通りだったか、ルール通りに決済していたらいくらだったか、
寄成の約定がどれだけ滑ったか、などを出します。

## 構成

```
config.yaml            設定（資金・目標・コスト・採用基準）
kabu/
  data/                銘柄マスタ（JPX）・株価取得（Yahoo Finance）・データ不良の除去
  indicators.py        テクニカル指標
  strategies.py        ルールの型（条件・探索範囲・日本語の説明）
  engine.py            バックテストエンジン
  research.py          探索・パラメータ選択・ウォークフォワード・採用基準
  rulebook.py          ルール台帳と自動更新（入れ替え・昇格・降格・停止・廃止）
  portfolio.py         複数ルールの合算成績・建玉サイズ・シグナル
  trades/              実トレードの取り込み・往復取引化・分析
  report/              RULES.md / SIGNALS.md / ダッシュボードの生成
rules/                 台帳（active.json）・探索結果・更新履歴・シグナル
trades/                取引履歴（inbox / manual.csv / ledger.csv）と分析
docs/                  スマホ用ダッシュボード
tests/                 テスト
```

## 既知の限界

- **生存バイアス**: 現在上場している銘柄だけで検証しているため、成績はやや良く出ます。
- **データ元**: Yahoo Finance の非公式 API です。提供元の仕様変更で取得できなくなる可能性があります（その場合 SIGNALS.md に警告が出ます）。
- **日足のみ**: 分足を使うデイトレは検証対象外です（無料で長期の分足が手に入らないため）。
- **楽天証券 CSV**: 列の並びは公開されている形式に合わせていますが、実際のファイルでの確認はまだです。読めない列があればエラーで知らせます。
- 投資判断はご自身の責任で行ってください。
