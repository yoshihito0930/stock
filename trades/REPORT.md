# 実トレード分析

まだ取引履歴が取り込まれていません。

## 取り込み方

1. 楽天証券（Web）→ 口座管理 → 取引履歴 → 国内株式 → 期間を指定して **CSV で保存**。
2. その CSV を、このリポジトリの `trades/inbox/` に入れて push する（スマホのブラウザで GitHub を開き「Add file → Upload files」でも可）。
3. GitHub Actions が自動で取り込み、分析を更新する。手元なら `python -m kabu import-trades`。

手入力する場合は `trades/manual.csv` に 1 行ずつ追記（列: `date,code,side,qty,price,fee,name,note`、side は buy / sell / short / cover）。

