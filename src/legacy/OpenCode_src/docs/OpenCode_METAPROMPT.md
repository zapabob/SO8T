# OpenCode Worktree - 運用ガイドライン

このドキュメントは `git worktree` で作成された OpenCode フォルダ専用の運用指針です。

## ワークツリー構成

| ブランチ | 目的 | 場所 |
|---------|------|------|
| main | 安定運用（ブート自動化、再現可能ベンチ、HF/GH公開） | 親ディレクトリ |
| OpenCode | 実験・検証・大規模改修・外部研究統合 | `OpenCode/` フォルダ |

## 業務範囲

OpenCode ワークツリーで扱う作業:

1. **自動ランチャー運用**: `scripts/utils/boot_pipeline_launcher.py`
2. **ABC レポート再生成**: `scripts/analysis/abc_summary_report.py`
3. **日英ドキュメント整備**: `docs/` 配下の Markdown/LaTeX
4. **データ参照**: `data/` (symlink)、`datasets/`、`results/`

## 実行コマンド集

### パイプライン起動（SQL 追跡付き）
```bash
cd OpenCode
py -3 scripts/utils/boot_pipeline_launcher.py
```

### 進捗モニタ（SQL 表示付き）
```bash
cd OpenCode
py -3 scripts/utils/monitor_pipeline.py
```

### ABC ベンチマーク実行
```bash
cd OpenCode
py -3 scripts/evaluation/run_comprehensive_abc_benchmark.py
```

### ABC レポート生成
```bash
cd OpenCode
py -3 scripts/analysis/abc_summary_report.py
```

### SQL 進捗確認（CLI）
```bash
cd OpenCode
py -3 scripts/utils/pipeline_progress_store.py  # テスト出力
```

## チェックポイント

| 種類 | パス | 更新頻度 |
|-----|------|---------|
| 最新チェックポイント | `checkpoints/latest_checkpoint.json` | 学習時 |
| ローリングスナップショット | `checkpoints/rolling_snapshots/` | 5分ごと（3世代保持） |
| SQL 履歴 | `logs/pipeline_progress.sqlite` | あらゆるcheckpoint/ログ記録時 |

## マージルール

OpenCode → main へのマージは以下の条件を満たした場合にのみ実行:

1. 再現性が確認されている（seed、config、dataset version が記録されている）
2. ログ・ドキュメントが揃っている
3. テストが通過している
4. 破壊的変更が main に入らない

## 共有ファイルの管理

- 共通モジュールは **symlink** または **git submodule** で管理
- 巨大データファイルは LFS 追跡
- 秘密情報は `.env.local` のみ（コミット禁止）

## ステータス確認

```bash
# Git ワークツリー状態
git worktree list

# SQL データベース確認
cd OpenCode && python -c "
from scripts.utils.pipeline_progress_store import get_all_runs, init_db
init_db()
for r in get_all_runs():
    print(r['run_id'], r['status'])
"
```
