# 2026-02-04 実装ログ: GRAPE Additive選択肢 + Dry-Runスモーク

- Date: 2026-02-04
- Worktree: main

## 実施内容
- GRAPEのAdditive(ALiBi/FoX)選択肢を追加（モデル設定フラグ＋ALiBi傾き保持）
- GRAPEバリアントを環境変数/CLIで切替可能に拡張
- Dry-Runモードを追加し、モデルロード/変換をスキップ可能に
- 統合パイプラインのスモークテストスクリプトを追加

## 変更ファイル
- scripts/models/grape_position_encoding.py
- experiments/enhanced_moonshot_pipeline.py
- scripts/pipeline/integrated_moonshot_pipeline_2025_2026.py
- run_moonshot_pipeline_2025_2026.py
- scripts/tests/smoke_moonshot_pipeline.py

## 動作確認
- `py -3 scripts/tests/smoke_moonshot_pipeline.py --grape-variant additive`
  - SO8T_DRYRUN=1 で完走を確認

## 次の作業
- GRAPE Additive を実カーネルに反映するパッチ（attention bias適用）
- SFT→GRPO→mHC 統合の軽量スモーク（実モデル読み込み）
- HF CLI 実行で新規データセット取得
