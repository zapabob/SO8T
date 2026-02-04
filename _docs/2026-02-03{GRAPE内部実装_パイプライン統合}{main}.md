# GRAPE内部実装_パイプライン統合

- Date: 2026-02-03
- Worktree: main

## 実施内容
- GRAPE rotary embedding を内部モジュール化 (scripts/models/grape_position_encoding.py)
- EnhancedMoonshotPipeline の GRAPE パッチを新モジュール経由に統一
- 内部パッチ処理で patched_modules を記録

## 次のアクション
- GRAPE Additive 版(ALiBi/FoX)の選択肢追加
- SFT→GRPO→mHC統合スモークの再実行
