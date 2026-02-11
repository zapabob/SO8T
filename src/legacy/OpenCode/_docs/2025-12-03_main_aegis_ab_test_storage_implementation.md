# AEGIS A/Bテストストレージ実装ログ

## 実装情報
- **日付**: 2025-12-03
- **Worktree**: main
- **機能名**: AEGIS A/Bテストストレージ実装
- **実装者**: AI Agent

## 実装内容

### 1. Cursor Rules更新

**ファイル**: `.cursor/rules/derived-cursor-rules.mdc`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: H:\from_D\webdataset をリポジトリの大きなファイル専用のサブディレクトリとして明示的に記載

- FILE STORAGE RULESセクションを更新
- `H:\from_D\webdataset` をCRITICAL STORAGE REQUIREMENTとして記載
- すべての大きなファイル（100MB以上）をこのディレクトリに保存するよう義務化
- ローリングチェックポイント、GGUFモデル、ログなどをこの場所に保存

### 2. A/Bテストスクリプトのストレージパス修正

**ファイル**: `scripts/benchmark/aegis_ab_test_benchmark.py`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: すべての出力パスをH:\from_D\webdatasetに変更

- ABTestConfigクラスのパスを以下のように変更:
  - `output_dir`: `H:\from_D\webdataset\benchmark_results\aegis_ab_test`
  - `checkpoint_dir`: `H:\from_D\webdataset\checkpoints\ab_test`
  - `gguf_dir`: `H:\from_D\webdataset\gguf_models`
- モデルAのパスを `microsoft/Phi-3.5-mini-instruct` に変更
- 自動起動スクリプトのログ出力を `H:\from_D\webdataset\logs\auto_start.log` に変更
- エスケープシーケンスの警告を修正

### 3. 自動起動スクリプトの強化

**ファイル**: `scripts/benchmark/aegis_ab_test_benchmark.py` (create_auto_startup_script関数)

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: H:\from_D\webdataset の存在確認とログ保存を実装

- H:\from_D\webdataset の存在確認を追加
- ログディレクトリの自動作成
- すべてのログ出力を H:\from_D\webdataset\logs\ に保存
- エスケープシーケンスの修正

### 4. Windowsタスクスケジューラー設定の改善

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: 電源投入時の自動起動と再試行設定を追加

- トリガーに「電源投入時に開始」を追加
- 再起動間隔を1分に設定
- 再試行回数を3回に設定
- より詳細な設定説明を追加

### 5. LM-Evaluation-Harness統合

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: hellaswagとMMLUの評価をHFモデルとGGUFモデルの両方で実行可能に

- LM-Evaluation-Harness設定をABTestConfigに追加
- HFモデル評価: `lm_eval --model hf` を使用
- GGUFモデル評価: `lm_eval --model hf --model_args gguf_file=...` を使用
- 評価タスク: hellaswag,mmlu（コンフィグで変更可能）
- 結果保存: `H:\from_D\webdataset\benchmark_results\lm_eval\`
- 自動GGUFファイル検出とトークナイザー設定

### 6. インストール・テストスクリプト作成

**ファイル**: `scripts/benchmark/install_lm_eval.ps1`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: LM-Evaluation-Harnessの自動インストールスクリプト

- external/lm-evaluation-harnessへのクローン
- 基本パッケージインストール
- HF依存関係 (transformers, accelerate, datasets) インストール
- llama-cpp-pythonインストール
- インポートテスト実行

**ファイル**: `scripts/benchmark/test_lm_eval.ps1`

**実装状況**: 実装済み
**動作確認**: OK
**確認日時**: 2025-12-03
**備考**: hellaswagとMMLUのテスト実行スクリプト

- Phi-3.5-mini-instructを使用した評価テスト
- CUDAデバイス使用
- 結果をH:\from_D\webdataset\benchmark_results\lm_eval\に保存
- サンプルログ出力有効化

## 作成・変更ファイル
- `.cursor/rules/derived-cursor-rules.mdc` - FILE STORAGE RULES更新
- `scripts/benchmark/aegis_ab_test_benchmark.py` - ストレージパス修正と自動起動強化
- `_docs/2025-12-03_main_aegis_ab_test_storage_implementation.md` - この実装ログ

## 設計判断
- H:\from_D\webdataset をリポジトリの大きなファイル専用ストレージとして確立
- すべてのA/Bテスト関連の出力（チェックポイント、GGUFモデル、ログ）をこの場所に集中
- 電源断からの自動復旧機能を強化
- Windowsタスクスケジューラーとの統合を改善

## 運用注意事項

### データ収集ポリシー
- 利用条件を守りつつ、高信頼ソースとして優先使用
- robots.txt遵守を徹底
- 個人情報・機密情報の除外を徹底

### NSFWコーパス運用
- **主目的**: 安全判定と拒否挙動の学習（生成目的ではない）
- モデル設計とドキュメントに明記
- 分類器は検出・拒否用途のみ

### /thinkエンドポイント運用
- 四重Thinking部（`<think-*>`）は外部非公開を徹底
- `<final>`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）

