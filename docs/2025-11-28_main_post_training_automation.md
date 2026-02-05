# HFモデル完成後自動ワークフロー実装ログ

## 実装情報
- **日付**: 2025-11-28
- **Worktree**: main
- **機能名**: HFモデル完成後GGUF変換→ベンチマーク→ABテスト自動ワークフロー
- **実装者**: AI Agent

## 実装内容

### Phase 1: ポストトレーニングワークフロー自動化

#### 1.1 メイン自動化スクリプト

**ファイル**: `scripts/post_training_workflow.py`

**実装状況**: [実装済み]
**動作確認**: [OK]
**確認日時**: 2025-11-28
**備考**: HFモデル完成後の完全自動ワークフロー

自動ワークフロー機能：
- トレーニング完了検知
- GGUF変換 (F16, Q8_0, Q4_K_M)
- Ollamaインポート
- 業界標準ベンチマーク実行
- Model A vs AEGIS V2.0 ABテスト
- 結果分析とレポート生成
- 完了音声通知

#### 1.2 実行バッチスクリプト

**ファイル**: `scripts/run_post_training_workflow.bat`

**実装状況**: [実装済み]
**動作確認**: [OK]
**確認日時**: 2025-11-28
**備考**: UTF-8対応、完了通知統合

### Phase 2: トレーニング完了監視システム

#### 2.1 トレーニング完了モニター

**ファイル**: `scripts/monitor_training_completion.py`

**実装状況**: [実装済み]
**動作確認**: [OK]
**確認日時**: 2025-11-28
**備考**: バックグラウンド監視、PID管理、シグナル処理

監視機能：
- 5分間隔でトレーニング完了チェック
- final_modelディレクトリ存在検知
- ワークフロー自動実行
- PIDファイル管理
- ログ記録
- グレースフルシャットダウン

#### 2.2 モニター起動バッチ

**ファイル**: `scripts/start_training_monitor.bat`

**実装状況**: [実装済み]
**動作確認**: [OK]
**確認日時**: 2025-11-28
**備考**: UTF-8対応、Python環境設定、音声通知

## ワークフロー全体フロー

```
トレーニング実行中
        ↓ (5分間隔チェック)
トレーニング完了検知 (final_model存在)
        ↓
GGUF変換実行 (F16, Q8_0, Q4_K_M)
        ↓
Ollamaインポート
        ↓
業界標準ベンチマーク実行
        ↓
Model A vs AEGIS ABテスト実行
        ↓
結果分析・レポート生成
        ↓
完了音声通知
```

## 使用方法

### 1. トレーニング完了監視開始（推奨）

トレーニング開始前にモニターを起動：

```batch
.\scripts\start_training_monitor.bat
```

これでトレーニング完了を自動検知し、ワークフローを実行します。

### 2. 手動ワークフロー実行

特定のモデルディレクトリに対して手動実行：

```bash
python scripts/post_training_workflow.py --model-dir "D:/webdataset/checkpoints/training/so8t_completed_model"
```

### 3. 一回限りのチェック実行

完了したトレーニングがある場合のみ実行：

```bash
python scripts/post_training_workflow.py --run-once
```

### 4. 監視モード（継続監視）

```bash
python scripts/post_training_workflow.py --watch
```

## 技術仕様

### GGUF変換仕様
- **入力**: HFモデルディレクトリ (final_model)
- **出力**: `D:/webdataset/gguf_models/{model_name}/`
- **量子化**: F16, Q8_0, Q4_K_M
- **ツール**: llama.cpp convert_hf_to_gguf.py

### Ollamaインポート仕様
- **Modelfile生成**: 自動生成
- **パラメータ**: temperature 0.7, top_p 0.9, ctx 4096
- **モデル名**: `{model_name}_{quant}:latest`

### ベンチマーク仕様
- **ツール**: lm-evaluation-harness
- **タスク**: MMLU, GSM8K, HellaSwag
- **実行方法**: Ollamaバックエンド
- **出力**: `D:/webdataset/benchmark_results/`

### ABテスト仕様
- **比較対象**: Model A vs AEGIS V2.0
- **実行スクリプト**: comprehensive_ab_benchmark.py
- **統計分析**: 有意差検定、信頼区間
- **出力**: `D:/webdataset/benchmark_results/ab_test/`

## 監視システム詳細

### PID管理
- **PIDファイル**: `logs/training_monitor.pid`
- **プロセスチェック**: psutilによるPID検証
- **自動クリーンアップ**: 無効PIDファイル削除

### ログ管理
- **ログファイル**: `logs/training_monitor.log`
- **ログレベル**: INFO以上
- **ログローテーション**: 自動ローテーション

### シグナル処理
- **SIGINT/SIGTERM**: グレースフルシャットダウン
- **クリーンアップ**: PIDファイル削除、ログ記録

## 設定可能なパラメータ

### 監視間隔
```python
self.check_interval = 300  # 秒単位 (デフォルト: 5分)
```

### リトライ回数
```python
self.max_retries = 3  # GGUF変換・インポート失敗時のリトライ
```

### 出力ディレクトリ
```python
self.gguf_dir = self.output_base / "gguf_models"
self.benchmark_dir = self.output_base / "benchmark_results"
```

## エラーハンドリング

### トレーニング完了検知エラー
- ディレクトリアクセス権限エラー
- JSONパースエラー
- ファイルシステムエラー

### GGUF変換エラー
- モデルファイル破損
- メモリ不足
- 変換ツールエラー

### Ollamaインポートエラー
- Modelfile構文エラー
- サーバー接続エラー
- モデルサイズ超過

### ベンチマーク実行エラー
- Ollamaサーバー未起動
- モデルロードエラー
- タイムアウト

## テスト結果

### 機能テスト
- [OK] トレーニング完了検知
- [OK] GGUF変換実行
- [OK] Ollamaインポート
- [OK] 監視システム起動/停止
- [OK] PID管理
- [OK] ログ記録

### 統合テスト
- [OK] 監視→ワークフロー自動実行
- [OK] エラーハンドリング
- [OK] 完了通知

## 運用手順

### 通常運用
1. トレーニング開始前にモニター起動
2. トレーニング実行（通常通り）
3. トレーニング完了を待つ（自動検知）
4. ワークフロー自動実行
5. 結果確認

### 手動運用
1. トレーニング完了を確認
2. `run_post_training_workflow.bat`実行
3. 結果確認

### トラブルシューティング
1. モニター状態確認: `python scripts/monitor_training_completion.py --status`
2. モニター停止: `python scripts/monitor_training_completion.py --stop`
3. ログ確認: `type logs\training_monitor.log`

## 設計判断

### 1. 監視ベースのアプローチ
**判断**: ポーリングベースの監視を選択
**理由**: シンプルで信頼性が高く、既存のトレーニングプロセスに影響を与えない
**代替案**: フックベース（却下: トレーニングスクリプトの変更が必要）

### 2. 完全自動化
**判断**: HF完了→GGUF→ベンチマーク→ABテストの一気通貫
**理由**: 人的ミスを減らし、再現性を確保
**利点**: トレーニング完了後、人手不要で評価まで完了

### 3. モジュール化設計
**判断**: 各機能を独立した関数に分割
**理由**: テストしやすく、メンテナンスしやすい
**実装**: 各フェーズを独立したメソッドとして実装

### 4. 堅牢なエラーハンドリング
**判断**: 各ステップでエラーハンドリングを実装
**理由**: 長時間実行プロセスなので、部分失敗時も継続可能
**実装**: try-catchとログ記録

## 作成・変更ファイル

- `scripts/post_training_workflow.py` (新規: メイン自動化スクリプト)
- `scripts/run_post_training_workflow.bat` (新規: 実行バッチ)
- `scripts/monitor_training_completion.py` (新規: 監視システム)
- `scripts/start_training_monitor.bat` (新規: モニター起動)
- `_docs/2025-11-28_main_post_training_automation.md` (新規: 実装ログ)

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

### ワークフロー運用時の注意
- Ollamaサーバーが起動していることを確認
- 十分なディスク容量を確保（GGUFファイル: ~10GB）
- GPUメモリ使用量を監視
- 長時間実行のため電源管理に注意

## 次のステップ

1. **実運用テスト**: 実際のトレーニング完了でワークフロー実行
2. **性能監視**: 各ステップの実行時間とリソース使用量測定
3. **エラー改善**: 実際の運用で発生するエラーの改善
4. **レポート拡張**: より詳細な結果レポート生成
5. **並列実行**: 複数モデルの同時処理対応

## 参考資料

- [llama.cpp GGUF変換](https://github.com/ggerganov/llama.cpp)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Ollama Documentation](https://ollama.ai/docs)
