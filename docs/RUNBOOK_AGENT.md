# SO8T 安全エージェント 運用マニュアル

## 概要

このドキュメントは、SO8T安全エージェントの運用に関する手順、設定、トラブルシューティングガイドを提供します。

## システム要件

### ハードウェア要件
- **GPU**: RTX3060級以上（12GB VRAM推奨）
- **RAM**: 32GB以上
- **ストレージ**: 50GB以上の空き容量
- **OS**: Windows 11, Ubuntu 20.04+, macOS 12+

### ソフトウェア要件
- Python 3.11+
- CUDA 12.0+ (GPU使用時)
- PyTorch 2.0+
- Transformers 4.35+
- llama.cpp (GGUF推論用)

## インストール手順

### 1. リポジトリのクローン
```bash
git clone <repository-url>
cd SO8T
```

### 2. 仮想環境の作成
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# または
.venv\Scripts\activate  # Windows
```

### 3. 依存関係のインストール
```bash
pip install -r requirements.txt
```

### 4. モデルのダウンロード
```bash
# ベースモデルのダウンロード
python scripts/download_model.py --model Qwen/Qwen2.5-7B-Instruct

# 学習済みSO8Tモデルのダウンロード（利用可能な場合）
python scripts/download_so8t_model.py --version latest
```

## 起動方法

### 1. 基本起動
```bash
python -m inference.agent_runtime --model-path checkpoints/so8t_qwen2.5-7b_sft_fp16
```

### 2. GGUFモデルでの起動
```bash
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-q4_k_m.gguf --use-gguf
```

### 3. 設定ファイルを使用した起動
```bash
python -m inference.agent_runtime --config configs/production.yaml
```

## 設定ファイル

### 基本設定 (configs/basic.yaml)
```yaml
model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  checkpoint_path: "checkpoints/so8t_qwen2.5-7b_sft_fp16"
  use_gguf: false
  gguf_path: null

inference:
  max_tokens: 2048
  temperature: 0.7
  top_p: 0.9
  safety_threshold: 0.8

logging:
  log_level: "INFO"
  log_file: "logs/agent_runtime.log"
  audit_log: "logs/audit.jsonl"
  max_log_size: "100MB"
  backup_count: 5

escalation:
  enable_notifications: true
  notification_webhook: null
  email_notifications: false
  escalation_timeout: 300  # seconds
```

### 本番環境設定 (configs/production.yaml)
```yaml
model:
  base_model: "Qwen/Qwen2.5-7B-Instruct"
  checkpoint_path: "dist/so8t_qwen2.5-7b-safeagent-q4_k_m.gguf"
  use_gguf: true

inference:
  max_tokens: 1024
  temperature: 0.5
  top_p: 0.8
  safety_threshold: 0.9

logging:
  log_level: "WARNING"
  log_file: "/var/log/so8t/agent.log"
  audit_log: "/var/log/so8t/audit.jsonl"
  max_log_size: "1GB"
  backup_count: 10

escalation:
  enable_notifications: true
  notification_webhook: "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
  email_notifications: true
  smtp_server: "smtp.company.com"
  escalation_timeout: 600
```

## ログ管理

### ログファイルの場所
- **アプリケーションログ**: `logs/agent_runtime.log`
- **監査ログ**: `logs/audit.jsonl`
- **エラーログ**: `logs/error.log`
- **パフォーマンスログ**: `logs/performance.log`

### ログローテーション
- 最大ファイルサイズ: 100MB（本番環境では1GB）
- バックアップ数: 5個（本番環境では10個）
- 圧縮: 古いログは自動的にgzip圧縮

### 監査ログの形式
```json
{
  "timestamp": "2025-01-27T10:30:00Z",
  "request_id": "req_12345",
  "user_id": "user_67890",
  "input": "ユーザーの要求テキスト",
  "decision": "ESCALATE",
  "rationale": "個人情報の開示要求のため、人間の判断が必要です",
  "human_required": true,
  "model_version": "so8t-qwen2.5-7b-v1.0",
  "processing_time_ms": 150,
  "confidence_score": 0.95
}
```

## エスカレーション処理

### エスカレーション時の通知先

#### 1. 医療関連
- **即座の通知**: 医療担当者、緊急時は救急車
- **連絡先**: medical@company.com
- **Slackチャンネル**: #medical-escalation

#### 2. 法務・コンプライアンス
- **通知先**: 法務部門、コンプライアンス担当
- **連絡先**: legal@company.com
- **Slackチャンネル**: #legal-escalation

#### 3. 人事・ハラスメント
- **通知先**: 人事部門、ハラスメント相談窓口
- **連絡先**: hr@company.com
- **Slackチャンネル**: #hr-escalation

#### 4. セキュリティ関連
- **通知先**: ITセキュリティ部門、CSIRT
- **連絡先**: security@company.com
- **Slackチャンネル**: #security-alerts

#### 5. その他
- **通知先**: 直属の上司、管理職
- **連絡先**: manager@company.com
- **Slackチャンネル**: #general-escalation

### 通知の設定
```bash
# Slack通知の設定
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"

# メール通知の設定
export SMTP_SERVER="smtp.company.com"
export SMTP_PORT="587"
export SMTP_USERNAME="so8t-agent@company.com"
export SMTP_PASSWORD="your-password"
```

## 監視とメトリクス

### 主要メトリクス
- **リクエスト数**: 1分間あたりの処理リクエスト数
- **レスポンス時間**: 平均・95パーセンタイル・99パーセンタイル
- **エラー率**: エラーの割合
- **エスカレーション率**: エスカレーションの割合
- **安全性スコア**: Refuse Recall, Escalate Precision

### 監視ダッシュボード
```bash
# メトリクス収集の開始
python -m monitoring.collect_metrics --config configs/monitoring.yaml

# ダッシュボードの起動
python -m monitoring.dashboard --port 8080
```

### アラート設定
- **高エラー率**: 5%を超えた場合
- **高レスポンス時間**: 95パーセンタイルが5秒を超えた場合
- **異常なエスカレーション率**: 通常の2倍を超えた場合
- **システムリソース不足**: CPU 90%以上、メモリ 90%以上

## トラブルシューティング

### よくある問題

#### 1. モデルの読み込みエラー
```bash
# エラー: CUDA out of memory
# 解決策: バッチサイズを小さくする、またはGGUFモデルを使用
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-q4_k_m.gguf --use-gguf --batch-size 1
```

#### 2. 推論速度が遅い
```bash
# 解決策: 量子化モデルを使用、またはGPU使用を確認
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-q4_k_s.gguf --use-gguf --device cuda
```

#### 3. ログファイルが大きくなりすぎる
```bash
# 解決策: ログローテーションの設定を確認
python -m scripts.rotate_logs --log-dir logs --max-size 100MB --backup-count 5
```

### ログの確認
```bash
# アプリケーションログの確認
tail -f logs/agent_runtime.log

# エラーログの確認
grep "ERROR" logs/agent_runtime.log

# 監査ログの確認
tail -f logs/audit.jsonl | jq .
```

### パフォーマンスの確認
```bash
# システムリソースの確認
python -m monitoring.check_resources

# モデルのパフォーマンステスト
python -m eval.eval_latency --model-path checkpoints/so8t_qwen2.5-7b_sft_fp16
```

## 再学習フロー

### 1. 監査ログからのデータ抽出
```bash
python -m training.extract_training_data --audit-log logs/audit.jsonl --output data/additional_training.jsonl
```

### 2. データの前処理
```bash
python -m training.preprocess_data --input data/additional_training.jsonl --output data/processed_training.jsonl
```

### 3. モデルの再学習
```bash
python -m training.train_qlora --config configs/retrain.yaml --data data/processed_training.jsonl
```

### 4. モデルの評価
```bash
python -m eval.eval_safety --model-path checkpoints/so8t_qwen2.5-7b_retrained --test-data data/test_set.jsonl
```

### 5. 本番環境へのデプロイ
```bash
# 新しいモデルをGGUFに変換
python -m scripts.convert_to_gguf --input checkpoints/so8t_qwen2.5-7b_retrained --output dist/

# 本番環境でのテスト
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-retrained-q4_k_m.gguf --use-gguf --test-mode
```

## セキュリティ考慮事項

### アクセス制御
- モデルファイルへのアクセスは制限する
- 監査ログは暗号化して保存
- APIキーや認証情報は環境変数で管理

### データプライバシー
- 個人情報はログから除外
- 監査ログは定期的に削除
- データの匿名化を実施

### 監査とコンプライアンス
- すべての判断を記録
- 定期的なセキュリティ監査
- コンプライアンス要件の遵守

## 緊急時の対応

### システム停止
```bash
# エージェントの停止
pkill -f "agent_runtime"

# 緊急時の設定で再起動
python -m inference.agent_runtime --config configs/emergency.yaml
```

### データのバックアップ
```bash
# 重要なデータのバックアップ
python -m scripts.backup_data --output backup/$(date +%Y%m%d_%H%M%S)
```

### インシデント対応
1. 問題の特定と影響範囲の確認
2. 必要に応じてシステムの停止
3. ログの分析と原因の特定
4. 修正の実施とテスト
5. システムの復旧と監視

---

**注意**: このマニュアルは定期的に更新されます。最新版は常にこのドキュメントで確認してください。緊急時は、このマニュアルに従って迅速に対応してください。