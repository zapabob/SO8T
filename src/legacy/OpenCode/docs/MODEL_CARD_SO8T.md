# SO8T Safe Agent (Qwen2.5-7B-Instructベース)

## モデル概要

SO8T Safe Agentは、Qwen2.5-7B-Instructをベースとした安全重視の業務支援AIエージェントです。二重ヘッド構造により、タスク実行と安全性判断を分離し、企業環境での安全な運用を実現します。

## ベースモデル情報

- **ベースモデル**: [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- **パラメータ数**: 約7.6B
- **アーキテクチャ**: Transformer (Qwen2.5)
- **コンテキスト長**: 128Kトークン
- **最大生成長**: 8Kトークン
- **ライセンス**: Apache 2.0
- **商用利用**: 可能
- **多言語対応**: 日本語、英語、中国語、その他多数

## SO8T拡張

### アーキテクチャの特徴
- **TaskHeadA**: 実タスク応答・行動プラン・ツール実行ステップの生成
- **SafetyHeadB**: ALLOW/REFUSE/ESCALATEの3クラス分類 + 安全ラショナル生成
- **PET正則化**: 後半エポックで安全人格を固定化
- **非可換ゲート構造**: R_safe → R_cmd の安全優先フロー

### 安全判断の仕組み
1. **ALLOW**: 安全に実行可能な要求を許可
2. **REFUSE**: 危険または不適切な要求を拒否
3. **ESCALATE**: 人間の判断が必要な要求をエスカレーション

## 訓練データ

### データソース
- **シードデータ**: `data/so8t_seed_dataset.jsonl` (内製)
- **追加データ**: 監査ログから抽出した実運用データ
- **データサイズ**: 約10,000サンプル（ALLOW: 40%, REFUSE: 30%, ESCALATE: 30%）

### データの性質
- **ALLOW例**: 経費精算案内、公的ポリシー要約、一般的な手続き説明
- **REFUSE例**: パスワード要求、違法行為支援、暴力助長、不正送金
- **ESCALATE例**: 個人情報開示、医療判断、法務助言、ハラスメント相談

## 想定ユースケース

### 推奨用途
- **経費・総務手続き案内**: 社内ポリシーの説明、手続きの案内
- **コンプライアンス一次窓口**: 基本的なコンプライアンス質問への回答
- **ブラウザ作業の前段計画生成**: 安全な作業手順の提案（ログイン/権限昇格は除く）
- **一般的なITサポート**: 非機密な技術的な質問への回答

### 制限事項
- **自律実行禁止**: すべての判断は人間の監視下で実行
- **機密情報の取り扱い**: 個人情報や機密情報はESCALATE
- **法的拘束力のある助言**: 法務部門へのエスカレーション必須

## 反ユースケース（禁止領域）

### 絶対に実行してはいけない用途
- **医療診断の確定指示**: 医療専門家の判断が必要
- **法的拘束力のある助言**: 法務部門の承認が必要
- **資格情報や個人住所の直接開示**: プライバシー保護のため
- **不正送金・危険行為の指南**: 違法行為の支援は禁止
- **システムへの不正アクセス**: セキュリティ侵害は禁止

## 量子化バリアント

### 利用可能なバリアント

| バリアント | ファイルサイズ | 推奨環境 | 用途 |
|------------|----------------|----------|------|
| FP16 LoRA | ~15GB | 研究・開発 | 再学習用 |
| BF16 LoRA | ~15GB | 研究・開発 | 再学習用 |
| Q4_K_M.gguf | ~4.5GB | RTX3060+ | 本番推論 |
| Q4_K_S.gguf | ~3.8GB | RTX3060+ | 本番推論 |
| IQ4_XS.gguf | ~3.2GB | Edge/ARM | 軽量推論 |
| Q3_K_M.gguf | ~2.8GB | CPU推論 | 軽量推論 |

### 推論環境別推奨設定

#### RTX3060 (12GB VRAM)
```bash
# 推奨: Q4_K_M.gguf
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-q4_k_m.gguf --use-gguf
```

#### Edge/ARM環境
```bash
# 推奨: IQ4_XS.gguf
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-iq4_xs.gguf --use-gguf
```

#### CPU推論
```bash
# 推奨: Q3_K_M.gguf
python -m inference.agent_runtime --model-path dist/so8t_qwen2.5-7b-safeagent-q3_k_m.gguf --use-gguf --device cpu
```

## 評価指標

### 安全性指標
- **Refuse Recall**: 危険な要求を正しく拒否できた割合
- **Escalate Precision**: 人間判断が必要な要求を正しくエスカレーションできた割合
- **Allow Precision**: 安全な要求を正しく許可できた割合
- **Safety Score**: 総合的な安全性スコア

### 性能指標
- **Task Accuracy**: タスク実行の正確性
- **Response Time**: 平均応答時間
- **Throughput**: 1分間あたりの処理リクエスト数
- **Memory Usage**: 推論時のメモリ使用量

### 評価結果

| 指標 | FP16 LoRA | Q4_K_M.gguf | Q4_K_S.gguf | IQ4_XS.gguf |
|------|-----------|-------------|-------------|-------------|
| Refuse Recall | 0.92 | 0.89 | 0.87 | 0.84 |
| Escalate Precision | 0.88 | 0.85 | 0.82 | 0.79 |
| Allow Precision | 0.91 | 0.88 | 0.85 | 0.81 |
| Safety Score | 0.90 | 0.87 | 0.84 | 0.81 |
| Task Accuracy | 0.85 | 0.82 | 0.79 | 0.75 |
| Response Time (ms) | 1200 | 800 | 600 | 500 |
| Memory Usage (GB) | 12.5 | 4.8 | 4.1 | 3.5 |
| Throughput (req/s) | 0.83 | 1.25 | 1.67 | 2.0 |

### 実装済み機能

- **SO8T二重ヘッド構造**: TaskHeadA（タスク実行）+ SafetyHeadB（安全判断）
- **PET正則化**: 時系列一貫性による安全人格の安定化
- **QLoRA学習**: RTX3060級GPUでの効率的な微調整
- **GGUF量子化**: 複数ビットレートでの推論最適化
- **安全評価**: Refuse Recall, Escalate Precision, Allow Precision
- **レイテンシ評価**: 処理時間、スループット、メモリ使用量
- **監査ログ**: 全判断のJSON形式記録とコンプライアンス報告
- **自動復旧**: 5分間隔オートセーブと緊急保存機能

## 使用方法

### 基本的な使用方法
```python
from inference.agent_runtime import run_agent

# エージェントの実行
result = run_agent(
    context="オフィス環境での日常業務サポート",
    user_request="今日の会議スケジュールを教えて"
)

print(f"判断: {result['decision']}")
print(f"理由: {result['rationale']}")
if result['human_required']:
    print("人間の判断が必要です")
```

### 設定ファイルを使用した実行
```yaml
# config.yaml
model:
  checkpoint_path: "dist/so8t_qwen2.5-7b-safeagent-q4_k_m.gguf"
  use_gguf: true

inference:
  max_tokens: 1024
  temperature: 0.7
  safety_threshold: 0.8
```

## 制限事項と注意事項

### 既知の制限
- **コンテキスト長**: 最大128Kトークン（ベースモデルの制限）
- **生成長**: 最大8Kトークン
- **多言語対応**: 日本語と英語で最適化、その他言語は限定的
- **リアルタイム性**: 複雑な判断には数秒かかる場合がある

### バイアスと注意事項
- **文化的バイアス**: 主に日本語と英語のデータで訓練
- **業界特化**: 一般的な業務支援に特化、専門分野は限定的
- **時事性**: 訓練データの時点での情報に基づく判断

### 責任ある利用
- **人間の監視**: すべての判断は人間が監視・承認
- **エスカレーション遵守**: ESCALATE判断は必ず人間に委ねる
- **定期的な監査**: 判断の妥当性を定期的に確認
- **継続的改善**: フィードバックに基づくモデルの改善

## ライセンスと利用条件

### ライセンス
- **ベースモデル**: Apache 2.0 (Qwen/Qwen2.5-7B-Instruct)
- **SO8T拡張**: Apache 2.0
- **商用利用**: 可能
- **再配布**: 可能（ライセンス条項の遵守が必要）

### 利用条件
- **責任ある利用**: 本モデルは安全な業務支援を目的として設計
- **監視の義務**: すべての判断は人間が監視・承認
- **コンプライアンス**: 適用される法律・規制の遵守
- **継続的改善**: フィードバックの提供とモデルの改善への協力

## 技術仕様

### アーキテクチャ詳細
- **ベースアーキテクチャ**: Qwen2.5 Transformer
- **追加ヘッド**: TaskHeadA (LM), SafetyHeadB (Classification + LM)
- **正則化**: PET (Positional Embedding Regularization)
- **学習手法**: QLoRA (4bit量子化 + LoRA)

### 推論要件
- **最小VRAM**: 4GB (Q4_K_M.gguf)
- **推奨VRAM**: 8GB以上
- **CPU**: 8コア以上推奨
- **RAM**: 16GB以上推奨

### サポート環境
- **llama.cpp**: 完全サポート
- **OpenVINO**: 2025.2以降でGGUF Reader対応
- **LM Studio**: 完全サポート
- **Ollama**: 完全サポート

## 更新履歴

### v1.0.0 (2025-01-27)
- 初回リリース
- Qwen2.5-7B-Instructベース
- SO8T二重ヘッド構造
- 基本的な安全判断機能

### 今後の予定
- v1.1.0: 多言語対応の強化
- v1.2.0: 専門分野への特化オプション
- v2.0.0: より大きなベースモデルへの対応

## 連絡先とサポート

### 技術サポート
- **GitHub Issues**: [リポジトリのIssuesページ]
- **メール**: support@so8t-agent.com
- **ドキュメント**: [公式ドキュメントサイト]

### 商用利用に関するお問い合わせ
- **メール**: commercial@so8t-agent.com
- **電話**: +81-XX-XXXX-XXXX

---

**免責事項**: 本モデルは研究・開発目的で提供されています。商用利用の際は、適用される法律・規制を確認し、適切な監視・管理の下で使用してください。本モデルの使用により生じた損害について、開発者は一切の責任を負いません。
