# SO8T Thinking Model 実装ログ

## 実装日時
2025-11-07

## 概要
SO8TモデルにThinking機能を統合しました。**四重推論アーキテクチャ**（Task/Safety/Policy/Final）を実装し、内部推論と最終回答を分離。Safety/Verifierヘッドによる安全ゲート、安全なデータ収集ポリシー、NSFW/危険コンテンツの統計的フィルタリング、情報リーク防止のデータ分割を実装しました。

## 実装ファイル

### 1. 特殊トークン定義とトークナイザー拡張
- `so8t-mmllm/src/models/thinking_tokens.py`: Thinking特殊トークンの定義とトークナイザー拡張機能
  - 基本形式: `<think>`, `</think>`, `<final>`, `</final>`
  - 四重推論形式: `<think-task>`, `</think-task>`, `<think-safety>`, `</think-safety>`, `<think-policy>`, `</think-policy>`, `<final>`, `</final>`

### 2. SO8TThinkingModel実装
- `so8t-mmllm/src/models/so8t_thinking_model.py`: `SafetyAwareSO8TModel`を継承し、Thinking出力形式をサポートするモデル

### 3. ユーティリティ
- `so8t-mmllm/src/utils/thinking_utils.py`: Thinking/Final抽出、Safety判定、Verifierスコア計算などのヘルパー関数

### 4. 安全なデータ収集
- `scripts/data/crawl_safe_sources.py`: Playwright + Chromiumを使用した安全なデータクローラー
  - robots.txt遵守
  - 危険コンテンツの自動フィルタリング
  - 公開情報のみ収集（Wikipedia、官公庁、技術ブログ等）

### 5. NSFW/危険コンテンツの統計的ラベリング
- `scripts/data/label_nsfw_content.py`: scikit-learnを使用した自動ラベリング
  - TF-IDF + LogisticRegressionによる分類器
  - 小規模手動ラベルから大規模コーパスへの自動拡張
  - 信頼度閾値による品質保証

### 6. 安全なデータセット分割
- `scripts/data/split_dataset_safe.py`: 情報リーク防止のデータ分割
  - ドメイン別層化分割
  - 同一URL/同一ソースからtrain/test両方に入らないよう制御

### 7. データセット作成
- `scripts/data/create_thinking_dataset.py`: 既存データセットのThinking形式への変換と新規データセット生成スクリプト
  - 基本形式と四重推論形式の両方をサポート

### 8. QLoRA訓練スクリプト
- `so8t-mmllm/scripts/training/train_so8t_thinking_qlora.py`: Thinking形式データでのQLoRA訓練スクリプト

### 9. 推論API実装
- `scripts/api/serve_think_api.py`: `/think`エンドポイントの完全置き換え版（SO8TThinkingModelを使用）

### 10. 評価スクリプト
- `scripts/evaluation/evaluate_thinking_model.py`: Thinking品質、Safety精度、Verifier相関評価スクリプト

### 11. 設定ファイル
- `configs/so8t_thinking_config.yaml`: モデル、訓練、推論、API、評価の設定

## 機能詳細

### Thinking形式データフォーマット

#### 基本形式
```json
{
  "instruction": "数学の問題を解いて",
  "input": "x + 3 = 7",
  "output": "<think>等式の両辺から3を引く: x + 3 - 3 = 7 - 3 → x = 4</think><final>答えは4です。</final>",
  "safety_label": "ALLOW",
  "verifier_label": {
    "logical": 1.0,
    "faithful": 1.0
  }
}
```

#### 四重推論形式
```json
{
  "instruction": "次の英文仕様書を日本語に翻訳し、要点をまとめてください。",
  "input": "The missile defense system tracks objects using radar and ... (公開資料)",
  "output": "<think-task>We must translate technical defense system description. Maintain accuracy, but this is public, high-level info, no operational instructions.</think-task><think-safety>This content is high-level and public. No explicit harmful instructions. Safe to answer.</think-safety><think-policy>Domain: defense. Provide only descriptive, non-operational translation and summary.</think-policy><final>このシステムはレーダーを用いて飛翔体を追尾し、飛行軌道を解析することで迎撃の可否を判断する仕組みを説明しています。技術的な原理の概要にとどまり、具体的な運用手順や機密情報には触れていません。</final>",
  "safety_label": "ALLOW",
  "policy_domain": "defense_public",
  "verifier_label": {
    "logical": 1.0,
    "faithful": 1.0
  }
}
```

### SO8TThinkingModelの主要機能

1. **Thinking生成**
   - `generate_thinking()`: Thinking形式でテキストを生成
   - 基本形式: `<think>...</think><final>...</final>`
   - 四重推論形式: `<think-task>...</think-task><think-safety>...</think-safety><think-policy>...</think-policy><final>...</final>`

2. **Safety/Verifier評価**
   - `evaluate_safety_and_verifier()`: ThinkingとFinalに対してSafety/Verifier評価を実行
   - Safety判定: ALLOW/ESCALATE/REFUSE
   - Verifierスコア: plausibility, self_confidence

3. **完全フロー**
   - `generate_with_safety_gate()`: Thinking生成 → Safety/Verifier評価 → Final抽出の完全フロー
   - 四重推論対応: Task/Safety/Policy/Finalの4段階推論をサポート

### /thinkエンドポイントのフロー

1. **Thinking生成**
   - ユーザークエリからThinking生成用プロンプトを構築
   - `SO8TThinkingModel.generate_with_safety_gate()`でThinking形式を生成

2. **Safety/Verifier評価**
   - ThinkingとFinalに対してSafety/Verifier評価を実行
   - Safety判定: ALLOW/ESCALATE/REFUSE
   - Verifierスコア: plausibility, self_confidence

3. **安全ゲート**
   - REFUSE: 拒否メッセージを返す
   - ESCALATE: エスカレーションメッセージを返す
   - ALLOW: Finalのみをユーザーに返す（Thinkingは非公開）

4. **監査ログ**
   - Thinkingハッシュ、Safety判定、Verifierスコアを記録
   - SQL監査ログシステムと統合

## 設計判断

### 特殊トークンの選択
- `<think>`と`<final>`を使用（ユーザーが提示した`<think>`形式もサポート）
- トークナイザーに特殊トークンを追加し、モデルが学習可能に

### モデルアーキテクチャ
- `SafetyAwareSO8TModel`を継承し、既存のSafety/Verifierヘッドを活用
- SO(8)回転ゲートとPET正則化を継承

### 訓練方式
- QLoRA（8bit）で効率的に訓練
- 段階的アプローチ: QLoRAで検証 → 必要に応じてフル調整

### 安全ゲート
- ThinkingとFinalの両方に対してSafety評価を実行
- Fail-Closed設計: 信頼度が低い場合はESCALATEに倒す

## 使用例

### 安全なデータ収集

```bash
# 公開ソースから安全にクロール
python scripts/data/crawl_safe_sources.py \
  --output-dir data/crawled \
  --max-pages 100 \
  --delay 1.0
```

### NSFW/危険コンテンツのラベリング

```bash
# 分類器を訓練
python scripts/data/label_nsfw_content.py \
  --mode train \
  --labeled-data data/manual_labels.jsonl \
  --model-path models/nsfw_classifier.joblib

# 大規模データセットを自動ラベリング
python scripts/data/label_nsfw_content.py \
  --mode label \
  --input-data data/crawled/all_crawled.jsonl \
  --output-data data/crawled/labeled.jsonl \
  --model-path models/nsfw_classifier.joblib \
  --confidence-threshold 0.9
```

### 安全なデータセット分割

```bash
# 情報リーク防止のデータ分割
python scripts/data/split_dataset_safe.py \
  --input data/crawled/labeled.jsonl \
  --train data/splits/train.jsonl \
  --val data/splits/val.jsonl \
  --test data/splits/test.jsonl \
  --stratify-by safety_label
```

### データセット作成

```bash
# 既存データセットをThinking形式に変換（基本形式）
python scripts/data/create_thinking_dataset.py \
  --mode convert \
  --input data/japanese_complex_dataset_enhanced.jsonl \
  --output data/thinking_dataset.jsonl

# 四重推論形式に変換
python scripts/data/create_thinking_dataset.py \
  --mode convert \
  --input data/japanese_complex_dataset_enhanced.jsonl \
  --output data/thinking_quadruple_dataset.jsonl \
  --use-quadruple
```

### 訓練

```bash
python so8t-mmllm/scripts/training/train_so8t_thinking_qlora.py \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --dataset data/thinking_dataset.jsonl \
  --output-dir models/so8t_thinking \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 5e-5 \
  --num-epochs 3
```

### 推論API起動

```bash
# 環境変数を設定
export SO8T_BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
export SO8T_MODEL_PATH="models/so8t_thinking"
export SO8T_API_PORT=8000

# APIサーバーを起動
python scripts/api/serve_think_api.py
```

### 評価

```bash
python scripts/evaluation/evaluate_thinking_model.py \
  --model-path models/so8t_thinking \
  --base-model microsoft/Phi-3-mini-4k-instruct \
  --test-dataset data/thinking_test.jsonl \
  --output-dir evaluation_results
```

## ベストプラクティス

1. **データセット準備**
   - 既存データセットをThinking形式に変換
   - Safety/Verifierラベルを適切に付与
   - 多様なドメインと難易度のサンプルを含める

2. **訓練**
   - QLoRAで効率的に訓練
   - チェックポイントを定期的に保存
   - 損失曲線を監視し、過学習を防ぐ

3. **推論**
   - Safety Gateを常に有効化
   - Thinkingは内部に閉じ込め、Finalのみを返す
   - 監査ログを適切に記録

4. **評価**
   - Thinking品質、Safety精度、Verifier相関を定期的に評価
   - 混同行列を可視化し、誤分類パターンを分析

## 今後の拡張

1. **ReActスタイルの統合**
   - `<think>`内にAction/Observationを入れる
   - ツール使用と推論の統合

2. **RLHF/DPOの統合**
   - Verifierスコアを報酬として使用
   - 人間フィードバックによる改善

3. **監査ログの拡張**
   - LAGゲートウェイとの統合
   - より詳細な監査ログスキーマ

4. **多言語対応**
   - 日本語以外の言語でのThinking形式サポート
   - 言語固有の特殊トークン対応

## セキュリティ・法規制リスク管理

### やってはいけないこと（この設計でも避ける）
- 兵器製造、攻撃手順、サイバー攻撃手法などの「具体的な実行手順」を学習・生成させること
- 実名個人情報や秘匿情報、アクセス制限付きサイトのクロール・学習
- RedditやなんJなどから、個人情報・差別表現・違法コンテンツをフィルタせず生取り込み

### やってよい・やるべき方向
- 公開情報に基づく制度・政策・組織構造・技術概念の理解（軍事/航空宇宙/官庁/インフラ等のオープンソース・オープンデータ）
- 医療・創薬も査読論文、ガイドライン、添付文書など正規ソースからのみ
- NSFW領域は「コンテンツ分類・安全フィルタ用」に留める（性的暴力などの検出・拒否）

## 注意事項

- Thinkingは常に内部に閉じ込め、ユーザーに返さない
- Safety Gateは常に有効化し、Fail-Closed設計を維持
- 監査ログは適切に記録し、コンプライアンス要件を満たす
- モデルの推論結果は常に検証し、品質を保証する
- データ収集時は必ずrobots.txtを確認し、利用規約を遵守
- 危険コンテンツは統計的フィルタリングで除外し、手動確認も実施
- データ分割時は情報リークを防止（同一ソースからtrain/test両方に入らない）

