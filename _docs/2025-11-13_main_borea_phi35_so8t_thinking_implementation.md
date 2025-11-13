# Borea-Phi-3.5-mini-Instruct-Jp SO8T/thinking化実装ログ

## 実装情報
- **日付**: 2025-11-13
- **Worktree**: main
- **機能名**: Borea-Phi-3.5 SO8T/thinking化
- **実装者**: AI Agent

## 実装内容

### 1. /think形式SFTデータセット作成スクリプト

**ファイル**: `scripts/data/create_thinking_sft_dataset.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Phi-3.5チャットテンプレート準拠の/think形式データセット作成

- `format_phi35_chat_template()`関数を実装
  - `<|system|>`, `<|user|>`, `<|assistant|>`形式に対応
  - `# 思考ステップ`と`# 最終回答`を分離した構造
- `create_thinking_output()`関数を実装
  - 思考ステップと最終回答を結合
- `convert_sample_to_thinking_format()`関数を実装
  - 既存サンプルを/think形式に変換
  - 既に/think形式の場合はそのまま使用
- `convert_dataset_to_thinking_format()`関数を実装
  - JSONLファイルを一括変換
- `merge_multiple_datasets()`関数を実装
  - 複数データセットをマージして変換

### 2. 選択的SO8Tレイヤー適用機能

**ファイル**: `models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 全レイヤーにSO8Tを適用せず、選択的レイヤー適用に対応

- `SO8TPhi3Model.__init__()`に`so8t_layer_indices`パラメータを追加
  - `None`の場合は全レイヤーに適用
  - リスト指定の場合は指定レイヤーのみに適用
- 選択的レイヤー適用ロジックを実装
  - SO8T適用レイヤー: `SO8TPhi3DecoderLayer`を使用
  - 標準レイヤー: `Phi3DecoderLayer`を使用
- `SO8TPhi3ForCausalLM.__init__()`に`so8t_layer_indices`パラメータを追加
  - `SO8TPhi3Model`に選択的レイヤー適用を伝播

### 3. PET正則化統合学習スクリプト

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: SO8T + PET + /think形式データセットで学習

- `ThinkingSFTDataset`クラスを実装
  - /think形式データセットの読み込み
  - Phi-3.5チャットテンプレート形式に対応
- `SO8TPETTrainer`クラスを実装
  - `Trainer`を継承
  - PET正則化損失を統合
  - `output_hidden_states=True`でhidden_statesを取得
- `load_model_with_so8t()`関数を実装
  - 選択的SO8T統合モデルの読み込み
  - 標準モデルからSO8Tモデルへの重みコピー（詳細実装）
    - `embed_tokens`と`lm_head`の重みをコピー
    - 各レイヤーのアテンション重み（`q_proj`, `k_proj`, `v_proj`, `o_proj`）をコピー
    - MLP重み（`gate_proj`, `up_proj`, `down_proj`）をコピー
    - RMSNorm重み（`input_layernorm`, `post_attention_layernorm`, `norm`）をコピー
    - 形状チェックを実装して、不一致の場合は警告を出力
  - QLoRA 8bit設定に対応
- `main()`関数を実装
  - 設定ファイル読み込み
  - データセット準備
  - PET正則化設定（3相スケジュール）
  - 学習実行

### 4. 焼き込み処理統合

**ファイル**: `scripts/training/bake_borea_phi35_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: SO8T回転ゲートをo_projに焼き込み、標準Phi3アーキテクチャに戻す

- `get_rotation_matrices_from_layer()`関数を実装
  - `SO8TPhi3DecoderLayer`から回転行列を取得
  - `so8t_rotation_gate`から回転行列を計算
  - 複数のインポートパスに対応（`so8t_mmllm.src.so8t_layer`、`so8t_layer`）
- `bake_rotation_into_linear()`関数を実装
  - 右掛け焼き込み: `W' = W @ R`
  - ブロック対角回転行列を構築
- `bake_so8t_model()`関数を実装
  - 学習済みSO8Tモデルを読み込み
  - 各レイヤーのSO8T回転ゲートを`o_proj`に焼き込み
  - 検証: 焼き込み前後で出力の一致を確認
  - Hugging Face形式で保存

### 5. GGUF変換スクリプト

**ファイル**: `scripts/conversion/convert_borea_so8t_to_gguf.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 焼き込み済みモデルをGGUF形式に変換

- `convert_to_gguf()`関数を実装
  - `llama.cpp`の`convert-hf-to-gguf.py`を使用
  - 複数量子化形式を生成（F16, Q8_0, Q4_K_M）
  - 出力先: `D:/webdataset/gguf_models/borea_phi35_so8t_thinking/`

### 6. 設定ファイル作成

**ファイル**: `configs/train_borea_phi35_so8t_thinking.yaml`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 学習設定をYAML形式で定義

- モデル設定: ベースモデルパス、SO8T設定
- SO8T設定: 選択的レイヤー適用、初期化スケール、直交性正則化
- PET設定: 3相スケジュール、λ値
- QLoRA設定: r, alpha, target_modules
- データ設定: /think形式データセットパス
- 学習設定: エポック数、バッチサイズ、学習率
- パイプライン設定: 入力データセット、出力ディレクトリ

### 7. 統合パイプラインスクリプト

**ファイル**: `scripts/pipelines/borea_phi35_so8t_thinking_pipeline.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: 全ステップを統合したパイプライン

- `BoreaSO8TThinkingPipeline`クラスを実装
  - Step 1: /thinkデータセット作成
  - Step 2: SO8T統合学習
  - Step 3: 焼き込み処理
  - Step 4: GGUF変換
- 各ステップのチェックポイント管理
- エラーハンドリングとログ記録
- メタデータ保存（`pipeline_metadata.json`）

## 作成・変更ファイル

1. `scripts/data/create_thinking_sft_dataset.py` - 新規作成
2. `models/Borea-Phi-3.5-mini-Instruct-Jp/modeling_phi3_so8t.py` - 選択的レイヤー適用機能を追加
3. `scripts/training/train_borea_phi35_so8t_thinking.py` - 新規作成
4. `scripts/training/bake_borea_phi35_so8t.py` - 新規作成
5. `scripts/conversion/convert_borea_so8t_to_gguf.py` - 新規作成
6. `configs/train_borea_phi35_so8t_thinking.yaml` - 新規作成
7. `scripts/pipelines/borea_phi35_so8t_thinking_pipeline.py` - 新規作成

## 設計判断

### SO8T統合位置
- Self-Attentionの直後（`attn_outputs`にSO8T回転を適用）
- 選択的レイヤー適用: 設定ファイルで指定されたレイヤーのみ

### PET正則化
- 3相スケジュール: warmup（λ=0.01）→ main（λ=0.05）→ anneal（λ=0.1）
- 二階差分ペナルティ: `Δ²x[t] = x[t+2] - 2*x[t+1] + x[t]`
- `output_hidden_states=True`でhidden_statesを取得してPET損失を計算

### /think形式
- Phi-3.5チャットテンプレートに準拠
- 思考ステップと最終回答を分離
- システムメッセージで思考プロセスを指示

### 焼き込み処理
- 右掛け: `W' = W @ R`（`o_proj`の重みに回転行列を右から掛ける）
- SO8Tブロック削除後、標準Phi3アーキテクチャとして保存
- 検証: 焼き込み前後で出力の一致を確認

## 実装の詳細改善

### モデル読み込みの改善
- `load_model_with_so8t()`関数で標準モデルからSO8Tモデルへの重みコピーを実装
  - `embed_tokens`と`lm_head`の重みをコピー
  - 各レイヤーのアテンション重み（`q_proj`, `k_proj`, `v_proj`, `o_proj`）をコピー
  - MLP重み（`gate_proj`, `up_proj`, `down_proj`）をコピー
  - RMSNorm重み（`input_layernorm`, `post_attention_layernorm`）をコピー
  - 形状チェックを実装して、不一致の場合は警告を出力
- `SO8TPhi3ForCausalLM.__init__()`に`so8t_layer_indices`パラメータを追加
  - `SO8TPhi3Model`に選択的レイヤー適用を伝播

### インポートエラーの修正
- `bake_borea_phi35_so8t.py`でSO8T回転ゲートのインポートパスを修正
  - `so8t_mmllm.src.so8t_layer`からのインポートを試行
  - フォールバックとして`so8t_layer`からのインポートを試行
- `train_borea_phi35_so8t_thinking.py`でloggerの定義順序を修正
  - ロギング設定を早期に実行して、インポートエラー時にloggerが使用可能になるように修正

### コード品質の改善
- 未使用のインポートを削除（警告レベル）
- リンターエラーの修正（logger未定義エラーを解決）

## テスト結果

- リンターエラー: 警告のみ（実行に影響なし）
  - 未使用インポートの警告（`os`, `time`, `signal`, `Dict`, `datetime`, `nn`, `DataLoader`, `TrainerCallback`, `tqdm`）
  - モジュールレベルのインポート順序の警告（実行には影響なし）
- 実装完了: 全7ステップ完了
- 実装日時: 2025-11-13 17:33:09

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
- 思考ステップ（`# 思考ステップ`）は外部非公開を徹底
- `# 最終回答`のみ返す実装を維持
- 監査ログでThinkingハッシュを記録（内容は非公開）

## 次のステップ

1. /think形式データセットの作成と検証
2. SO8T統合学習の実行と検証
3. 焼き込み処理の実行と検証
4. GGUF変換の実行と検証
5. Ollamaでの動作確認


