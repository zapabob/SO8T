# Borea-Phi-3.5 SO8T/thinking 実装ログ

## 実装情報
- **日付**: 2025-11-13
- **Worktree**: main
- **機能名**: Borea-Phi-3.5 SO8T/thinking化
- **実装者**: AI Agent

## 実装内容

### 1. /think形式データセットの作成

**ファイル**: `scripts/data/create_thinking_sft_dataset.py`

**実装状況**: [実装済み]  
**動作確認**: [OK]  
**確認日時**: 2025-11-13  
**備考**: 既存の4値分類データセットから/think形式のSFTデータセットを生成

- 入力データセット: `D:/webdataset/processed/four_class/four_class_*.jsonl`
- 出力データセット: `D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl`
- 生成サンプル数: 1,441サンプル
- フォーマット: Phi-3.5チャットテンプレート形式（`<|system|>`, `<|user|>`, `<|assistant|>`）
- 思考ステップと最終回答の分離: `# 思考ステップ` と `# 最終回答` を含む形式

**実行コマンド**:
```bash
$files = Get-ChildItem "D:\webdataset\processed\four_class\four_class_*.jsonl" | Select-Object -ExpandProperty FullName
py -3 scripts/data/create_thinking_sft_dataset.py --inputs $files --output "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl"
```

### 2. SO8T統合学習の実行

**ファイル**: `scripts/training/train_borea_phi35_so8t_thinking.py`

**実装状況**: [実装済み]  
**動作確認**: [進行中]  
**確認日時**: 2025-11-13  
**備考**: 選択的SO8T統合 + PET正則化 + /think形式データセットで学習中

- 設定ファイル: `configs/train_borea_phi35_so8t_thinking.yaml`
- データセット: `D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl`
- 出力ディレクトリ: `D:/webdataset/checkpoints/training/borea_phi35_so8t_thinking`
- RTX3060対応設定:
  - `per_device_train_batch_size: 1`
  - `gradient_accumulation_steps: 16`
  - `fp16: true`
  - `gradient_checkpointing: true`
- SO8T設定:
  - `enabled: true`
  - `layer_indices: null` (全レイヤーに適用)
  - `init_scale: 0.05`
  - `orthogonal_reg: 1.0e-4`
- PET正則化設定:
  - `enabled: true`
  - 3相スケジュール（探索相、遷移相、安定相）
  - `lambda_exploration: 0.01`
  - `lambda_transition: 0.05`
  - `lambda_stabilization: 0.1`

**実行コマンド**:
```bash
py -3 scripts/training/train_borea_phi35_so8t_thinking.py \
    --config configs/train_borea_phi35_so8t_thinking.yaml \
    --dataset "D:\webdataset\processed\thinking_sft\thinking_sft_dataset.jsonl" \
    --output-dir "D:\webdataset\checkpoints\training\borea_phi35_so8t_thinking"
```

**学習状況**: モデル読み込み中（2025-11-13 18:43時点）

**修正内容**:
- HuggingFaceキャッシュをDドライブに設定（`D:\webdataset\hf_cache`）
- ディスク容量不足エラーを解決（CドライブではなくDドライブを使用）
- 環境変数設定: `HF_HOME`, `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`

### 3. 焼き込み処理の実行

**ファイル**: `scripts/training/bake_borea_phi35_so8t.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Step 2完了後に実行予定

- 学習済みSO8Tモデルの回転ゲートを`o_proj`に焼き込み（右掛け: `W' = W @ R`）
- 標準Phi3アーキテクチャに戻す
- 入力: `D:/webdataset/checkpoints/training/borea_phi35_so8t_thinking/final_model`
- 出力: `D:/webdataset/borea_phi35_so8t_thinking/baked_model`

**実行コマンド（予定）**:
```bash
py -3 scripts/training/bake_borea_phi35_so8t.py \
    --model-path D:/webdataset/checkpoints/training/borea_phi35_so8t_thinking/final_model \
    --output-path D:/webdataset/borea_phi35_so8t_thinking/baked_model
```

### 4. GGUF変換の実行

**ファイル**: `scripts/conversion/convert_borea_so8t_to_gguf.py`

**実装状況**: [実装済み]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Step 3完了後に実行予定

- 焼き込み済みモデルをGGUF形式に変換
- 複数量子化形式を生成（F16, Q8_0, Q4_K_M）
- 入力: `D:/webdataset/borea_phi35_so8t_thinking/baked_model`
- 出力: `D:/webdataset/gguf_models/borea_phi35_so8t_thinking/`

**実行コマンド（予定）**:
```bash
py -3 scripts/conversion/convert_borea_so8t_to_gguf.py \
    --model-path D:/webdataset/borea_phi35_so8t_thinking/baked_model \
    --output-dir D:/webdataset/gguf_models/borea_phi35_so8t_thinking \
    --model-name borea_phi35_so8t_thinking \
    --quantization-types f16 q8_0 q4_k_m
```

### 5. Ollamaでの動作確認

**実装状況**: [未実装]  
**動作確認**: [未確認]  
**確認日時**: -  
**備考**: Step 4完了後に実行予定

- GGUFモデルをOllamaにインポート
- Modelfileの作成（Phi-3.5チャットテンプレート対応）
- /think形式での推論テスト
- 思考ステップと最終回答の分離確認

**実行コマンド（予定）**:
```bash
# Modelfileの作成
ollama create borea-phi35-so8t-thinking -f modelfiles/borea_phi35_so8t_thinking.modelfile

# 推論テスト
ollama run borea-phi35-so8t-thinking "以下の問題を解いてください。まず思考ステップを整理し、その後最終回答を出してください。"
```

## 作成・変更ファイル
- `scripts/data/create_thinking_sft_dataset.py` (既存、使用)
- `scripts/training/train_borea_phi35_so8t_thinking.py` (既存、使用)
- `scripts/training/bake_borea_phi35_so8t.py` (既存、使用)
- `scripts/conversion/convert_borea_so8t_to_gguf.py` (既存、使用)
- `configs/train_borea_phi35_so8t_thinking.yaml` (既存、使用)
- `D:/webdataset/processed/thinking_sft/thinking_sft_dataset.jsonl` (新規作成)

## 設計判断
- **データセット形式**: Phi-3.5チャットテンプレート形式を採用し、思考ステップと最終回答を分離
- **SO8T適用**: 全レイヤーに適用（`layer_indices: null`）
- **PET正則化**: 3相スケジュールで段階的に正則化強度を増加
- **RTX3060対応**: バッチサイズ1、勾配累積16で実効バッチサイズ16を維持
- **量子化**: QLoRA 8bitでメモリ効率化

## テスト結果
- **データセット検証**: 1,441サンプル、Phi-3.5テンプレート形式、思考ステップ/最終回答分離確認済み
- **学習**: 進行中（モデル読み込み完了後、学習開始予定）

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

## 次のステップ
1. Step 2（学習）の完了を待つ
2. Step 3（焼き込み処理）を実行
3. Step 4（GGUF変換）を実行
4. Step 5（Ollama動作確認）を実行
