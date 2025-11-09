# Phi-4 SO8T 日本語ファインチューニング実行ガイド

## 実装完了状況

### ✅ 完了済み項目

1. **環境セットアップ** ✓
   - PyTorch 2.5.1+cu121（CUDA 12.1）
   - RTX 3060 12GB認識
   - 全依存ライブラリインストール済み

2. **SO8Tコア実装** ✓
   - `so8t_core/so8t_layer.py`: SO8T回転ゲート
   - `so8t_core/burn_in.py`: 焼き込み機構
   - `so8t_core/pet_regularizer.py`: PET正規化（既存）

3. **Phi-4統合** ✓
   - `phi4_so8t_integrated/`: 設定ファイル準備完了
   - SO8Tパラメータ追加済み（32層）

4. **データセット構築** ✓
   - `data/phi4_japanese_synthetic.jsonl`: 4,999サンプル
   - ドメイン: defense, aerospace, transport, general
   - 三重推論データ: ALLOW/ESCALATION/DENY

5. **学習スクリプト** ✓
   - `scripts/train_phi4_so8t_japanese.py`
   - QLoRA 8bit対応
   - PET正規化統合
   - 電源断リカバリー対応
   - 3分間隔チェックポイント（5個ストック）

6. **エージェントシステム** ✓
   - `so8t_core/triple_reasoning_agent.py`: 三重推論
   - `utils/audit_logger.py`: SQL監査ログ
   - `so8t_core/secure_rag.py`: 閉域RAG

7. **焼き込み+GGUF変換スクリプト** ✓
   - `scripts/burn_in_and_convert_gguf.py`

8. **Ollama統合スクリプト** ✓
   - `scripts/ollama_integration_test.bat`

9. **温度較正スクリプト** ✓
   - `scripts/temperature_calibration.py`

10. **包括的評価スクリプト** ✓
    - `scripts/evaluate_comprehensive.py`

11. **最終レポート生成** ✓
    - `scripts/generate_final_report.py`

### ⏳ 実行待ち項目

**学習実行（3-5時間）**
- スクリプトは完成
- データセット準備完了
- ディスク容量確保が必要（約10GB）

---

## 実行手順

### Phase 1: ディスク容量確保（必須）

学習実行には約10GBの空き容量が必要です：

```powershell
# 不要ファイル削除
# 例：古いチェックポイント、一時ファイル等
```

### Phase 2: 学習実行

```bash
# 学習開始（RTX 3060で約3-5時間）
py -3 scripts/train_phi4_so8t_japanese.py \
  --model_path Phi-4-mini-instruct \
  --data_path data/phi4_japanese_synthetic.jsonl \
  --output_dir checkpoints/phi4_so8t_japanese_final \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_epochs 3 \
  --lora_r 64 \
  --lora_alpha 128
```

**電源断リカバリー対応**:
- Ctrl+Cで中断しても自動保存
- `--resume_from_checkpoint checkpoints/phi4_so8t_japanese_checkpoints/checkpoint_XXX_YYYYMMDD_HHMMSS` で再開可能

### Phase 3: 焼き込み + GGUF変換

```bash
# 焼き込み + F16 GGUF変換
py -3 scripts/burn_in_and_convert_gguf.py \
  --model_path checkpoints/phi4_so8t_japanese_final \
  --burn_in_output models/phi4_so8t_baked \
  --gguf_output models/phi4_so8t_baked \
  --quantization F16

# （オプション）Q4_K_M量子化
py -3 scripts/burn_in_and_convert_gguf.py \
  --model_path models/phi4_so8t_baked \
  --gguf_output models/phi4_so8t_baked \
  --quantization Q4_K_M \
  --skip_burn_in
```

### Phase 4: Ollama統合 + テスト

```bash
# Ollama統合テスト実行
scripts\ollama_integration_test.bat
```

テスト項目：
1. 基本的な数学問題
2. SO(8)群の説明
3. 日本語複雑推論
4. 三重推論テスト（ALLOW）
5. 三重推論テスト（ESCALATION）
6. 三重推論テスト（DENY）

### Phase 5: 温度較正

```bash
# 温度較正実行（ECE最小化）
py -3 scripts/temperature_calibration.py \
  --model_path checkpoints/phi4_so8t_japanese_final \
  --data_path data/phi4_japanese_synthetic.jsonl \
  --output_path _docs/temperature_calibration_report.json \
  --temperature_min 0.5 \
  --temperature_max 2.0 \
  --n_steps 16 \
  --max_samples 100
```

### Phase 6: 包括的評価

```bash
# 包括的評価実行
py -3 scripts/evaluate_comprehensive.py \
  --model_path checkpoints/phi4_so8t_japanese_final \
  --data_path data/phi4_japanese_synthetic.jsonl \
  --output_path _docs/comprehensive_evaluation_report.json \
  --max_samples 100
```

評価項目：
- 精度（Accuracy）
- 較正（ECE, Brier Score）
- 推論速度（Tokens/sec）
- 安定性（長文発振チェック）
- 三重推論精度（ALLOW/ESCALATION/DENY）

### Phase 7: 最終レポート生成

```bash
# 最終レポート生成
py -3 scripts/generate_final_report.py \
  --output _docs/2025-11-06_phi4_so8t_final_report.md
```

---

## トラブルシューティング

### ディスク容量不足

```powershell
# Pipキャッシュクリア
py -3 -m pip cache purge

# 古いチェックポイント削除
Remove-Item -Recurse -Force checkpoints/old_*

# Hugging Faceキャッシュクリア
Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface
```

### CUDA OOMエラー

学習スクリプトのパラメータ調整：
- `--batch_size 1`（デフォルト2から削減）
- `--gradient_accumulation_steps 16`（デフォルト8から増加）
- `--max_length 256`（デフォルト512から削減）

### モデルロードエラー

```bash
# 軽量版統合を再実行
py -3 scripts/integrate_phi4_so8t_lightweight.py \
  --model_path Phi-4-mini-instruct \
  --output_path phi4_so8t_integrated
```

---

## 実装ファイル一覧

### コアライブラリ（`so8t_core/`）
- `so8t_layer.py`: SO8T回転ゲート層
- `burn_in.py`: 焼き込み機構
- `pet_regularizer.py`: PET正規化
- `triple_reasoning_agent.py`: 三重推論エージェント
- `secure_rag.py`: 閉域RAGシステム

### ユーティリティ（`utils/`）
- `audit_logger.py`: SQL監査ログ

### スクリプト（`scripts/`）
- `integrate_phi4_so8t_lightweight.py`: SO8T統合（軽量版）
- `collect_public_datasets.py`: 公開データ収集
- `generate_synthetic_japanese.py`: 合成データ生成
- `train_phi4_so8t_japanese.py`: QLoRA学習
- `burn_in_and_convert_gguf.py`: 焼き込み+GGUF変換
- `ollama_integration_test.bat`: Ollama統合テスト
- `temperature_calibration.py`: 温度較正
- `evaluate_comprehensive.py`: 包括的評価
- `generate_final_report.py`: 最終レポート生成

### データ（`data/`）
- `phi4_japanese_synthetic.jsonl`: 合成訓練データ（4,999サンプル）

### モデル保存先
- `phi4_so8t_integrated/`: SO8T統合設定
- `checkpoints/phi4_so8t_japanese_final/`: 学習済みモデル（実行後）
- `checkpoints/phi4_so8t_japanese_checkpoints/`: チェックポイント（実行中）
- `models/phi4_so8t_baked/`: 焼き込み済みモデル+GGUF（実行後）

---

## 推定所要時間

- **学習**: 3-5時間（RTX 3060, 5,000サンプル, 3エポック）
- **焼き込み**: 15-30分
- **GGUF変換**: 10-20分
- **温度較正**: 30-60分
- **包括的評価**: 30-60分

**合計**: 約5-8時間

---

## 次のステップ

1. ✅ 全スクリプト実装完了
2. ⏳ **ディスク容量確保**（約10GB）
3. ⏳ **学習実行**（Phase 2開始）
4. ⏳ 焼き込み→GGUF変換→Ollama→評価→レポート

---

**実装完了日時**: 2025-11-06 20:49  
**ステータス**: スクリプト実装完了、学習実行待ち  
**音声通知**: 完了済み

























