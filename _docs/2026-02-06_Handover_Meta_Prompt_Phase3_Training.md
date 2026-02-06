# 引き継ぎメタプロンプト: Enhanced Moonshot Pipeline Phase 3 実行中

## 1. コンテキスト & 直近の状況

現在、**Enhanced Moonshot Pipeline の Phase 3 (本格的再学習)** がバックグラウンドで実行中です。
A/B/C ベンチマークの準備（Model A/B の GGUF変換）は完了しており、Model C の再学習完了を待っている状態です。

### 🕒 現在時刻

2026-02-06 09:37 (JST)

---

## 2. 実行中のプロセス (最重要)

**Unsloth SFT/GRPO トレーニング** が進行中です。

- **コマンド**:
  ```powershell
  $env:PYTHONPATH = "c:\Users\downl\Desktop\SO8T"
  $env:SO8T_USE_UNSLOTH = "1"
  $env:SO8T_DRYRUN = "0"
  $env:SO8T_GRAPE_VARIANT = "multiplicative"
  $env:SO8T_CHECKPOINT_INTERVAL = "300"
  $env:SO8T_CHECKPOINT_ROLLING = "3"
  py -3 src/training/train_unsloth_so8t.py --phase full --config src/infrastructure/config/borea_training.json
  ```
- **補足**: `src.utils` のインポートエラーを解消するため、`$env:PYTHONPATH` を設定して実行しています。
- **ログ確認**: `logs/aegis_v3_pipeline.log` または標準出力を確認してください。

---

## 3. 完了済みのタスク

### A/B/C ベンチマーク準備

- ✅ **Model A (Phi-3.5)**: ダウンロード完了 & GGUF(BF16)変換完了 (`H:\from_D\SO8T_models\gguf\model_a.bf16.gguf`)
- ✅ **Model B (Borea)**: ダウンロード完了 & GGUF(BF16)変換完了 (`H:\from_D\SO8T_models\gguf\model_b.bf16.gguf`)

### パイプライン

- ✅ **Phase 1**: 環境整備
- ✅ **Phase 2**: データ収集 (`osint_source_collector.py` 及び `hf_cli_dataset_fetch.py` 完了)

---

## 4. 次のアクション（引き継ぎ後）

1. **Phase 3 の監視**:
   - トレーニングが正常に完了するまでログを監視してください。
   - エラーが発生した場合は、`logs/` ディレクトリを確認し対処してください。

2. **Phase 4: Sakana AI 統合エージェント実行**:

   ```powershell
   py -3 src/agents/sakana_ai_integrated_agent.py
   ```

3. **Phase 5: 統計ベンチマーク (A/B/C テスト)**:
   - 再学習された Model C を使用して実施します。
   - `task.md` の手順に従い、HFウェイトを用いた統計的比較を行います。

   ```powershell
   py -3 src/evaluation/phase6_statistical_benchmark.py
   ```

4. **Phase 6: HF アップロード**:
   ```powershell
   py -m huggingface_hub.commands.huggingface_cli upload ...
   ```

## 5. 関連ファイル

- `C:\Users\downl\.gemini\antigravity\brain\5a1d5371-5593-4741-b1ea-fdf1927047de\task.md`: タスク管理
- `src/training/train_unsloth_so8t.py`: 現在実行中のスクリプト
