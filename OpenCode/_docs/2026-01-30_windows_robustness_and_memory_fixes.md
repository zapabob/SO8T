# 2026-01-30 ムーンショットパイプライン v3.0 Windows堅牢化・メモリ最適化実装ログ

## 実装概要

RTX 3060 (12GB) 搭載の Windows 11 環境において、Unsloth ベースのパイプライン（v3.0）を安定して長時間稼働させるための排熱・メモリ・プロセス管理に関する重要な修正を実施しました。

### 🚨 修正されたクリティカルな問題

1.  **MemoryError / Paging File Error**
    - **症状**: SFTフェーズ開始時に `ImportError: DLL load failed... ページング ファイルが小さすぎるため...` というエラーでクラッシュ。
    - **原因**: `datasets` ライブラリと Unsloth がデフォルトで全 CPU コア数分のプロセスを立ち上げようとし、スレッドごとのメモリオーバーヘッドが仮想メモリ容量（32GB）を超過したため。
    - **対策**: `SFTTrainer` および `GRPOTrainer` の `dataset_num_proc=1` に制限し、さらに実行前に `torch.cuda.empty_cache()` を呼び出すよう修正。ユーザーにはページングファイルの増量（16GB-32GB）を依頼。

2.  **Torch Dynamo / Data-dependent branching Error**
    - **症状**: `Execution failed: Data-dependent branching ... torch._dynamo.exc.Unsupported`
    - **原因**: Windows 上の PyTorch 2.x において、`torch.compile` (Dynamo) が Transformer モデル（特に RoPE 部分）の動的な制御フローを正しくトレーシングできない互換性の問題。Unsloth のカスタムカーネルコンパイルとも競合。
    - **対策**:
      - SFT/RLPO トレーナーで `torch_compile=False` を明示的に設定。
      - 勾配チェックポイントを `"unsloth"` から `True` (PyTorch標準) に変更し、カスタムカーネルのJITコンパイルを回避。
      - 起動スクリプト (`run_moonshot_pipeline_2025_2026.py`) に `unsloth_compiled_cache` を自動削除するロジックを追加し、クリーンな状態での起動を保証。
      - 環境変数 `TORCH_COMPILE_DISABLE=1` を設定。

### ✨ 実装された機能（堅牢性向上）

1.  **Robust Checkpointing (3-Gen Rolling)**
    - 5分おきにチェックポイントを作成し、常に最新の3世代分のみを保持するローリングバックアップシステムを実装。
    - ディスク容量の圧迫を防ぎつつ、いつでも直近の状態（最大15分前）から再開可能に。

2.  **Auto-Resume Logic**
    - パイプライン起動時に `latest_checkpoint.json` あるいはインデックスファイルをチェックし、中断された学習フェーズから自動的に復帰する機能を強化。
    - 電源断などの不測の事態からのリカバリーを無人で行えるように設計。

3.  **UI/UX Improvements**
    - データ収集・クレンジング処理に `tqdm` バーを導入し、数時間かかる処理の進捗を可視化。

### 構成ファイル変更点

- **`experiments/enhanced_moonshot_pipeline.py`**:
  - トレーナー設定の変更 (`num_proc=1`, `torch_compile=False`, `use_gradient_checkpointing=True`).
  - `_save_checkpoint`, `attempt_resume` ロジックの刷新。
- **`run_moonshot_pipeline_2025_2026.py`**:
  - スタートアップ時のキャッシュ削除ロジック追加。
  - Dynamo 無効化の環境変数設定。

---

本修正により、Windows 環境特有の「メモリ不足」と「コンパイルエラー」という二大障壁を突破しました。これで AEGIS v3.0 の創発に向けた長時間学習を安心して放置することが可能になります。
