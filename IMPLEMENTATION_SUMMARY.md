# SO8T Pipeline Implementation Summary

## Files Created

### Core Models
- `src/core/models/so8t_moe_router.py` - SO8群トライアリティルータ + MoEレイヤ

### Evolution Modules
- `src/training/evolution/ebbinghaus_forgetting.py` - Ebbinghaus忘却曲線
- `src/training/evolution/shinka_evolve.py` - ShinkaEvolve最適化
- `src/training/evolution/__init__.py` - モジュール初期化

### Regularization
- `src/training/regularization/pet_regularizer.py` - PET正則化
- `src/training/regularization/__init__.py` - モジュール初期化

### Quantization
- `src/core/quantization/imatrix.py` - imatrix量子化
- `src/core/quantization/__init__.py` - モジュール初期化

### Pipeline & Utils
- `src/training/so8t_moe_pipeline.py` - 基本訓練パイプライン
- `src/training/so8t_moe_unsloth_pipeline.py` - Unsloth BF16拡張パイプライン
- `src/utils/checkpoint_manager.py` - ローリングチェックポイント(5分,3スロット)
- `src/utils/progress_tracker.py` - tqdm+logging進捗表示

### Tests
- `tests/test_so8t_components.py` - コンポーネントテスト
- `tests/test_so8t_pipeline_integration.py` - 統合テスト

## Quick Start

```powershell
# ドライラン
py -3 tests/test_so8t_pipeline_integration.py

# 訓練実行
py -3 src/training/so8t_moe_unsloth_pipeline.py \
  --model-name "microsoft/Phi-3.5-mini-instruct" \
  --output-dir "D:\webdataset\models\so8t_moe_final" \
  --batch-size 4 \
  --epochs 3 \
  --bf16
```

## 環境変数

| 変数 | デフォルト | 説明 |
|------|----------|------|
| SO8T_BASE_MODEL | microsoft/Phi-3.5-mini-instruct | ベースモデル |
| SO8T_OUTPUT_DIR | D:\webdataset\models\so8t_moe_final | 出力ディレクトリ |
| SO8T_CHECKPOINT_DIR | D:\webdataset\checkpoints\training | チェックポイント |
| SO8T_CHECKPOINT_INTERVAL | 300 | チェックポイント間隔(秒) |
| SO8T_CHECKPOINT_ROLLING | 3 | ローリングチェックポイント数 |

## コンポーネント設定

### SO8MoELayer
- num_experts: 4
- top_k: 2
- hidden_dim: 3072

### EbbinghausForgettingCurve
- decay_rate: 0.1
- reinforcement_rate: 0.1
- retention_threshold: 0.3

### ShinkaEvolveOptimizer
- evolution_interval: 100
- mutation_scale: 0.01
- retention_threshold: 0.3
