from __future__ import annotations

import sys
import torch


def test_ebbinghaus_import():
    from src.training.evolution.ebbinghaus_forgetting import EbbinghausForgettingCurve

    curve = EbbinghausForgettingCurve()
    curve.update([1, 2, 3], is_reinforced=[True, False, False])
    stats = curve.get_stats()
    assert "avg_retention" in stats
    print("[OK] EbbinghausForgettingCurve")


def test_shinka_evolve_import():
    from src.training.evolution.shinka_evolve import (
        ShinkaEvolveOptimizer,
        EvolutionConfig,
    )

    config = EvolutionConfig()
    assert config.evolution_interval == 100
    print("[OK] ShinkaEvolveOptimizer")


def test_pet_import():
    from src.training.regularization.pet_regularizer import PETRegularizer, PETConfig

    config = PETConfig()
    assert config.lambda_reg == 0.01
    print("[OK] PETRegularizer")


def test_moe_router_import():
    from src.core.models.so8t_moe_router import SO8MoELayer, SO8TrialityRouter

    router = SO8TrialityRouter(num_experts=4, hidden_dim=768)
    moe = SO8MoELayer(hidden_dim=768, num_experts=4)
    assert router is not None
    assert moe is not None
    print("[OK] SO8MoELayer")


def test_quantizer_import():
    from src.core.quantization.imatrix import IMatrixQuantizer, QuantizationConfig

    config = QuantizationConfig()
    assert config.num_bins == 256
    print("[OK] IMatrixQuantizer")


def test_checkpoint_manager_import():
    from src.utils.checkpoint_manager import RollingCheckpointManager, CheckpointConfig

    config = CheckpointConfig()
    assert config.interval_seconds == 300
    assert config.max_slots == 3
    print("[OK] RollingCheckpointManager")


def test_progress_tracker_import():
    from src.utils.progress_tracker import TrainingProgressTracker, ProgressConfig

    config = ProgressConfig()
    assert config.log_interval == 10
    print("[OK] TrainingProgressTracker")


def test_pipeline_import():
    from src.training.so8t_moe_pipeline import SO8TMoETrainer, SO8TPipelineConfig

    config = SO8TPipelineConfig()
    assert config.num_experts == 4
    print("[OK] SO8TPipelineConfig")


if __name__ == "__main__":
    try:
        test_ebbinghaus_import()
        test_shinka_evolve_import()
        test_pet_import()
        test_moe_router_import()
        test_quantizer_import()
        test_checkpoint_manager_import()
        test_progress_tracker_import()
        test_pipeline_import()
        print("\n[OK] All tests passed")
    except Exception as e:
        print(f"\n[NG] Test failed: {e}")
        sys.exit(1)
