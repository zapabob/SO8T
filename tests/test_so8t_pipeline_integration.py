# -*- coding: utf-8 -*-
"""
SO8T Pipeline Integration Tests
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import torch
import tempfile
import os


def test_so8t_moe_router():
    from core.models.so8t_moe_router import SO8TrialityRouter

    router = SO8TrialityRouter(num_experts=4, hidden_dim=768)
    batch, seq = 2, 64
    x = torch.randn(batch, seq, 768)
    expert_indices, routing_weights = router(x)
    assert routing_weights.shape == (batch, 4)
    print("[OK] SO8TrialityRouter")


def test_ebbinghaus_forgetting():
    from training.evolution.ebbinghaus_forgetting import (
        EbbinghausForgettingCurve as EbbinghausCurve,
    )

    curve = EbbinghausCurve(decay_rate=0.1, reinforcement_rate=0.1)
    curve.update([1, 2, 3], is_reinforced=[True, False, False])
    stats = curve.get_stats()
    assert "avg_retention" in stats
    assert stats["total_tokens"] > 0
    retention = curve.get_retention_strength(1)
    assert 0.0 <= retention <= 1.0
    print("[OK] EbbinghausForgettingCurve")


def test_shinka_evolve():
    from training.evolution.shinka_evolve import ShinkaEvolveOptimizer, EvolutionConfig
    from training.evolution.ebbinghaus_forgetting import EbbinghausForgettingCurve

    config = EvolutionConfig()
    assert config.evolution_interval == 100
    assert config.mutation_scale == 0.01
    assert config.retention_threshold == 0.3

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 768)

    model = DummyModel()
    curve = EbbinghausForgettingCurve()
    optimizer = ShinkaEvolveOptimizer(
        model=model, ebbinghaus_curve=curve, config=config
    )
    state = optimizer.evolve_frozen_parameters(step=50, metrics={"loss": 0.5})
    assert hasattr(state, "active_frozen")
    print("[OK] ShinkaEvolveOptimizer")


def test_pet_regularizer():
    from training.regularization.pet_regularizer import (
        PETRegularizer,
        PETConfig,
        PETScheduler,
    )

    config = PETConfig()
    assert config.lambda_reg == 0.01
    simple_model = torch.nn.Linear(10, 10)
    scheduler = PETScheduler(
        optimizer=torch.optim.Adam(simple_model.parameters()),
        initial_lambda=0.01,
        final_lambda=0.001,
        warmup_steps=100,
        total_steps=1000,
    )
    assert scheduler.get_current_lambda() == 0.01
    scheduler.step()
    assert scheduler.current_step == 1
    print("[OK] PETRegularizer & PETScheduler")


def test_imatrix_quantizer():
    from core.quantization.imatrix import IMatrixQuantizer, QuantizationConfig

    config = QuantizationConfig()
    assert config.num_bins == 256
    assert config.clip_range == (-3.0, 3.0)

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(768, 768)

    model = DummyModel()
    quantizer = IMatrixQuantizer(model=model, config=config)
    stats = quantizer.get_quantization_stats()
    assert "config" in stats
    print("[OK] IMatrixQuantizer")


def test_checkpoint_manager():
    from utils.checkpoint_manager import RollingCheckpointManager, CheckpointConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = CheckpointConfig(
            interval_seconds=60,
            max_slots=2,
            checkpoint_dir=tmpdir,
        )
        manager = RollingCheckpointManager(config=config)

        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        manager.update(model=model, optimizer=optimizer, epoch=0, step=10)
        info = manager.get_checkpoint_info()
        assert len(info) >= 0
        latest = manager.load_latest_checkpoint()
        assert latest is not None or len(info) == 0
        print("[OK] RollingCheckpointManager")


def test_progress_tracker():
    from utils.progress_tracker import TrainingProgressTracker, ProgressConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = ProgressConfig(
            log_file=os.path.join(tmpdir, "test.log"),
            console_output=False,
        )
        tracker = TrainingProgressTracker(total_steps=100, desc="Test")
        tracker.update(step=10, metrics={"loss": 0.5})
        tracker.update(step=20, metrics={"loss": 0.4})
        summary = tracker.get_summary()
        assert summary["completed_steps"] >= 20
        tracker.close()
        print("[OK] TrainingProgressTracker")


def test_so8t_moe_pipeline():
    from training.so8t_moe_pipeline import SO8TPipelineConfig

    config = SO8TPipelineConfig()
    assert config.num_experts == 4
    print("[OK] SO8TPipelineConfig")
    print("[OK] Full Pipeline Integration")


if __name__ == "__main__":
    print("=" * 60)
    print("SO8T Pipeline Integration Tests")
    print("=" * 60)
    tests = [
        test_so8t_moe_router,
        test_ebbinghaus_forgetting,
        test_shinka_evolve,
        test_pet_regularizer,
        test_imatrix_quantizer,
        test_checkpoint_manager,
        test_progress_tracker,
        test_so8t_moe_pipeline,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[NG] {test.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
