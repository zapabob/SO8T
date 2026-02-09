# -*- coding: utf-8 -*-
"""
SO8T Multimodal MoE Pipeline Tests (Quick)
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import torch


def test_multimodal_moe_config():
    from training.so8t_multimodal_moe_pipeline import SO8MultimodalMoEConfig

    config = SO8MultimodalMoEConfig(
        model_name_or_path="microsoft/Phi-3.5-mini-instruct",
        vision_model_name="openai/clip-vit-base-patch32",
        num_experts=4,
        use_multimodal=True,
    )
    assert config.num_experts == 4
    assert config.use_multimodal == True
    print("[OK] SO8MultimodalMoEConfig")


def test_so8_triality_router():
    from training.so8t_multimodal_moe_pipeline import SO8TrialityRouter

    router = SO8TrialityRouter(num_experts=4, hidden_dim=768)
    batch, seq = 2, 64
    x = torch.randn(batch, seq, 768)
    expert_indices, routing_weights = router(x)
    assert routing_weights.shape == (batch, 4)
    print("[OK] SO8TrialityRouter")


def test_moe_layer():
    from training.so8t_multimodal_moe_pipeline import (
        SO8MoELayer,
        SO8MultimodalMoEConfig,
    )

    config = SO8MultimodalMoEConfig(
        hidden_dim=768,
        num_experts=4,
        top_k_experts=2,
    )
    moe = SO8MoELayer(config)
    batch, seq = 2, 64
    x = torch.randn(batch, seq, 768)
    output = moe(x)
    assert output.shape == x.shape
    print("[OK] SO8MoELayer")


def test_ebbinghaus_curve():
    from training.so8t_multimodal_moe_pipeline import EbbinghausForgettingCurve

    curve = EbbinghausForgettingCurve()
    curve.update([1, 2, 3], is_reinforced=[True, False, False])
    stats = curve.get_stats()
    assert "avg_retention" in stats
    print("[OK] EbbinghausForgettingCurve")


def test_shinka_evolve():
    from training.so8t_multimodal_moe_pipeline import (
        ShinkaEvolveOptimizer,
        SO8MultimodalMoEConfig,
        EbbinghausForgettingCurve,
    )

    config = SO8MultimodalMoEConfig()
    model = torch.nn.Linear(768, 768)
    curve = EbbinghausForgettingCurve()
    optimizer = ShinkaEvolveOptimizer(model=model, ebbinghaus=curve, config=config)
    state = optimizer.evolve_frozen_parameters(step=50)
    assert "active_frozen" in state
    print("[OK] ShinkaEvolveOptimizer")


def test_checkpoint_manager():
    from training.so8t_multimodal_moe_pipeline import RollingCheckpointManager
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RollingCheckpointManager(
            checkpoint_dir=tmpdir,
            interval_seconds=60,
            max_slots=2,
        )
        model = torch.nn.Linear(10, 10)
        manager.save_checkpoint(model=model, step=10)
        info = manager.load_latest_checkpoint()
        assert info is not None
        print("[OK] RollingCheckpointManager")


def test_training_progress():
    from training.so8t_multimodal_moe_pipeline import TrainingProgressTracker

    tracker = TrainingProgressTracker(total_steps=100, desc="Test")
    tracker.update(step=10, metrics={"loss": 0.5})
    tracker.update(step=20, metrics={"loss": 0.4})
    tracker.close()
    print("[OK] TrainingProgressTracker")


def test_full_pipeline_config():
    from training.so8t_multimodal_moe_pipeline import (
        SO8MultimodalMoETrainer,
        SO8MultimodalMoEConfig,
    )

    config = SO8MultimodalMoEConfig(
        model_name_or_path="microsoft/Phi-3.5-mini-instruct",
        num_experts=4,
        use_multimodal=True,
        batch_size=4,
        max_steps=100,
    )
    trainer = SO8MultimodalMoETrainer(config=config)
    assert trainer.config.num_experts == 4
    assert trainer.config.use_multimodal == True
    print("[OK] Full Pipeline Integration")


if __name__ == "__main__":
    print("=" * 60)
    print("SO8T Multimodal MoE Pipeline Tests (Quick)")
    print("=" * 60)
    tests = [
        test_multimodal_moe_config,
        test_so8_triality_router,
        test_moe_layer,
        test_ebbinghaus_curve,
        test_shinka_evolve,
        test_checkpoint_manager,
        test_training_progress,
        test_full_pipeline_config,
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
