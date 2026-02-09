# -*- coding: utf-8 -*-
"""
SO8T Extended Multimodal MoE Pipeline Tests

Tests comprehensive data collection and training:
- SO8 orthogonal transformations
- YouTube video collection
- HuggingFace datasets
- Multimodal CoT datasets
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

import torch


def test_so8_orthogonal_transform():
    from training.so8t_extended_moe_pipeline import SO8OrthogonalTransform

    transform = SO8OrthogonalTransform(dim=768)
    x = torch.randn(2, 10, 768)
    output = transform(x)
    assert output.shape == x.shape
    print("[OK] SO8OrthogonalTransform")


def test_so8_triality_router():
    from training.so8t_extended_moe_pipeline import SO8TrialityRouter

    router = SO8TrialityRouter(num_experts=4, hidden_dim=768)
    batch, seq = 2, 64
    x = torch.randn(batch, seq, 768)
    expert_indices, routing_weights = router(x)
    assert routing_weights.shape == (batch, 4)
    print("[OK] SO8TrialityRouter")


def test_moe_layer():
    from training.so8t_extended_moe_pipeline import SO8MoELayer, SO8MultimodalMoEConfig

    config = SO8MultimodalMoEConfig(
        hidden_dim=768,
        num_experts=4,
        use_so8_transform=True,
    )
    moe = SO8MoELayer(config)
    batch, seq = 2, 64
    x = torch.randn(batch, seq, 768)
    output = moe(x)
    assert output.shape == x.shape
    print("[OK] SO8MoELayer with SO8 transform")


def test_audio_encoder():
    from training.so8t_extended_moe_pipeline import AudioEncoder

    encoder = AudioEncoder(hidden_dim=768)
    audio = torch.randn(16000)
    features = encoder(audio)
    assert features is not None
    assert features.numel() > 0
    print("[OK] AudioEncoder")


def test_ebbinghaus_curve():
    from training.so8t_extended_moe_pipeline import EbbinghausForgettingCurve

    curve = EbbinghausForgettingCurve()
    curve.update([1, 2, 3], is_reinforced=[True, False, False])
    stats = curve.get_stats()
    assert "avg_retention" in stats
    print("[OK] EbbinghausForgettingCurve")


def test_shinka_evolve():
    from training.so8t_extended_moe_pipeline import (
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
    from training.so8t_extended_moe_pipeline import RollingCheckpointManager
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
    from training.so8t_extended_moe_pipeline import TrainingProgressTracker

    tracker = TrainingProgressTracker(total_steps=100, desc="Test")
    tracker.update(step=10, metrics={"loss": 0.5})
    tracker.update(step=20, metrics={"loss": 0.4})
    tracker.close()
    print("[OK] TrainingProgressTracker")


def test_extended_config():
    from training.so8t_extended_moe_pipeline import SO8MultimodalMoEConfig

    config = SO8MultimodalMoEConfig(
        model_name_or_path="microsoft/Phi-3.5-mini-instruct",
        num_experts=4,
        use_multimodal=True,
        use_so8_transform=True,
        use_youtube=True,
        use_hf_datasets=True,
    )
    assert config.num_experts == 4
    assert config.use_multimodal == True
    assert config.use_so8_transform == True
    print("[OK] Extended SO8MultimodalMoEConfig")


def test_multimodal_dataset_loader():
    from training.so8t_extended_moe_pipeline import (
        ExtendedMultimodalDataset,
        SO8MultimodalMoEConfig,
    )

    config = SO8MultimodalMoEConfig()
    data = [
        {"text": "Sample text", "reasoning": "Reasoning", "reasoning_type": "general"},
        {
            "text": "Another text",
            "reasoning": "More reasoning",
            "reasoning_type": "visual",
        },
    ]
    dataset = ExtendedMultimodalDataset(data, config=config)
    assert len(dataset) == 2
    item = dataset[0]
    assert "text" in item
    print("[OK] ExtendedMultimodalDataset")


def test_full_pipeline_config():
    from training.so8t_extended_moe_pipeline import (
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
    print("[OK] Full Extended Pipeline Integration")


if __name__ == "__main__":
    print("=" * 60)
    print("SO8T Extended Multimodal MoE Pipeline Tests")
    print("=" * 60)
    tests = [
        test_so8_orthogonal_transform,
        test_so8_triality_router,
        test_moe_layer,
        test_audio_encoder,
        test_ebbinghaus_curve,
        test_shinka_evolve,
        test_checkpoint_manager,
        test_training_progress,
        test_extended_config,
        test_multimodal_dataset_loader,
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
