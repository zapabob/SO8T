#!/usr/bin/env python3
"""Test ABC Pipeline components"""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.abc_pipeline import (
    MODELS,
    BENCHMARK_SUITE,
    PipelineState,
    RollingCheckpointManager,
    FreezeParameterEvolver,
    ABCBenchmarkHarness,
    StatisticalAnalyzer,
    ModelCardGenerator,
)


def test_models():
    """Test model configurations"""
    print("[TEST] Model Configurations")
    for key, config in MODELS.items():
        print(f"  {key}: {config.name}")
        print(f"    Ollama: {config.ollama_name}")
        print(f"    HF: {config.hf_repo_id}")
        print(f"    Pipeline Output: {config.is_pipeline_output}")
    assert len(MODELS) == 3
    print("[OK] Models configured\n")


def test_benchmark_suite():
    """Test benchmark suite"""
    print("[TEST] Benchmark Suite")
    total = 0
    for category, tasks in BENCHMARK_SUITE.items():
        count = sum(samples for _, samples in tasks)
        total += count
        print(f"  {category}: {count} samples across {len(tasks)} tasks")
    print(f"  Total: {total} samples")
    assert total > 0
    print("[OK] Benchmark suite valid\n")


def test_checkpoint_manager():
    """Test checkpoint manager"""
    print("[TEST] Rolling Checkpoint Manager")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        manager = RollingCheckpointManager(
            Path(tmpdir), interval_seconds=1, max_slots=2
        )
        state = PipelineState(
            phase="test",
            start_time=0,
            checkpoint_time=0,
            models_tested=["A"],
            benchmarks_completed=["test"],
            current_checkpoint=1,
        )
        manager.start_monitoring(state)
        time.sleep(2)
        manager.save_checkpoint()
        manager.stop()
        assert len(state.checkpoint_files) > 0
    print("[OK] Checkpoint manager functional\n")


def test_freeze_evolver():
    """Test freeze parameter evolution"""
    print("[TEST] Freeze Parameter Evolver")
    evolver = FreezeParameterEvolver(initial_freeze_rate=0.95, elimination_rate=0.02)
    for gen in range(5):
        scores = {"A": 0.80, "B": 0.75, "C": 0.85}
        result = evolver.evolve(scores)
        print(
            f"  Gen {result['generation']}: freeze_rate={result['freeze_rate']:.3f}, "
            f"avg={result['avg_score']:.3f}, pressure={result['elimination_pressure']}"
        )
    assert evolver.generation == 5
    print("[OK] Freeze evolution functional\n")


def test_statistical_analyzer():
    """Test statistical analysis"""
    print("[TEST] Statistical Analyzer")
    analyzer = StatisticalAnalyzer()
    scores = [0.72, 0.78, 0.75, 0.80, 0.77, 0.73, 0.79, 0.76]
    stats = analyzer.compute_statistics(scores)
    print(f"  N: {stats['n']}")
    print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"  95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
    print(f"  Acceptable: {stats['is_acceptable']}")
    assert stats["mean"] > 0.70
    assert stats["is_acceptable"]
    print("[OK] Statistical analysis valid\n")


def test_model_comparison():
    """Test model comparison"""
    print("[TEST] Model Comparison")
    analyzer = StatisticalAnalyzer()
    scores_a = [0.72, 0.75, 0.78, 0.73, 0.76]
    scores_b = [0.80, 0.82, 0.79, 0.81, 0.83]
    comparison = analyzer.compare_models(scores_a, scores_b)
    print(f"  Model A mean: {comparison['model_a']['mean']:.4f}")
    print(f"  Model B mean: {comparison['model_b']['mean']:.4f}")
    print(f"  p-value: {comparison['p_value']:.4f}")
    print(f"  Winner: {comparison['winner']}")
    print(f"  Significant: {comparison['significant_difference']}")
    print("[OK] Model comparison functional\n")


def test_visualization():
    """Test model card visualization"""
    print("[TEST] Model Card Generator")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        generator = ModelCardGenerator(Path(tmpdir))
        results = {
            "A": {"mean": 0.75, "std": 0.03, "ci_95": (0.70, 0.80)},
            "B": {"mean": 0.78, "std": 0.04, "ci_95": (0.72, 0.84)},
            "C": {"mean": 0.82, "std": 0.02, "ci_95": (0.79, 0.85)},
        }
        plot = generator.create_errorbar_plot(results, "Test", "test_benchmark.png")
        assert Path(plot).exists()
        print(f"  Generated: {Path(plot).name}")
    print("[OK] Visualization functional\n")


def main():
    print("=" * 60)
    print("ABC Pipeline Component Tests")
    print("=" * 60)
    print()

    test_models()
    test_benchmark_suite()
    test_checkpoint_manager()
    test_freeze_evolver()
    test_statistical_analyzer()
    test_model_comparison()
    test_visualization()

    print("=" * 60)
    print("[OK] All ABC Pipeline tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
