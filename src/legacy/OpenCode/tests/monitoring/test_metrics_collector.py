#!/usr/bin/env python3
"""
Unit Tests for Metrics Collector Module.

Best Practices:
- Pytest-based unit tests
- Isolated test cases with fixtures
- Comprehensive assertions
- Mock objects for external dependencies
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTrainingMetrics:
    """Test cases for TrainingMetrics dataclass."""

    def test_default_creation(self):
        """Test TrainingMetrics creation with defaults."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics()

        assert metrics.loss is None
        assert metrics.learning_rate is None
        assert metrics.step == 0
        assert metrics.total_steps == 0
        assert metrics.epoch == 1
        assert metrics.total_epochs == 1
        assert metrics.batch_size == 1
        assert metrics.data_progress == 0.0
        assert metrics.gpu_memory_used_gb == 0.0
        assert metrics.gpu_memory_total_gb == 0.0
        assert metrics.eta_seconds is None
        assert metrics.elapsed_seconds == 0.0
        assert metrics.status == "running"

    def test_full_creation(self):
        """Test TrainingMetrics creation with all fields."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics(
            run_id="test_run_001",
            loss=0.0234,
            learning_rate=2e-5,
            step=150,
            total_steps=500,
            epoch=1,
            total_epochs=3,
            batch_size=2,
            data_progress=0.3,
            phase_progress=0.3,
            gpu_memory_used_gb=8.2,
            gpu_memory_total_gb=12.0,
            gpu_utilization=68.5,
            eta_seconds=3600,
            elapsed_seconds=900,
            phase_name="sft",
        )

        assert metrics.run_id == "test_run_001"
        assert metrics.loss == 0.0234
        assert metrics.learning_rate == 2e-5
        assert metrics.step == 150
        assert metrics.eta_seconds == 3600

    def test_to_dict(self):
        """Test TrainingMetrics serialization."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics(
            run_id="test_run",
            loss=0.5,
            step=100,
            timestamp=datetime(2026, 2, 2, 12, 0, 0),
        )

        data = metrics.to_dict()

        assert isinstance(data, dict)
        assert data["run_id"] == "test_run"
        assert data["loss"] == 0.5
        assert data["step"] == 100
        assert data["timestamp"] == "2026-02-02T12:00:00"

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics(
            step=250,
            total_steps=1000,
        )

        assert metrics.progress_percentage == 25.0

    def test_progress_percentage_zero_total(self):
        """Test progress percentage with zero total steps."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics(
            step=100,
            total_steps=0,
        )

        assert metrics.progress_percentage == 0.0

    def test_eta_formatted_hours(self):
        """Test ETA formatting with hours."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.eta_seconds = 3665  # 1h 1m 5s

        assert metrics.eta_formatted == "1h 1m 5s"

    def test_eta_formatted_minutes(self):
        """Test ETA formatting with minutes."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.eta_seconds = 125  # 2m 5s

        assert metrics.eta_formatted == "2m 5s"

    def test_eta_formatted_none(self):
        """Test ETA formatting with None."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.eta_seconds = None

        assert metrics.eta_formatted == "N/A"

    def test_eta_formatted_days(self):
        """Test ETA formatting with days."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.eta_seconds = 90065  # 1d 1h 1m 5s

        assert metrics.eta_formatted == "1d 1h"

    def test_elapsed_formatted(self):
        """Test elapsed time formatting."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics()
        metrics.elapsed_seconds = 3723  # 1h 2m 3s

        assert metrics.elapsed_formatted == "1h 2m 3s"

    def test_gpu_memory_percentage(self):
        """Test GPU memory percentage calculation."""
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        metrics = TrainingMetrics(
            gpu_memory_used_gb=8.0,
            gpu_memory_total_gb=12.0,
        )

        assert metrics.gpu_memory_percentage == pytest.approx(66.67, rel=0.01)


class TestETACalculator:
    """Test cases for ETACalculator."""

    def test_initial_state(self):
        """Test ETA calculator initial state."""
        from scripts.monitoring.modules.metrics_collector import ETACalculator

        calculator = ETACalculator()

        assert calculator._ewma_step_time is None
        assert calculator._step_times == []

    def test_update_without_history(self):
        """Test ETA update without historical data."""
        from scripts.monitoring.modules.metrics_collector import ETACalculator

        calculator = ETACalculator()
        eta = calculator.update(step=10, elapsed_seconds=1.0)

        # Should return None due to insufficient samples
        assert eta is None

    def test_update_with_history(self):
        """Test ETA update with historical data."""
        from scripts.monitoring.modules.metrics_collector import ETACalculator

        calculator = ETACalculator(min_samples=5)

        # Add historical data
        for i in range(1, 11):
            time.sleep(0.001)
            calculator.update(step=i * 10, elapsed_seconds=i * 1.0)

        eta = calculator.calculate_eta(current_step=50, total_steps=100)

        # Should have valid ETA after min_samples
        assert eta is not None
        assert eta > 0

    def test_reset(self):
        """Test ETA calculator reset."""
        from scripts.monitoring.modules.metrics_collector import ETACalculator

        calculator = ETACalculator()

        # Add some data
        for i in range(1, 20):
            time.sleep(0.001)
            calculator.update(step=i * 10, elapsed_seconds=i * 1.0)

        calculator.reset()

        assert calculator._ewma_step_time is None
        assert calculator._step_times == []

    def test_calculate_eta_with_sufficient_samples(self):
        """Test ETA calculation with sufficient samples."""
        from scripts.monitoring.modules.metrics_collector import ETACalculator

        calculator = ETACalculator(
            smoothing_factor=0.1,
            min_samples=3,
        )

        # Simulate steps
        for i in range(1, 10):
            time.sleep(0.001)
            calculator.update(step=i * 10, elapsed_seconds=i * 0.5)

        eta = calculator.calculate_eta(current_step=50, total_steps=100)

        assert eta is not None
        assert eta >= 0


class TestGPUMonitor:
    """Test cases for GPU monitor implementations."""

    def test_simulated_gpu_monitor(self):
        """Test SimulatedGPUMonitor."""
        from scripts.monitoring.modules.metrics_collector import SimulatedGPUMonitor

        monitor = SimulatedGPUMonitor(used_gb=8.5, total_gb=12.0)

        assert monitor.get_memory_used_gb() == 8.5
        assert monitor.get_memory_total_gb() == 12.0
        assert monitor.get_utilization() == 68.0

    def test_simulated_gpu_monitor_custom_values(self):
        """Test SimulatedGPUMonitor with custom values."""
        from scripts.monitoring.modules.metrics_collector import SimulatedGPUMonitor

        monitor = SimulatedGPUMonitor(used_gb=4.0, total_gb=8.0)

        assert monitor.get_memory_used_gb() == 4.0
        assert monitor.get_memory_total_gb() == 8.0
        assert monitor.get_utilization() == 50.0


class TestMetricsCollector:
    """Test cases for MetricsCollector."""

    def test_initial_state(self):
        """Test MetricsCollector initial state."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")

        assert collector.run_id == "test_run"
        assert collector._start_time is None

    def test_start(self):
        """Test MetricsCollector start."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=1000, total_epochs=3)

        assert collector._start_time is not None
        assert collector._current_metrics.total_steps == 1000
        assert collector._current_metrics.total_epochs == 3
        assert collector._current_metrics.status == "running"

    def test_update_basic(self):
        """Test MetricsCollector update with basic fields."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=100)

        metrics = collector.update(
            step=50,
            loss=0.5,
            learning_rate=1e-5,
        )

        assert metrics.step == 50
        assert metrics.loss == 0.5
        assert metrics.learning_rate == 1e-5

    def test_update_all_fields(self):
        """Test MetricsCollector update with all fields."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=200, total_epochs=5)

        metrics = collector.update(
            step=100,
            loss=0.25,
            learning_rate=5e-6,
            epoch=2,
            batch_size=4,
            data_progress=0.5,
        )

        assert metrics.step == 100
        assert metrics.loss == 0.25
        assert metrics.learning_rate == 5e-6
        assert metrics.epoch == 2
        assert metrics.batch_size == 4
        assert metrics.data_progress == 0.5

    def test_complete(self):
        """Test MetricsCollector complete."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=100)
        collector.update(step=50, loss=0.5)

        metrics = collector.complete(status="complete")

        assert metrics.status == "complete"
        assert metrics.step == 50

    def test_error(self):
        """Test MetricsCollector error handling."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=100)

        metrics = collector.error("Out of memory")

        assert metrics.status == "error"
        assert metrics.error_message == "Out of memory"

    def test_reset(self):
        """Test MetricsCollector reset."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=100)
        collector.update(step=50, loss=0.5)

        collector.reset()

        assert collector._start_time is None
        assert collector._current_metrics.step == 0

    def test_get_current_metrics(self):
        """Test getting current metrics snapshot."""
        from scripts.monitoring.modules.metrics_collector import MetricsCollector

        collector = MetricsCollector(run_id="test_run")
        collector.start(total_steps=100)
        collector.update(step=75, loss=0.3)

        snapshot = collector.get_current_metrics()

        assert snapshot.step == 75
        assert snapshot.loss == 0.3


class TestMetricsCollectorWithGPU:
    """Test cases for MetricsCollector with GPU monitoring."""

    def test_with_simulated_gpu(self):
        """Test MetricsCollector with simulated GPU."""
        from scripts.monitoring.modules.metrics_collector import (
            MetricsCollector,
            SimulatedGPUMonitor,
        )

        gpu_monitor = SimulatedGPUMonitor(used_gb=8.0, total_gb=12.0)
        collector = MetricsCollector(
            run_id="test_run",
            gpu_monitor=gpu_monitor,
        )
        collector.start(total_steps=100)

        metrics = collector.update(step=50)

        assert metrics.gpu_memory_used_gb == 8.0
        assert metrics.gpu_memory_total_gb == 12.0
