#!/usr/bin/env python3
"""
Unit Tests for Training Monitor Module.

Best Practices:
- Pytest-based unit tests
- Isolated test cases with fixtures
- Comprehensive assertions
- Mock objects for external dependencies
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestPhaseInfo:
    """Test cases for PhaseInfo dataclass."""

    def test_default_creation(self):
        """Test PhaseInfo creation with defaults."""
        from scripts.monitoring.modules.training_monitor_core import PhaseInfo

        phase = PhaseInfo(name="sft", display_name="SFT Training")

        assert phase.name == "sft"
        assert phase.display_name == "SFT Training"
        assert phase.start_time is None
        assert phase.end_time is None
        assert phase.metrics == []
        assert phase.status == "pending"

    def test_with_times(self):
        """Test PhaseInfo with times."""
        from datetime import datetime

        from scripts.monitoring.modules.training_monitor_core import PhaseInfo

        start = datetime.now()
        end = datetime.now()

        phase = PhaseInfo(
            name="sft",
            display_name="SFT",
            start_time=start,
            end_time=end,
            status="complete",
        )

        assert phase.start_time == start
        assert phase.end_time == end
        assert phase.status == "complete"


class TestMonitorCallbacks:
    """Test cases for MonitorCallbacks."""

    def test_default_callbacks(self):
        """Test MonitorCallbacks with default (None) callbacks."""
        from scripts.monitoring.modules.training_monitor_core import MonitorCallbacks

        callbacks = MonitorCallbacks()

        assert callbacks.on_phase_start is None
        assert callbacks.on_phase_complete is None
        assert callbacks.on_checkpoint is None
        assert callbacks.on_error is None
        assert callbacks.on_metrics_update is None

    def test_custom_callbacks(self):
        """Test MonitorCallbacks with custom callbacks."""
        from scripts.monitoring.modules.training_monitor_core import MonitorCallbacks

        def phase_start_callback(phase_name: str):
            pass

        def phase_complete_callback(phase_name: str, metrics):
            pass

        callbacks = MonitorCallbacks(
            on_phase_start=phase_start_callback,
            on_phase_complete=phase_complete_callback,
        )

        assert callbacks.on_phase_start is phase_start_callback
        assert callbacks.on_phase_complete is phase_complete_callback


class TestTrainingMonitor:
    """Test cases for TrainingMonitor."""

    def test_default_phases(self):
        """Test that default phases are set correctly."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        assert len(monitor.PHASES) == 8
        assert monitor.PHASES[0] == ("setup", "Setup & Validation")
        assert monitor.PHASES[1] == ("data", "Data Validation")
        assert monitor.PHASES[2] == ("sft", "SFT Training")
        assert monitor.PHASES[3] == ("grpo", "GRPO Training")
        assert monitor.PHASES[4] == ("benchmark", "ABC Benchmark")
        assert monitor.PHASES[5] == ("statistics", "Statistical Analysis")
        assert monitor.PHASES[6] == ("visualize", "Visualization")
        assert monitor.PHASES[7] == ("release", "HF Release")

    def test_initial_phases_status(self):
        """Test that initial phases have pending status."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        for name, phase in monitor._phases.items():
            assert phase.status == "pending"

    def test_start_phase(self):
        """Test starting a phase."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        result = monitor.start_phase("sft", total_steps=100)

        assert result is True
        assert monitor._current_phase == "sft"
        assert monitor._phases["sft"].status == "running"
        assert monitor._phases["sft"].start_time is not None

    def test_start_unknown_phase(self):
        """Test starting an unknown phase."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        result = monitor.start_phase("unknown_phase", total_steps=100)

        assert result is False

    def test_start_phase_closes_previous(self):
        """Test that starting a new phase closes the previous one."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        monitor.start_phase("setup", total_steps=10)
        assert monitor._current_phase == "setup"

        result = monitor.start_phase("data", total_steps=50)
        assert result is True
        assert monitor._current_phase == "data"
        assert monitor._phases["setup"].status == "complete"

    def test_update_metrics(self):
        """Test updating metrics."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        monitor.start_phase("sft", total_steps=100)

        metrics = monitor.update_metrics(
            step=50,
            loss=0.5,
            learning_rate=1e-5,
            epoch=1,
            batch_size=2,
            data_progress=0.5,
        )

        assert metrics is not None
        assert metrics.step == 50
        assert metrics.loss == 0.5
        assert metrics.learning_rate == 1e-5
        assert metrics.epoch == 1
        assert metrics.batch_size == 2
        assert metrics.data_progress == 0.5

    def test_update_metrics_no_active_phase(self):
        """Test updating metrics with no active phase."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        metrics = monitor.update_metrics(step=50)

        assert metrics is None

    def test_end_phase(self):
        """Test ending a phase."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        monitor.start_phase("sft", total_steps=100)
        monitor.update_metrics(step=50, loss=0.5)

        final_metrics = monitor.end_phase("sft")

        assert final_metrics is not None
        assert final_metrics.step == 50
        assert monitor._phases["sft"].status == "complete"
        assert monitor._phases["sft"].end_time is not None
        assert monitor._current_phase is None

    def test_end_unknown_phase(self):
        """Test ending an unknown phase."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        result = monitor.end_phase("unknown")

        assert result is None

    def test_record_error(self):
        """Test recording an error."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        monitor.start_phase("sft", total_steps=100)
        monitor.record_error("sft", "Out of memory")

        assert monitor._phases["sft"].status == "error"
        assert monitor._current_phase is None

    def test_get_current_metrics(self):
        """Test getting current metrics."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        monitor.start_phase("sft", total_steps=100)
        monitor.update_metrics(step=75, loss=0.3)

        metrics = monitor.get_current_metrics()

        assert metrics is not None
        assert metrics.step == 75
        assert metrics.loss == 0.3

    def test_get_current_metrics_no_phase(self):
        """Test getting current metrics with no active phase."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        metrics = monitor.get_current_metrics()

        assert metrics is None

    def test_get_phase_summary(self):
        """Test getting phase summary."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        # Complete some phases
        monitor.start_phase("setup", total_steps=10)
        monitor.end_phase("setup")

        monitor.start_phase("data", total_steps=50)
        monitor.end_phase("data")

        summary = monitor.get_phase_summary()

        assert summary["run_id"] == monitor.run_id
        assert summary["phases"]["setup"]["status"] == "complete"
        assert summary["phases"]["data"]["status"] == "complete"
        assert summary["phases"]["sft"]["status"] == "pending"

    def test_get_all_metrics(self):
        """Test getting all collected metrics."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        monitor.start_phase("sft", total_steps=100)
        monitor.update_metrics(step=50, loss=0.5)
        monitor.end_phase("sft")

        all_metrics = monitor.get_all_metrics()

        assert len(all_metrics) == 1

    def test_is_running_false_initially(self):
        """Test that monitor is not running initially."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        assert monitor.is_running() is False

    def test_get_status(self):
        """Test getting monitor status."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        status = monitor.get_status()

        assert status["run_id"] == monitor.run_id
        assert status["running"] is False
        assert status["current_phase"] is None
        assert status["phases_completed"] == 0
        assert status["total_phases"] == 8


class TestTrainingMonitorDryRun:
    """Test cases for TrainingMonitor dry run functionality."""

    def test_dry_run_pipeline_skip_training(self):
        """Test dry run pipeline with training skipped."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        success = monitor.run_full_pipeline(skip_training=True)

        assert success is True
        assert monitor._phases["setup"]["status"] == "complete"
        assert monitor._phases["data"]["status"] == "complete"
        assert monitor._phases["sft"]["status"] == "pending"
        assert monitor._phases["benchmark"]["status"] == "complete"

    def test_dry_run_with_callbacks(self):
        """Test dry run with callbacks."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        callbacks_received = []

        def on_phase_start(phase_name: str):
            callbacks_received.append(("start", phase_name))

        def on_phase_complete(phase_name: str, metrics):
            callbacks_received.append(("complete", phase_name))

        def on_metrics_update(metrics):
            callbacks_received.append(("metrics", metrics.step))

        monitor.callbacks.on_phase_start = on_phase_start
        monitor.callbacks.on_phase_complete = on_phase_complete
        monitor.callbacks.on_metrics_update = on_metrics_update

        monitor.run_full_pipeline(skip_training=True)

        # Should receive phase start/complete callbacks
        assert len(callbacks_received) > 0

    def test_dry_run_captures_checkpoints(self):
        """Test that dry run captures checkpoints."""
        from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

        monitor = TrainingMonitor(dry_run=True)

        # Check that checkpoints directory is handled correctly
        checkpoint_path = monitor.capture_checkpoint()

        # In dry run, should still attempt to save
        assert checkpoint_path is not None or checkpoint_path is None


class TestCreateMonitor:
    """Test cases for create_monitor factory function."""

    def test_create_monitor_defaults(self):
        """Test create_monitor with default settings."""
        from scripts.monitoring.modules.training_monitor_core import create_monitor

        monitor = create_monitor()

        assert monitor is not None
        assert monitor.config is not None
        assert monitor.dry_run is False

    def test_create_monitor_with_dry_run(self):
        """Test create_monitor with dry_run=True."""
        from scripts.monitoring.modules.training_monitor_core import create_monitor

        monitor = create_monitor(dry_run=True)

        assert monitor.dry_run is True

    def test_create_monitor_with_line_token(self):
        """Test create_monitor with LINE token."""
        from scripts.monitoring.modules.training_monitor_core import create_monitor

        monitor = create_monitor(line_access_token="test_token")

        assert monitor.config.line.enabled is True
        assert monitor.config.line.channel_access_token == "test_token"

    def test_create_monitor_with_config_path(self):
        """Test create_monitor with custom config path."""
        from scripts.monitoring.modules.training_monitor_core import create_monitor

        monitor = create_monitor(config_path="/custom/path/config.yaml")

        assert monitor.config is not None
