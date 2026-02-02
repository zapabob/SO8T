#!/usr/bin/env python3
"""
Training Monitor Module.

Best Practices:
- Comprehensive training monitoring with LINE notifications
- Phase-based progress tracking
- Checkpoint management
- Error handling and recovery
- Thread-safe operation
- Modular architecture with dependency injection
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "monitoring"))

# Direct imports
from scripts.monitoring.modules import config_loader
from scripts.monitoring.modules import metrics_collector
from scripts.monitoring.modules import line_notifier


@dataclass
class PhaseInfo:
    """Information about a training phase."""

    name: str
    display_name: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: List[metrics_collector.TrainingMetrics] = field(default_factory=list)
    status: str = "pending"  # pending, running, complete, error


class MonitorCallbacks:
    """Callback functions for monitoring events."""

    def __init__(
        self,
        on_phase_start: Optional[Callable[[str], None]] = None,
        on_phase_complete: Optional[
            Callable[[str, metrics_collector.TrainingMetrics], None]
        ] = None,
        on_checkpoint: Optional[
            Callable[[metrics_collector.TrainingMetrics], None]
        ] = None,
        on_error: Optional[Callable[[str, Exception], None]] = None,
        on_metrics_update: Optional[
            Callable[[metrics_collector.TrainingMetrics], None]
        ] = None,
    ):
        self.on_phase_start = on_phase_start
        self.on_phase_complete = on_phase_complete
        self.on_checkpoint = on_checkpoint
        self.on_error = on_error
        self.on_metrics_update = on_metrics_update


class TrainingMonitor:
    """
    Comprehensive training monitor with LINE notifications.

    Features:
    - Phase-based progress tracking
    - Automatic checkpoint capture
    - LINE notifications for all events
    - ETA calculation
    - Error handling and recovery
    - Dry run support
    """

    PHASES = [
        ("setup", "Setup & Validation"),
        ("data", "Data Validation"),
        ("sft", "SFT Training"),
        ("grpo", "GRPO Training"),
        ("benchmark", "ABC Benchmark"),
        ("statistics", "Statistical Analysis"),
        ("visualize", "Visualization"),
        ("release", "HF Release"),
    ]

    def __init__(
        self,
        config: Optional[config_loader.MonitoringSettings] = None,
        line_notifier: Optional[line_notifier.LineNotifier] = None,
        gpu_monitor: Optional[metrics_collector.GPUMonitor] = None,
        callbacks: Optional[MonitorCallbacks] = None,
        run_id: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.config = config or config_loader.get_config()
        self.line_notifier = line_notifier
        self.gpu_monitor = gpu_monitor or metrics_collector.SimulatedGPUMonitor()
        self.callbacks = callbacks or MonitorCallbacks()
        self.run_id = run_id or f"moonshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.dry_run = dry_run

        self._logger = logging.getLogger(__name__)
        self._setup_logging()

        self._phases: Dict[str, PhaseInfo] = {}
        self._current_phase: Optional[str] = None
        self._metrics_collectors: Dict[str, metrics_collector.MetricsCollector] = {}
        self._start_time: Optional[datetime] = None
        self._lock = threading.Lock()
        self._running = False
        self._shutdown_requested = False

        # Initialize phases
        for name, display_name in self.PHASES:
            self._phases[name] = PhaseInfo(name=name, display_name=display_name)

        self._logger.info(f"TrainingMonitor initialized for run: {self.run_id}")

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_level = getattr(logging, self.config.logging.level, logging.INFO)
        log_format = self.config.logging.format

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))

        # File handler
        if self.config.logging.file_enabled:
            log_path = Path(self.config.logging.file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(logging.Formatter(log_format))
        else:
            file_handler = None

        # Configure root logger
        self._logger.setLevel(log_level)
        self._logger.handlers = [console_handler]
        if file_handler:
            self._logger.addHandler(file_handler)

    def _create_metrics_collector(
        self, phase_name: str
    ) -> metrics_collector.MetricsCollector:
        """Create a metrics collector for a phase."""
        eta_calculator = metrics_collector.ETACalculator(
            smoothing_factor=self.config.eta.smoothing_factor,
            min_samples=self.config.eta.min_samples_for_eta,
        )

        collector = metrics_collector.MetricsCollector(
            run_id=self.run_id,
            gpu_monitor=self.gpu_monitor,
            eta_calculator=eta_calculator,
        )

        return collector

    def start_phase(
        self, phase_name: str, total_steps: int = 1000, total_epochs: int = 1
    ) -> bool:
        """
        Start a monitoring phase.

        Args:
            phase_name: Name of the phase
            total_steps: Total number of steps in this phase
            total_epochs: Total number of epochs

        Returns:
            True if phase started successfully
        """
        with self._lock:
            if phase_name not in self._phases:
                self._logger.error(f"Unknown phase: {phase_name}")
                return False

            if self._current_phase is not None:
                self._logger.warning(f"Closing current phase: {self._current_phase}")
                self.end_phase(self._current_phase)

            self._current_phase = phase_name
            phase = self._phases[phase_name]
            phase.status = "running"
            phase.start_time = datetime.now()

            # Create metrics collector
            collector = self._create_metrics_collector(phase_name)
            collector.start(total_steps=total_steps, total_epochs=total_epochs)
            self._metrics_collectors[phase_name] = collector

            self._logger.info(f"Started phase: {phase_name} ({phase.display_name})")

            # Send LINE notification
            if self.line_notifier and self.config.notifications.phase_start:
                try:
                    emoji = "[SETUP]" if phase_name == "setup" else "[TRAINING]"
                    message = f"{emoji} Phase Started: {phase.display_name}"
                    self.line_notifier.send_message(message)
                except Exception as e:
                    self._logger.warning(
                        f"Failed to send phase start notification: {e}"
                    )

            # Call callback
            if self.callbacks.on_phase_start:
                try:
                    self.callbacks.on_phase_start(phase_name)
                except Exception as e:
                    self._logger.warning(f"Callback error: {e}")

            return True

    def update_metrics(
        self,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        epoch: Optional[int] = None,
        batch_size: Optional[int] = None,
        data_progress: Optional[float] = None,
    ) -> Optional[metrics_collector.TrainingMetrics]:
        """
        Update metrics for the current phase.

        Args:
            step: Current step number
            loss: Current loss value
            learning_rate: Current learning rate
            epoch: Current epoch number
            batch_size: Batch size
            data_progress: Data processing progress

        Returns:
            Updated TrainingMetrics or None if no active phase
        """
        if self._current_phase is None:
            self._logger.warning("No active phase for metrics update")
            return None

        collector = self._metrics_collectors.get(self._current_phase)
        if collector is None:
            return None

        metrics = collector.update(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            epoch=epoch,
            batch_size=batch_size,
            data_progress=data_progress,
        )

        # Set phase name
        metrics.phase_name = self._current_phase

        # Check for checkpoint interval
        if (
            step > 0
            and step % (self.config.monitoring.checkpoint_interval // 10) == 0
            and self.callbacks.on_checkpoint
        ):
            try:
                self.callbacks.on_checkpoint(metrics)
            except Exception as e:
                self._logger.warning(f"Checkpoint callback error: {e}")

        # Check for metrics update callback
        if self.callbacks.on_metrics_update:
            try:
                self.callbacks.on_metrics_update(metrics)
            except Exception as e:
                self._logger.warning(f"Metrics update callback error: {e}")

        return metrics

    def end_phase(self, phase_name: str) -> Optional[metrics_collector.TrainingMetrics]:
        """
        End a monitoring phase.

        Args:
            phase_name: Name of the phase

        Returns:
            Final TrainingMetrics or None
        """
        with self._lock:
            if phase_name not in self._phases:
                return None

            phase = self._phases[phase_name]
            if phase.status != "running":
                return None

            phase.status = "complete"
            phase.end_time = datetime.now()

            collector = self._metrics_collectors.get(phase_name)
            if collector:
                metrics = collector.complete()
                metrics.phase_name = phase_name
                phase.metrics.append(metrics)
            else:
                metrics = None

            # Calculate phase duration
            if phase.start_time and phase.end_time:
                duration = phase.end_time - phase.start_time
                self._logger.info(
                    f"Phase {phase_name} completed in {duration.total_seconds():.1f}s"
                )

            # Send LINE notification
            if (
                self.line_notifier
                and self.config.notifications.phase_complete
                and metrics
            ):
                try:
                    self.line_notifier.send_phase_complete(phase.display_name, metrics)
                except Exception as e:
                    self._logger.warning(
                        f"Failed to send phase complete notification: {e}"
                    )

            # Call callback
            if self.callbacks.on_phase_complete and metrics:
                try:
                    self.callbacks.on_phase_complete(phase_name, metrics)
                except Exception as e:
                    self._logger.warning(f"Phase complete callback error: {e}")

            if self._current_phase == phase_name:
                self._current_phase = None

            return metrics

    def record_error(self, phase_name: str, error_message: str) -> None:
        """
        Record an error in a phase.

        Args:
            phase_name: Name of the phase
            error_message: Error description
        """
        with self._lock:
            if phase_name in self._phases:
                phase = self._phases[phase_name]
                phase.status = "error"

            collector = self._metrics_collectors.get(phase_name)
            if collector:
                collector.error(error_message)

            self._logger.error(f"Error in {phase_name}: {error_message}")

            # Send LINE notification
            if self.line_notifier and self.config.notifications.error_occurred:
                try:
                    self.line_notifier.send_error_alert(error_message, phase_name)
                except Exception as e:
                    self._logger.warning(f"Failed to send error notification: {e}")

            # Call callback
            if self.callbacks.on_error:
                try:
                    self.callbacks.on_error(phase_name, Exception(error_message))
                except Exception as e:
                    self._logger.warning(f"Error callback error: {e}")

    def capture_checkpoint(
        self, checkpoint_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Capture a rolling checkpoint.

        Args:
            checkpoint_path: Optional path to checkpoint file

        Returns:
            Path to captured checkpoint or None
        """
        if self._current_phase is None:
            return None

        collector = self._metrics_collectors.get(self._current_phase)
        if collector is None:
            return None

        metrics = collector.get_current_metrics()
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

        # Create checkpoint data
        checkpoint_data = {
            "run_id": self.run_id,
            "phase": self._current_phase,
            "timestamp": timestamp,
            "metrics": metrics.to_dict(),
            "checkpoints_dir": str(Path(__file__).parent.parent.parent / "checkpoints"),
        }

        if checkpoint_path is None:
            checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(checkpoints_dir / f"checkpoint_{timestamp}.json")

        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            self._logger.info(f"Checkpoint saved: {checkpoint_path}")

            # Send LINE notification
            if self.line_notifier and self.config.notifications.checkpoint_saved:
                try:
                    self.line_notifier.send_message(
                        f"[CHECKPOINT] {self._current_phase} - Step {metrics.step}"
                    )
                except Exception as e:
                    self._logger.warning(f"Failed to send checkpoint notification: {e}")

            return checkpoint_path

        except Exception as e:
            self._logger.error(f"Failed to save checkpoint: {e}")
            return None

    def get_current_metrics(self) -> Optional[metrics_collector.TrainingMetrics]:
        """Get current phase metrics."""
        if self._current_phase is None:
            return None

        collector = self._metrics_collectors.get(self._current_phase)
        if collector:
            return collector.get_current_metrics()
        return None

    def get_phase_summary(self) -> Dict[str, Any]:
        """Get summary of all phases."""
        summary = {
            "run_id": self.run_id,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "phases": {},
        }

        for name, phase in self._phases.items():
            phase_data = {
                "display_name": phase.display_name,
                "status": phase.status,
                "duration_seconds": None,
            }

            if phase.start_time and phase.end_time:
                duration = phase.end_time - phase.start_time
                phase_data["duration_seconds"] = duration.total_seconds()

            if phase.metrics:
                final_metrics = phase.metrics[-1]
                phase_data["final_loss"] = final_metrics.loss
                phase_data["total_steps"] = final_metrics.step

            summary["phases"][name] = phase_data

        return summary

    def get_all_metrics(self) -> List[metrics_collector.TrainingMetrics]:
        """Get all collected metrics."""
        all_metrics = []
        for collector in self._metrics_collectors.values():
            all_metrics.append(collector.get_current_metrics())
        return all_metrics

    def run_full_pipeline(
        self,
        skip_training: bool = False,
        callback: Optional[
            Callable[[str, metrics_collector.TrainingMetrics], None]
        ] = None,
    ) -> bool:
        """
        Run the full monitoring pipeline.

        Args:
            skip_training: Skip SFT and GRPO phases
            callback: Optional callback for metrics updates

        Returns:
            True if pipeline completed successfully
        """
        self._start_time = datetime.now()
        self._running = True

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self._logger.info("Shutdown signal received")
            self._shutdown_requested = True

        old_signal_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_signal_handlers[sig] = signal.signal(sig, signal_handler)

        try:
            # Setup phase
            if not self.start_phase("setup", total_steps=10):
                return False
            time.sleep(1)  # Simulate setup
            self.end_phase("setup")

            # Data phase
            if not self.start_phase("data", total_steps=100):
                return False
            time.sleep(1)  # Simulate data validation
            self.end_phase("data")

            # SFT phase (if not skipped)
            if not skip_training:
                if not self.start_phase("sft", total_steps=500):
                    return False

                # Simulate training with metrics updates
                for step in range(1, 501):
                    if self._shutdown_requested:
                        self._logger.info("Shutdown requested during SFT")
                        return False

                    # Simulated metrics
                    loss = 1.0 / (step / 50 + 1)  # Decreasing loss
                    lr = 2e-5 * (1 - step / 600)  # Decreasing LR
                    epoch = (step // 167) + 1  # ~167 steps per epoch

                    self.update_metrics(
                        step=step,
                        loss=loss,
                        learning_rate=lr,
                        epoch=epoch,
                        batch_size=2,
                        data_progress=step / 500,
                    )

                    if callback:
                        metrics = self.get_current_metrics()
                        if metrics:
                            callback("sft", metrics)

                    if step % 100 == 0:
                        self.capture_checkpoint()

                    time.sleep(0.01)  # Simulate computation

                self.end_phase("sft")

            # GRPO phase (if not skipped)
            if not skip_training:
                if not self.start_phase("grpo", total_steps=300):
                    return False

                for step in range(1, 301):
                    if self._shutdown_requested:
                        self._logger.info("Shutdown requested during GRPO")
                        return False

                    loss = 0.5 / (step / 50 + 1)
                    lr = 1e-5 * (1 - step / 400)
                    epoch = (step // 100) + 1

                    self.update_metrics(
                        step=step,
                        loss=loss,
                        learning_rate=lr,
                        epoch=epoch,
                        batch_size=1,
                        data_progress=step / 300,
                    )

                    if callback:
                        metrics = self.get_current_metrics()
                        if metrics:
                            callback("grpo", metrics)

                    time.sleep(0.01)

                self.end_phase("grpo")

            # Benchmark phase
            if not self.start_phase("benchmark", total_steps=100):
                return False
            time.sleep(2)  # Simulate benchmark
            self.end_phase("benchmark")

            # Statistics phase
            if not self.start_phase("statistics", total_steps=50):
                return False
            time.sleep(1)  # Simulate statistics
            self.end_phase("statistics")

            # Visualization phase
            if not self.start_phase("visualize", total_steps=30):
                return False
            time.sleep(1)  # Simulate visualization
            self.end_phase("visualize")

            # Release phase
            if not self.start_phase("release", total_steps=20):
                return False
            time.sleep(1)  # Simulate release
            self.end_phase("release")

            # Send completion notification
            if self.line_notifier:
                total_time = (datetime.now() - self._start_time).total_seconds()
                metrics = (
                    self.get_current_metrics()
                    or metrics_collector.TrainingMetrics(run_id=self.run_id)
                )
                try:
                    self.line_notifier.send_training_complete(
                        metrics, f"{total_time:.0f}s"
                    )
                except Exception as e:
                    self._logger.warning(f"Failed to send completion notification: {e}")

            self._logger.info("Pipeline completed successfully")
            return True

        except Exception as e:
            self._logger.error(f"Pipeline error: {e}")
            if self._current_phase:
                self.record_error(self._current_phase, str(e))
            return False

        finally:
            self._running = False
            # Restore signal handlers
            for sig, handler in old_signal_handlers.items():
                signal.signal(sig, handler)

    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        return {
            "run_id": self.run_id,
            "running": self._running,
            "current_phase": self._current_phase,
            "phases_completed": sum(
                1 for p in self._phases.values() if p.status == "complete"
            ),
            "total_phases": len(self._phases),
            "start_time": self._start_time.isoformat() if self._start_time else None,
        }


def create_monitor(
    config_path: Optional[str] = None,
    line_access_token: Optional[str] = None,
    dry_run: bool = False,
) -> TrainingMonitor:
    """
    Factory function to create a TrainingMonitor with dependencies.

    Args:
        config_path: Path to config file
        line_access_token: LINE access token
        dry_run: Use dry run mode

    Returns:
        Configured TrainingMonitor instance
    """
    config = config_loader.get_config(config_path)

    # Override LINE settings
    if line_access_token:
        config.line.channel_access_token = line_access_token
        config.line.enabled = True

    # Create LINE notifier if enabled
    line_notifier = None
    if config.line.enabled and config.line.channel_access_token:
        line_notifier = LineNotifier(
            access_token=config.line.channel_access_token,
            channel_secret=config.line.channel_secret,
            gateway_url=config.line.gateway_url,
            retry_attempts=config.line.retry_attempts,
            retry_delay=config.line.retry_delay_seconds,
            message_format=config.line.message_format,
        )

    return TrainingMonitor(
        config=config,
        line_notifier=line_notifier,
        dry_run=dry_run,
    )
