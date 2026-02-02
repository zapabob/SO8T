#!/usr/bin/env python3
"""
Training Monitor - Main Entry Point.

Best Practices:
- Clean public API
- Comprehensive documentation
- Easy integration
- Best practices for MLOps monitoring
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Add project root and scripts to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.monitoring.modules.config_loader import get_config
from scripts.monitoring.modules.line_notifier import LineNotifier
from scripts.monitoring.modules.metrics_collector import TrainingMetrics
from scripts.monitoring.modules.training_monitor_core import (
    MonitorCallbacks,
    TrainingMonitor,
    create_monitor,
)


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=level, format=log_format)
    return logging.getLogger(__name__)


def print_banner() -> None:
    """Print the monitoring banner."""
    banner = """
================================================================================
    Moonshot Pipeline v3.0 - Training Monitor
    --------------------------------------------------------------
    Real-time training monitoring with LINE notifications
    Phase tracking | Checkpoints | ETA calculation
================================================================================
    """
    print(banner)


def print_metrics_summary(metrics: TrainingMetrics, phase_name: str) -> None:
    """Print a summary of current metrics."""
    print("\n" + "=" * 60)
    print(f"[METRICS] {phase_name.upper()} - Update")
    print("=" * 60)

    if metrics.loss is not None:
        print(f"  Loss:          {metrics.loss:.4f}")
    if metrics.learning_rate is not None:
        print(f"  Learning Rate: {metrics.learning_rate:.2e}")
    print(f"  Step:          {metrics.step}/{metrics.total_steps}")
    if metrics.total_epochs > 1:
        print(f"  Epoch:         {metrics.epoch}/{metrics.total_epochs}")
    print(f"  Batch Size:    {metrics.batch_size}")
    print(f"  Progress:      {metrics.progress_percentage:.1f}%")
    print(f"  Elapsed:       {metrics.elapsed_formatted}")
    if metrics.eta_seconds is not None:
        print(f"  ETA:           {metrics.eta_formatted}")
    if metrics.gpu_memory_total_gb > 0:
        print(
            f"  GPU Memory:    {metrics.gpu_memory_used_gb:.1f}/{metrics.gpu_memory_total_gb:.1f} GB ({metrics.gpu_memory_percentage:.1f}%)"
        )
    print("─" * 60 + "\n")


def run_dry_run(args: argparse.Namespace) -> int:
    """Run dry run tests."""
    print_banner()
    print("🧪 Running Dry Run Test Suite...\n")

    # Import and run dry run suite
    from scripts.monitoring.modules.dry_run_suite import DryRunSuite

    suite = DryRunSuite(verbose=not args.quiet)
    result = suite.run_all()

    return 0 if result.failed == 0 else 1


def run_monitor(args: argparse.Namespace) -> int:
    """Run the training monitor."""
    print_banner()

    logger = setup_logging(args.log_level)
    logger.info("Starting Training Monitor")

    # Create monitor
    monitor = create_monitor(
        config_path=args.config,
        line_access_token=args.line_token,
        dry_run=args.dry_run,
    )

    # Setup callbacks for console output
    def on_metrics_update(metrics: TrainingMetrics) -> None:
        print_metrics_summary(metrics, metrics.phase_name)

    def on_phase_start(phase_name: str) -> None:
        print(f"\n==> Starting Phase: {phase_name}")

    def on_phase_complete(phase_name: str, metrics: TrainingMetrics) -> None:
        print(f"\n[OK] Completed Phase: {phase_name}")

    def on_error(phase_name: str, error: Exception) -> None:
        print(f"\n[ERROR] Error in {phase_name}: {error}")

    monitor.callbacks = MonitorCallbacks(
        on_phase_start=on_phase_start,
        on_phase_complete=on_phase_complete,
        on_error=on_error,
        on_metrics_update=on_metrics_update,
    )

    # Run pipeline
    success = monitor.run_full_pipeline(skip_training=args.skip_training)

    if success:
        print("\n[SUCCESS] Training completed successfully!")
    else:
        print("\n[FAILED] Training failed or was interrupted.")

    return 0 if success else 1


def show_status(args: argparse.Namespace) -> int:
    """Show current monitor status."""
    print_banner()

    monitor = create_monitor()
    status = monitor.get_status()

    print("\n📊 Monitor Status")
    print("─" * 40)
    print(f"  Run ID:        {status['run_id']}")
    print(f"  Running:       {status['running']}")
    print(f"  Current Phase: {status['current_phase'] or 'None'}")
    print(f"  Phases Done:   {status['phases_completed']}/{status['total_phases']}")
    if status["start_time"]:
        print(f"  Start Time:    {status['start_time']}")
    print("─" * 40)

    return 0


def test_line_connection(args: argparse.Namespace) -> int:
    """Test LINE connection."""
    print_banner()

    if not args.line_token:
        print("\n❌ LINE token not provided. Use --line-token to specify.")
        return 1

    print("\n🔗 Testing LINE Connection...")

    from scripts.monitoring.modules.line_notifier import LineNotifier, MockHttpClient

    # Test with mock first
    mock_client = MockHttpClient(should_fail=False)
    notifier = LineNotifier(
        access_token=args.line_token,
        http_client=mock_client,
    )

    response = notifier.send_message("🧪 Connection test from Moonshot Monitor")

    if response.success:
        print("\n✅ LINE connection successful!")
        print(f"   Message sent: Connection test")
        print(f"   Response: {response.response_body}")
    else:
        print(f"\n❌ LINE connection failed: {response.error_message}")
        return 1

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Moonshot Pipeline v3.0 - Training Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run dry run tests
  python training_monitor.py --dry-run

  # Run full monitor
  python training_monitor.py

  # Run monitor with LINE notifications
  python training_monitor.py --line-token YOUR_TOKEN

  # Show current status
  python training_monitor.py --status

  # Test LINE connection
  python training_monitor.py --test-line --line-token YOUR_TOKEN

  # Skip training phases
  python training_monitor.py --skip-training

  # Use custom config
  python training_monitor.py --config /path/to/config.yaml
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run dry run tests (default: False)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current monitor status",
    )
    parser.add_argument(
        "--test-line",
        action="store_true",
        help="Test LINE connection",
    )
    parser.add_argument(
        "--line-token",
        type=str,
        default="",
        help="LINE channel access token",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip SFT and GRPO training phases",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output in dry run",
    )

    args = parser.parse_args()

    # Determine action
    if args.dry_run:
        return run_dry_run(args)
    elif args.status:
        return show_status(args)
    elif args.test_line:
        return test_line_connection(args)
    else:
        return run_monitor(args)


if __name__ == "__main__":
    sys.exit(main())
