#!/usr/bin/env python3
"""
Dry Run Test Suite.

Best Practices:
- Comprehensive test coverage for all monitoring components
- Isolated test cases with clear pass/fail criteria
- Mock objects for dependency injection
- Detailed reporting with metrics
- Self-contained and reproducible
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "monitoring"))

# Direct imports
from scripts.monitoring.modules import config_loader
from scripts.monitoring.modules import line_notifier
from scripts.monitoring.modules import metrics_collector
from scripts.monitoring.modules import training_monitor_core


@dataclass
class TestResult:
    """Result of a test case."""

    name: str
    passed: bool
    duration_ms: float
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteResult:
    """Result of a test suite."""

    suite_name: str
    total_tests: int
    passed: int
    failed: int
    duration_ms: float
    results: List[TestResult]
    timestamp: datetime = field(default_factory=datetime.now)


class TestReporter:
    """Test result reporter."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[TestSuiteResult] = []

    def print_result(self, result: TestSuiteResult) -> None:
        """Print test suite result."""
        print("\n" + "=" * 70)
        print(f"[TEST] Test Suite: {result.suite_name}")
        print("=" * 70)

        for test in result.results:
            status = "[PASS]" if test.passed else "[FAIL]"
            print(f"  {status} [{test.duration_ms:.1f}ms] {test.name}")
            if not test.passed:
                print(f"         -> {test.message}")
                if test.details:
                    for k, v in test.details.items():
                        print(f"            {k}: {v}")

        print("-" * 70)
        print(
            f"  Total: {result.total_tests} | Passed: {result.passed} | Failed: {result.failed}"
        )
        print(f"  Duration: {result.duration_ms:.1f}ms")
        print("=" * 70 + "\n")

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "suites": [
                {
                    "name": s.suite_name,
                    "total_tests": s.total_tests,
                    "passed": s.passed,
                    "failed": s.failed,
                    "duration_ms": s.duration_ms,
                    "results": [
                        {
                            "name": r.name,
                            "passed": r.passed,
                            "duration_ms": r.duration_ms,
                            "message": r.message,
                            "details": r.details,
                        }
                        for r in s.results
                    ],
                }
                for s in self.results
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)


class ConfigTests:
    """Test cases for configuration module."""

    @staticmethod
    def test_default_config() -> TestResult:
        """Test default configuration loading."""
        start = time.perf_counter()
        try:
            config_loader.ConfigLoader.reset()
            config = config_loader.get_config()

            assert config is not None
            assert hasattr(config, "database")
            assert hasattr(config, "line")
            assert hasattr(config, "monitoring")

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_default_config",
                passed=True,
                duration_ms=duration,
                message="Default configuration loaded successfully",
                details={
                    "database_type": config.database.type,
                    "line_enabled": config.line.enabled,
                    "checkpoint_interval": config.monitoring.checkpoint_interval,
                },
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_default_config",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_config_environment_substitution() -> TestResult:
        """Test environment variable substitution."""
        start = time.perf_counter()
        try:
            config_loader.ConfigLoader.reset()

            # Set environment variable
            import os

            os.environ["TEST_API_KEY"] = "test_key_12345"

            # Create minimal config with env var
            config_content = """
line:
  channel_access_token: "${TEST_API_KEY}"
"""
            config_path = Path(__file__).parent / "test_config_temp.yaml"
            config_path.write_text(config_content)

            config = config_loader.get_config(str(config_path))

            # Clean up
            config_path.unlink(missing_ok=True)
            del os.environ["TEST_API_KEY"]

            assert config.line.channel_access_token == "test_key_12345"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_config_environment_substitution",
                passed=True,
                duration_ms=duration,
                message="Environment variable substitution works",
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_config_environment_substitution",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )


class MetricsTests:
    """Test cases for metrics collection."""

    @staticmethod
    def test_metrics_creation() -> TestResult:
        """Test TrainingMetrics creation."""
        start = time.perf_counter()
        try:
            metrics = metrics_collector.TrainingMetrics(
                run_id="test_run",
                step=100,
                total_steps=500,
                loss=0.0234,
                learning_rate=2e-5,
                epoch=1,
                total_epochs=3,
                batch_size=2,
                gpu_memory_used_gb=8.2,
                gpu_memory_total_gb=12.0,
                eta_seconds=3600,
                elapsed_seconds=300,
            )

            assert metrics.step == 100
            assert metrics.loss == 0.0234
            assert metrics.eta_formatted == "1h 0m"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_metrics_creation",
                passed=True,
                duration_ms=duration,
                message="TrainingMetrics created successfully",
                details={
                    "loss": metrics.loss,
                    "eta": metrics.eta_formatted,
                    "progress": metrics.progress_percentage,
                },
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_metrics_creation",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_metrics_eta_formats() -> TestResult:
        """Test ETA formatting."""
        start = time.perf_counter()
        try:
            metrics = metrics_collector.TrainingMetrics()

            # Test various ETA values
            metrics.eta_seconds = 65
            assert metrics.eta_formatted == "1m 5s", (
                f"Expected '1m 5s' but got '{metrics.eta_formatted}'"
            )

            metrics.eta_seconds = 3665  # 1h 1m 5s
            assert metrics.eta_formatted == "1h 1m", (
                f"Expected '1h 1m' but got '{metrics.eta_formatted}'"
            )

            metrics.eta_seconds = 90065  # 1d 1h 1m 5s
            assert metrics.eta_formatted == "1d 1h", (
                f"Expected '1d 1h' but got '{metrics.eta_formatted}'"
            )

            metrics.eta_seconds = None
            assert metrics.eta_formatted == "N/A"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_metrics_eta_formats",
                passed=True,
                duration_ms=duration,
                message="ETA formatting works correctly",
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_metrics_eta_formats",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_eta_calculator() -> TestResult:
        """Test ETA calculation with smoothing."""
        start = time.perf_counter()
        try:
            calculator = metrics_collector.ETACalculator(
                smoothing_factor=0.1,
                min_samples=5,
            )

            # Simulate steps with varying times
            for step in range(1, 20):
                time.sleep(0.001)  # Tiny delay
                calculator.update(step, step * 0.1)

            eta = calculator.calculate_eta(10, 100)
            assert eta is not None, "ETA should be calculated"

            calculator.reset()
            assert calculator._ewma_step_time is None, "Reset should clear state"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_eta_calculator",
                passed=True,
                duration_ms=duration,
                message="ETA calculator works correctly",
                details={"eta_seconds": eta},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_eta_calculator",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_metrics_collector() -> TestResult:
        """Test MetricsCollector functionality."""
        start = time.perf_counter()
        try:
            collector = metrics_collector.MetricsCollector(
                run_id="test_run",
                gpu_monitor=metrics_collector.SimulatedGPUMonitor(
                    used_gb=8.5, total_gb=12.0
                ),
            )

            collector.start(total_steps=100, total_epochs=3)

            for step in range(1, 51):
                collector.update(
                    step=step,
                    loss=1.0 / step,
                    learning_rate=2e-5 * (1 - step / 100),
                    epoch=(step // 34) + 1,
                    batch_size=2,
                    data_progress=step / 100,
                )

            metrics = collector.get_current_metrics()

            assert metrics.step == 50
            assert metrics.loss is not None
            assert metrics.gpu_memory_used_gb > 0
            assert metrics.epoch == 2

            collector.complete()

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_metrics_collector",
                passed=True,
                duration_ms=duration,
                message="MetricsCollector works correctly",
                details={
                    "step": metrics.step,
                    "loss": metrics.loss,
                    "gpu_memory": f"{metrics.gpu_memory_used_gb:.1f}GB",
                },
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_metrics_collector",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )


class LineNotifierTests:
    """Test cases for LINE notifier."""

    @staticmethod
    def test_line_message_formatting() -> TestResult:
        """Test detailed message formatting."""
        start = time.perf_counter()
        try:
            notifier = line_notifier.LineNotifier(
                access_token="test_token",
                message_format="detailed",
            )

            metrics = metrics_collector.TrainingMetrics(
                run_id="test_run_12345678",
                step=150,
                total_steps=500,
                loss=0.0234,
                learning_rate=2e-5,
                epoch=1,
                total_epochs=3,
                batch_size=2,
                gpu_memory_used_gb=8.2,
                gpu_memory_total_gb=12.0,
                eta_seconds=754,
                elapsed_seconds=323,
                phase_name="sft",
                timestamp=datetime.now(),
            )

            message = notifier._format_detailed_message(metrics, "SFT Training")

            # Verify message contains key elements
            assert "SFT Training" in message, "Phase name should be in message"
            assert "Loss: 0.0234" in message, "Loss should be in message"
            assert "ETA: 12m 34s" in message, "ETA should be formatted"
            assert "GPU Memory: 8.2/12.0 GB" in message, (
                "GPU memory should be in message"
            )
            assert "━━━━━━━━━━━━━━━━━━━━━━━━" in message, "Divider should be present"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_line_message_formatting",
                passed=True,
                duration_ms=duration,
                message="LINE message formatting works correctly",
                details={"message_length": len(message)},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_line_message_formatting",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_line_mock_client() -> TestResult:
        """Test LINE notifier with mock HTTP client."""
        start = time.perf_counter()
        try:
            mock_client = line_notifier.MockHttpClient(should_fail=False)

            notifier = line_notifier.LineNotifier(
                access_token="test_token",
                http_client=mock_client,
            )

            response = notifier.send_message("Test message")

            assert response.success, "Response should be successful"
            assert response.status_code == 200, "Status code should be 200"
            assert len(mock_client.requests) == 1, "One request should be made"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_line_mock_client",
                passed=True,
                duration_ms=duration,
                message="Mock HTTP client works correctly",
                details={"requests_count": len(mock_client.requests)},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_line_mock_client",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_line_retry_logic() -> TestResult:
        """Test LINE notifier retry logic."""
        start = time.perf_counter()
        try:
            # Test with client that fails once then succeeds
            class FailOnceClient:
                def __init__(self):
                    self.call_count = 0

                def post(self, url, data, headers):
                    self.call_count += 1
                    if self.call_count == 1:
                        from urllib.error import HTTPError

                        raise HTTPError(url, 500, "Server Error", {}, None)
                    return 200, b'{"status":"ok"}'

            fail_client = FailOnceClient()

            notifier = line_notifier.LineNotifier(
                access_token="test_token",
                http_client=fail_client,
                retry_attempts=3,
                retry_delay=0.01,
            )

            response = notifier.send_message("Test message")

            assert response.success, "Response should succeed after retry"
            assert fail_client.call_count == 2, (
                "Should make 2 calls (1 fail + 1 success)"
            )

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_line_retry_logic",
                passed=True,
                duration_ms=duration,
                message="Retry logic works correctly",
                details={"retry_count": fail_client.call_count - 1},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_line_retry_logic",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )


class TrainingMonitorTests:
    """Test cases for TrainingMonitor."""

    @staticmethod
    def test_monitor_creation() -> TestResult:
        """Test TrainingMonitor creation."""
        start = time.perf_counter()
        try:
            monitor = training_monitor_core.create_monitor(dry_run=True)

            assert monitor is not None
            assert monitor.run_id.startswith("moonshot_")
            assert len(monitor._phases) == 8, "Should have 8 phases"

            status = monitor.get_status()
            assert status["total_phases"] == 8
            assert status["running"] == False

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_monitor_creation",
                passed=True,
                duration_ms=duration,
                message="TrainingMonitor created successfully",
                details={"run_id": monitor.run_id[:20] + "..."},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_monitor_creation",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_phase_lifecycle() -> TestResult:
        """Test phase start/end lifecycle."""
        start = time.perf_counter()
        try:
            monitor = training_monitor_core.create_monitor(dry_run=True)

            # Start phase
            assert monitor.start_phase("sft", total_steps=100)
            assert monitor._current_phase == "sft"
            assert monitor._phases["sft"].status == "running"

            # Update metrics
            for step in range(1, 51):
                monitor.update_metrics(
                    step=step,
                    loss=1.0 / step,
                    learning_rate=2e-5,
                )

            metrics = monitor.get_current_metrics()
            assert metrics is not None
            assert metrics.step == 50

            # End phase
            final_metrics = monitor.end_phase("sft")
            assert monitor._phases["sft"].status == "complete"
            assert final_metrics is not None
            assert final_metrics.step == 50

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_phase_lifecycle",
                passed=True,
                duration_ms=duration,
                message="Phase lifecycle works correctly",
                details={"final_step": final_metrics.step if final_metrics else None},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_phase_lifecycle",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_callbacks() -> TestResult:
        """Test monitoring callbacks."""
        start = time.perf_counter()
        try:
            callbacks_received = []

            def on_phase_start(phase_name: str):
                callbacks_received.append(("phase_start", phase_name))

            def on_phase_complete(phase_name: str, metrics):
                callbacks_received.append(("phase_complete", phase_name, metrics.step))

            def on_error(phase_name: str, error: Exception):
                callbacks_received.append(("error", phase_name, str(error)))

            monitor_callbacks = training_monitor_core.MonitorCallbacks(
                on_phase_start=on_phase_start,
                on_phase_complete=on_phase_complete,
                on_error=on_error,
            )

            monitor = training_monitor_core.create_monitor(
                dry_run=True,
            )
            monitor.callbacks = monitor_callbacks

            monitor.start_phase("data", total_steps=10)
            monitor.end_phase("data")

            assert len(callbacks_received) == 2
            assert callbacks_received[0][0] == "phase_start"
            assert callbacks_received[1][0] == "phase_complete"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_callbacks",
                passed=True,
                duration_ms=duration,
                message="Callbacks work correctly",
                details={"callbacks_received": len(callbacks_received)},
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_callbacks",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )

    @staticmethod
    def test_dry_run_pipeline() -> TestResult:
        """Test full dry run pipeline."""
        start = time.perf_counter()
        try:
            monitor = training_monitor_core.create_monitor(dry_run=True)

            # Track metrics updates
            metrics_updates = []

            def on_metrics_update(metrics):
                metrics_updates.append(metrics.step)

            monitor.callbacks.on_metrics_update = on_metrics_update

            # Run pipeline with training enabled to get metrics updates
            success = monitor.run_full_pipeline(skip_training=False)

            assert success, "Pipeline should complete successfully"
            assert len(metrics_updates) > 0, "Should receive metrics updates"

            summary = monitor.get_phase_summary()
            assert summary["phases"]["setup"]["status"] == "complete"
            assert summary["phases"]["data"]["status"] == "complete"
            assert summary["phases"]["benchmark"]["status"] == "complete"

            duration = (time.perf_counter() - start) * 1000

            return TestResult(
                name="test_dry_run_pipeline",
                passed=True,
                duration_ms=duration,
                message="Dry run pipeline works correctly",
                details={
                    "phases_completed": summary["phases"]["setup"]["status"],
                    "metrics_updates": len(metrics_updates),
                },
            )
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            return TestResult(
                name="test_dry_run_pipeline",
                passed=False,
                duration_ms=duration,
                message=str(e),
            )


class DryRunSuite:
    """Complete dry run test suite."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.reporter = TestReporter(verbose)

    def run_all(self) -> TestSuiteResult:
        """Run all test suites."""
        all_results = []

        # Config tests
        config_results = self._run_suite(
            "Config",
            [
                ConfigTests.test_default_config,
                ConfigTests.test_config_environment_substitution,
            ],
        )
        all_results.append(config_results)

        # Metrics tests
        metrics_results = self._run_suite(
            "Metrics",
            [
                MetricsTests.test_metrics_creation,
                MetricsTests.test_metrics_eta_formats,
                MetricsTests.test_eta_calculator,
                MetricsTests.test_metrics_collector,
            ],
        )
        all_results.append(metrics_results)

        # LINE notifier tests
        line_results = self._run_suite(
            "LINE Notifier",
            [
                LineNotifierTests.test_line_message_formatting,
                LineNotifierTests.test_line_mock_client,
                LineNotifierTests.test_line_retry_logic,
            ],
        )
        all_results.append(line_results)

        # Training monitor tests
        monitor_results = self._run_suite(
            "Training Monitor",
            [
                TrainingMonitorTests.test_monitor_creation,
                TrainingMonitorTests.test_phase_lifecycle,
                TrainingMonitorTests.test_callbacks,
                TrainingMonitorTests.test_dry_run_pipeline,
            ],
        )
        all_results.append(monitor_results)

        # Aggregate results
        total_tests = sum(r.total_tests for r in all_results)
        passed = sum(r.passed for r in all_results)
        failed = sum(r.failed for r in all_results)
        total_duration = sum(r.duration_ms for r in all_results)

        # Print summary
        print("\n" + "=" * 70)
        print("* DRY RUN TEST SUITE - FINAL SUMMARY *")
        print("=" * 70)
        print(f"  Total Suites: {len(all_results)}")
        print(f"  Total Tests:  {total_tests}")
        print(f"  [PASS] Passed:    {passed}")
        print(f"  [FAIL] Failed:    {failed}")
        print(f"  Duration:  {total_duration:.1f}ms")
        print("=" * 70)

        if failed > 0:
            print("\n[!] Some tests failed. Check individual suite results above.")
        else:
            print("\n[*] All tests passed!")

        return TestSuiteResult(
            suite_name="Complete Dry Run",
            total_tests=total_tests,
            passed=passed,
            failed=failed,
            duration_ms=total_duration,
            results=[r for suite in all_results for r in suite.results],
        )

    def _run_suite(
        self, name: str, test_funcs: List[Callable[[], TestResult]]
    ) -> TestSuiteResult:
        """Run a test suite."""
        results = []
        start = time.perf_counter()

        for test_func in test_funcs:
            result = test_func()
            results.append(result)

        duration = (time.perf_counter() - start) * 1000

        suite_result = TestSuiteResult(
            suite_name=name,
            total_tests=len(results),
            passed=sum(1 for r in results if r.passed),
            failed=sum(1 for r in results if not r.passed),
            duration_ms=duration,
            results=results,
        )

        self.reporter.print_result(suite_result)
        return suite_result


def main():
    """Main entry point for dry run tests."""
    print("\n" + "=" * 70)
    print("   Moonshot Pipeline v3.0 - Dry Run Test Suite")
    print("   Testing all monitoring components for correctness")
    print("=" * 70 + "\n")

    # Configure logging
    logging.basicConfig(level=logging.WARNING)

    # Reset config loader
    config_loader.ConfigLoader.reset()

    # Run suite
    suite = DryRunSuite(verbose=True)
    result = suite.run_all()

    # Exit with appropriate code
    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
