import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Add monitoring modules path
MONITORING_MODULES = ROOT / "scripts" / "monitoring" / "modules"
if str(MONITORING_MODULES) not in sys.path:
    sys.path.insert(0, str(MONITORING_MODULES))

import pytest


@pytest.fixture(scope="session")
def project_root():
    """Get project root path."""
    return ROOT


@pytest.fixture(scope="session")
def config_path(project_root):
    """Get config file path."""
    return project_root / "config" / "monitoring_config.yaml"


@pytest.fixture(scope="function")
def reset_config_loader():
    """Reset config loader before each test."""
    from scripts.monitoring.modules.config_loader import ConfigLoader

    ConfigLoader.reset()
    yield
    ConfigLoader.reset()


@pytest.fixture
def sample_training_metrics():
    """Create sample training metrics."""
    from datetime import datetime

    from scripts.monitoring.modules.metrics_collector import TrainingMetrics

    return TrainingMetrics(
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
        eta_seconds=754,
        elapsed_seconds=323,
        phase_name="sft",
        timestamp=datetime(2026, 2, 2, 12, 30, 45),
    )


@pytest.fixture
def mock_line_token():
    """Mock LINE token for testing."""
    return "test_line_token_12345"


@pytest.fixture
def mock_gpu_monitor():
    """Mock GPU monitor for testing."""
    from scripts.monitoring.modules.metrics_collector import SimulatedGPUMonitor

    return SimulatedGPUMonitor(used_gb=8.5, total_gb=12.0)


@pytest.fixture
def training_monitor_factory():
    """Factory for creating training monitors."""
    from scripts.monitoring.modules.training_monitor_core import TrainingMonitor

    def create(dry_run: bool = True, run_id: str = "test_run"):
        return TrainingMonitor(
            config=None,
            line_notifier=None,
            dry_run=dry_run,
            run_id=run_id,
        )

    return create
