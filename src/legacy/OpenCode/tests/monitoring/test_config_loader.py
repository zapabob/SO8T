#!/usr/bin/env python3
"""
Unit Tests for Config Loader Module.

Best Practices:
- Pytest-based unit tests
- Isolated test cases with fixtures
- Comprehensive assertions
- Mock objects for external dependencies
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestConfigLoader:
    """Test cases for ConfigLoader."""

    def setup_method(self):
        """Reset config loader before each test."""
        from scripts.monitoring.modules.config_loader import ConfigLoader

        ConfigLoader.reset()

    def teardown_method(self):
        """Clean up after each test."""
        from scripts.monitoring.modules.config_loader import ConfigLoader

        ConfigLoader.reset()

    def test_get_config_returns_monitoring_settings(self):
        """Test that get_config returns MonitoringSettings."""
        from scripts.monitoring.modules.config_loader import get_config

        config = get_config()

        assert config is not None
        assert hasattr(config, "database")
        assert hasattr(config, "line")
        assert hasattr(config, "monitoring")

    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        from scripts.monitoring.modules.config_loader import get_config

        config = get_config()

        assert config.database.type == "sqlite"
        assert config.database.path == "logs/pipeline_progress.sqlite"
        assert config.database.table_prefix == "monitor_"

    def test_line_config_defaults(self):
        """Test LINE configuration defaults."""
        from scripts.monitoring.modules.config_loader import get_config

        config = get_config()

        assert config.line.enabled is False
        assert config.line.gateway_url == "http://localhost:18789"
        assert config.line.message_format == "detailed"
        assert config.line.retry_attempts == 3

    def test_monitoring_config_defaults(self):
        """Test monitoring configuration defaults."""
        from scripts.monitoring.modules.config_loader import get_config

        config = get_config()

        assert config.monitoring.checkpoint_interval == 300
        assert config.monitoring.max_rolling_checkpoints == 3
        assert config.monitoring.metrics_interval == 10
        assert config.monitoring.enable_realtime is True

    def test_gpu_config_defaults(self):
        """Test GPU configuration defaults."""
        from scripts.monitoring.modules.config_loader import get_config

        config = get_config()

        assert config.gpu.device_id == 0
        assert config.gpu.memory_unit == "GB"
        assert config.gpu.threshold_warning_gb == 10.0
        assert config.gpu.threshold_critical_gb == 11.0

    def test_eta_config_defaults(self):
        """Test ETA configuration defaults."""
        from scripts.monitoring.modules.config_loader import get_config

        config = get_config()

        assert config.eta.smoothing_factor == 0.1
        assert config.eta.min_samples_for_eta == 10

    def test_environment_variable_substitution(self):
        """Test environment variable substitution in config."""
        from scripts.monitoring.modules.config_loader import ConfigLoader

        # Set test environment variable
        os.environ["TEST_LINE_TOKEN"] = "test_token_12345"

        # Create temporary config file
        config_content = """
line:
  channel_access_token: "${TEST_LINE_TOKEN}"
  enabled: true
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_content)
            config_path = f.name

        try:
            config = ConfigLoader(config_path).config

            assert config.line.channel_access_token == "test_token_12345"
            assert config.line.enabled is True
        finally:
            os.unlink(config_path)
            del os.environ["TEST_LINE_TOKEN"]

    def test_invalid_config_path_uses_defaults(self):
        """Test that invalid config path uses defaults."""
        from scripts.monitoring.modules.config_loader import get_config

        # This should use defaults
        config = get_config("/nonexistent/path/config.yaml")

        assert config is not None
        assert config.database.type == "sqlite"

    def test_singleton_pattern(self):
        """Test that ConfigLoader uses singleton pattern."""
        from scripts.monitoring.modules.config_loader import ConfigLoader

        loader1 = ConfigLoader()
        loader2 = ConfigLoader()

        assert loader1 is loader2


class TestConfigDataclasses:
    """Test cases for configuration dataclasses."""

    def test_database_config_creation(self):
        """Test DatabaseConfig creation."""
        from scripts.monitoring.modules.config_loader import DatabaseConfig

        config = DatabaseConfig(
            type="sqlite",
            path="test.db",
            table_prefix="test_",
        )

        assert config.type == "sqlite"
        assert config.path == "test.db"
        assert config.table_prefix == "test_"

    def test_line_config_creation(self):
        """Test LineConfig creation."""
        from scripts.monitoring.modules.config_loader import LineConfig

        config = LineConfig(
            enabled=True,
            gateway_url="http://localhost:3000",
            channel_access_token="token123",
            channel_secret="secret456",
            message_format="simple",
            retry_attempts=5,
            retry_delay_seconds=2,
        )

        assert config.enabled is True
        assert config.gateway_url == "http://localhost:3000"
        assert config.retry_attempts == 5

    def test_monitoring_config_creation(self):
        """Test MonitoringConfig creation."""
        from scripts.monitoring.modules.config_loader import MonitoringConfig

        config = MonitoringConfig(
            checkpoint_interval=600,
            max_rolling_checkpoints=5,
            metrics_interval=15,
            enable_realtime=False,
        )

        assert config.checkpoint_interval == 600
        assert config.max_rolling_checkpoints == 5

    def test_gpu_config_creation(self):
        """Test GpuConfig creation."""
        from scripts.monitoring.modules.config_loader import GpuConfig

        config = GpuConfig(
            device_id=1,
            memory_unit="MB",
            threshold_warning_gb=8192.0,
            threshold_critical_gb=10240.0,
        )

        assert config.device_id == 1
        assert config.memory_unit == "MB"

    def test_eta_config_creation(self):
        """Test EtaConfig creation."""
        from scripts.monitoring.modules.config_loader import EtaConfig

        config = EtaConfig(
            smoothing_factor=0.2,
            min_samples_for_eta=15,
        )

        assert config.smoothing_factor == 0.2
        assert config.min_samples_for_eta == 15

    def test_dry_run_config_creation(self):
        """Test DryRunConfig creation."""
        from scripts.monitoring.modules.config_loader import DryRunConfig

        config = DryRunConfig(
            enabled=True,
            num_phases=10,
            steps_per_phase=200,
            loss_range=(0.05, 1.5),
            lr_range=(1e-6, 1e-4),
        )

        assert config.enabled is True
        assert config.num_phases == 10
        assert config.loss_range == (0.05, 1.5)
