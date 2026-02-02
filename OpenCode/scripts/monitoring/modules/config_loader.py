#!/usr/bin/env python3
"""
Configuration Loader Module.

Best Practices:
- Dependency Injection: Configuration is injected, not hardcoded
- YAML-based: Externalized configuration for environment-specific settings
- Error Handling: Comprehensive error handling for missing files/keys
- Type Safety: Full type hints for IDE support and documentation
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml


@dataclass
class DatabaseConfig:
    """Database configuration."""

    type: str = "sqlite"
    path: str = "logs/pipeline_progress.sqlite"
    table_prefix: str = "monitor_"


@dataclass
class LineConfig:
    """LINE Bot configuration."""

    enabled: bool = False
    gateway_url: str = "http://localhost:18789"
    channel_access_token: str = ""
    channel_secret: str = ""
    message_format: str = "detailed"
    retry_attempts: int = 3
    retry_delay_seconds: int = 1


@dataclass
class GpuConfig:
    """GPU metrics configuration."""

    device_id: int = 0
    memory_unit: str = "GB"
    threshold_warning_gb: float = 10.0
    threshold_critical_gb: float = 11.0


@dataclass
class EtaConfig:
    """ETA calculation configuration."""

    smoothing_factor: float = 0.1
    min_samples_for_eta: int = 10


@dataclass
class MonitoringConfig:
    """Core monitoring configuration."""

    checkpoint_interval: int = 300
    max_rolling_checkpoints: int = 3
    metrics_interval: int = 10
    enable_realtime: bool = True


@dataclass
class MetricsEnabled:
    """Enabled metrics configuration."""

    loss: bool = True
    learning_rate: bool = True
    step: bool = True
    eta: bool = True
    gpu_memory: bool = True
    epoch: bool = True
    batch_size: bool = True
    data_progress: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "logs/monitoring.log"
    max_size_mb: int = 100
    backup_count: int = 5


@dataclass
class DryRunConfig:
    """Dry run configuration."""

    enabled: bool = True
    num_phases: int = 8
    steps_per_phase: int = 100
    loss_range: tuple = (0.1, 2.0)
    lr_range: tuple = (1e-6, 2e-5)


@dataclass
class NotificationConfig:
    """Notification configuration."""

    phase_start: bool = True
    phase_complete: bool = True
    checkpoint_saved: bool = True
    error_occurred: bool = True
    training_complete: bool = True
    min_interval_seconds: int = 60
    max_per_hour: int = 10
    include_timestamp: bool = True
    include_run_id: bool = True
    include_emoji: bool = True


@dataclass
class MonitoringSettings:
    """Aggregated monitoring settings."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    line: LineConfig = field(default_factory=LineConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    gpu: GpuConfig = field(default_factory=GpuConfig)
    eta: EtaConfig = field(default_factory=EtaConfig)
    notifications: NotificationConfig = field(default_factory=NotificationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dry_run: DryRunConfig = field(default_factory=DryRunConfig)


class ConfigLoader:
    """
    Configuration loader with dependency injection support.

    Best Practices:
    - Singleton pattern for consistent config access
    - Environment variable substitution
    - Validation on load
    - Default fallback for missing values
    """

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[MonitoringSettings] = None

    def __new__(cls, config_path: Optional[str] = None) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: Optional[str] = None) -> None:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default to project root config
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "config" / "monitoring_config.yaml")

        config_file = Path(config_path)

        if not config_file.exists():
            self._config = MonitoringSettings()
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}

            self._config = self._parse_config(raw_config)

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    def _parse_config(self, raw: Dict[str, Any]) -> MonitoringSettings:
        """Parse raw YAML config into typed dataclasses."""
        settings = MonitoringSettings()

        if "database" in raw:
            db_raw = raw["database"]
            settings.database = DatabaseConfig(
                type=db_raw.get("type", "sqlite"),
                path=db_raw.get("path", "logs/pipeline_progress.sqlite"),
                table_prefix=db_raw.get("table_prefix", "monitor_"),
            )

        if "line" in raw:
            line_raw = raw["line"]
            settings.line = LineConfig(
                enabled=line_raw.get("enabled", False),
                gateway_url=line_raw.get("gateway_url", "http://localhost:18789"),
                channel_access_token=self._substitute_env(
                    line_raw.get("channel_access_token", "")
                ),
                channel_secret=self._substitute_env(line_raw.get("channel_secret", "")),
                message_format=line_raw.get("message_format", "detailed"),
                retry_attempts=line_raw.get("retry_attempts", 3),
                retry_delay_seconds=line_raw.get("retry_delay_seconds", 1),
            )

        if "monitoring" in raw:
            mon_raw = raw["monitoring"]
            settings.monitoring = MonitoringConfig(
                checkpoint_interval=mon_raw.get("checkpoint_interval", 300),
                max_rolling_checkpoints=mon_raw.get("max_rolling_checkpoints", 3),
                metrics_interval=mon_raw.get("metrics_interval", 10),
                enable_realtime=mon_raw.get("enable_realtime", True),
            )

        if "metrics" in raw and "gpu" in raw["metrics"]:
            gpu_raw = raw["metrics"]["gpu"]
            settings.gpu = GpuConfig(
                device_id=gpu_raw.get("device_id", 0),
                memory_unit=gpu_raw.get("memory_unit", "GB"),
                threshold_warning_gb=gpu_raw.get("threshold_warning_gb", 10.0),
                threshold_critical_gb=gpu_raw.get("threshold_critical_gb", 11.0),
            )

        if "metrics" in raw and "eta" in raw["metrics"]:
            eta_raw = raw["metrics"]["eta"]
            settings.eta = EtaConfig(
                smoothing_factor=eta_raw.get("smoothing_factor", 0.1),
                min_samples_for_eta=eta_raw.get("min_samples_for_eta", 10),
            )

        if "notifications" in raw:
            notif_raw = raw["notifications"]
            settings.notifications = NotificationConfig(
                phase_start=notif_raw.get("phase_start", True),
                phase_complete=notif_raw.get("phase_complete", True),
                checkpoint_saved=notif_raw.get("checkpoint_saved", True),
                error_occurred=notif_raw.get("error_occurred", True),
                training_complete=notif_raw.get("training_complete", True),
                min_interval_seconds=notif_raw.get("min_interval_seconds", 60),
                max_per_hour=notif_raw.get("max_per_hour", 10),
                include_timestamp=notif_raw.get("include_timestamp", True),
                include_run_id=notif_raw.get("include_run_id", True),
                include_emoji=notif_raw.get("include_emoji", True),
            )

        if "logging" in raw:
            log_raw = raw["logging"]
            settings.logging = LoggingConfig(
                level=log_raw.get("level", "INFO"),
                format=log_raw.get(
                    "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                ),
                file_enabled=log_raw.get("file", {}).get("enabled", True),
                file_path=log_raw.get("file", {}).get("path", "logs/monitoring.log"),
                max_size_mb=log_raw.get("file", {}).get("max_size_mb", 100),
                backup_count=log_raw.get("file", {}).get("backup_count", 5),
            )

        if "dry_run" in raw:
            dry_raw = raw["dry_run"]
            settings.dry_run = DryRunConfig(
                enabled=dry_raw.get("enabled", True),
                num_phases=dry_raw.get("num_phases", 8),
                steps_per_phase=dry_raw.get("steps_per_phase", 100),
                loss_range=tuple(dry_raw.get("loss_range", [0.1, 2.0])),
                lr_range=tuple(dry_raw.get("lr_range", [1e-6, 2e-5])),
            )

        return settings

    def _substitute_env(self, value: str) -> str:
        """Substitute environment variables in config values."""
        if not isinstance(value, str):
            return value

        # Pattern: ${ENV_VAR}
        pattern = r"\$\{(\w+)\}"

        def replace(match):
            env_var = match.group(1)
            return os.environ.get(env_var, match.group(0))

        return re.sub(pattern, replace, value)

    @property
    def config(self) -> MonitoringSettings:
        """Get loaded configuration."""
        if self._config is None:
            self._load_config()
        return self._config

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        cls._instance = None
        cls._config = None


def get_config(config_path: Optional[str] = None) -> MonitoringSettings:
    """Get monitoring configuration."""
    loader = ConfigLoader(config_path)
    return loader.config
