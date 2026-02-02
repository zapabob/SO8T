#!/usr/bin/env python3
"""
Monitoring Modules Package.

Best Practices:
- Explicit imports for public API
- Version information
- Comprehensive module documentation
"""

from scripts.monitoring.modules.config_loader import (
    ConfigLoader,
    DatabaseConfig,
    DryRunConfig,
    EtaConfig,
    GpuConfig,
    LineConfig,
    LoggingConfig,
    MonitoringConfig,
    MonitoringSettings,
    NotificationConfig,
    get_config,
)
from scripts.monitoring.modules.line_notifier import (
    LineAPIError,
    LineMessage,
    LineNotifier,
    LineResponse,
    MockHttpClient,
)
from scripts.monitoring.modules.metrics_collector import (
    ETACalculator,
    GPUMonitor,
    MetricsCollector,
    SimulatedGPUMonitor,
    TrainingMetrics,
)
from scripts.monitoring.modules.training_monitor_core import (
    MonitorCallbacks,
    PhaseInfo,
    TrainingMonitor,
    create_monitor,
)

__version__ = "3.0.0"
__author__ = "Moonshot AI"

__all__ = [
    # Config
    "ConfigLoader",
    "get_config",
    "MonitoringSettings",
    "DatabaseConfig",
    "LineConfig",
    "MonitoringConfig",
    "GpuConfig",
    "EtaConfig",
    "NotificationConfig",
    "LoggingConfig",
    "DryRunConfig",
    # Metrics
    "TrainingMetrics",
    "MetricsCollector",
    "ETACalculator",
    "GPUMonitor",
    "SimulatedGPUMonitor",
    # LINE
    "LineNotifier",
    "LineMessage",
    "LineResponse",
    "LineAPIError",
    "MockHttpClient",
    # Training Monitor
    "TrainingMonitor",
    "create_monitor",
    "MonitorCallbacks",
    "PhaseInfo",
]
