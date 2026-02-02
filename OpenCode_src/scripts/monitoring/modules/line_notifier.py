#!/usr/bin/env python3
"""
LINE Notifier Module.

Best Practices:
- Dependency injection for HTTP client
- Retry logic with exponential backoff
- Detailed message formatting with emojis
- Rich error handling
- Type-safe implementation
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class LineMessage:
    """LINE message container."""

    text: str
    message_type: str = "text"


@dataclass
class LineResponse:
    """LINE API response."""

    success: bool
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None


@runtime_checkable
class LineClientProtocol(Protocol):
    """Protocol for LINE API client."""

    def send_message(self, message: LineMessage) -> LineResponse: ...

    def broadcast(self, messages: List[LineMessage]) -> LineResponse: ...


class LineAPIError(Exception):
    """Custom exception for LINE API errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class HttpClient(ABC):
    """Abstract HTTP client for LINE API."""

    @abstractmethod
    def post(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> tuple[int, bytes]: ...


class UrllibHttpClient(HttpClient):
    """urllib-based HTTP client implementation."""

    def post(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> tuple[int, bytes]:
        """Execute POST request."""
        json_data = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"

        request = Request(url, data=json_data, headers=headers)
        with urlopen(request, timeout=30) as response:
            return response.status, response.read()


class MockHttpClient(HttpClient):
    """Mock HTTP client for testing."""

    def __init__(self, should_fail: bool = False, fail_status: int = 500):
        self.should_fail = should_fail
        self.fail_status = fail_status
        self._requests: List[Dict[str, Any]] = []

    def post(
        self,
        url: str,
        data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> tuple[int, bytes]:
        """Record request and return mock response."""
        self._requests.append({"url": url, "data": data, "headers": headers})

        if self.should_fail:
            raise HTTPError(url, self.fail_status, "Mock error", {}, None)

        return 200, b'{"status":"ok"}'

    @property
    def requests(self) -> List[Dict[str, Any]]:
        """Get recorded requests."""
        return self._requests.copy()


class LineNotifier:
    """
    LINE Bot notifier with detailed message formatting.

    Features:
    - Detailed message format with emojis
    - Retry logic with exponential backoff
    - Message throttling
    - Rich error handling
    """

    BASE_URL = "https://api.line.me/v2/bot/message"

    def __init__(
        self,
        access_token: str,
        channel_secret: str = "",
        http_client: Optional[HttpClient] = None,
        gateway_url: str = "http://localhost:18789",
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
        message_format: str = "detailed",
    ):
        self.access_token = access_token
        self.channel_secret = channel_secret
        self.http_client = http_client or UrllibHttpClient()
        self.gateway_url = gateway_url
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.message_format = message_format

        self._logger = logging.getLogger(__name__)
        self._last_notification_time: Optional[float] = None
        self._notification_count = 0

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for LINE API."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }

    def _should_throttle(self, min_interval: float = 60.0) -> bool:
        """Check if notification should be throttled."""
        if self._last_notification_time is None:
            return False

        elapsed = time.time() - self._last_notification_time
        return elapsed < min_interval

    def _send_with_retry(
        self,
        endpoint: str,
        data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> LineResponse:
        """Send message with retry logic."""
        url = (
            f"{self.gateway_url}/{endpoint}"
            if self.gateway_url
            else f"{self.BASE_URL}/{endpoint}"
        )

        for attempt in range(self.retry_attempts):
            try:
                status_code, response_body = self.http_client.post(url, data, headers)

                if status_code == 200:
                    return LineResponse(
                        success=True,
                        status_code=status_code,
                        response_body=response_body.decode("utf-8")
                        if response_body
                        else None,
                    )
                else:
                    error_msg = f"HTTP {status_code}"
                    self._logger.warning(f"LINE API error: {error_msg}")
                    return LineResponse(
                        success=False,
                        status_code=status_code,
                        error_message=error_msg,
                    )

            except HTTPError as e:
                error_msg = f"HTTP Error {e.code}: {e.reason}"
                self._logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

                if attempt < self.retry_attempts - 1:
                    sleep_time = self.retry_delay * (2**attempt)
                    time.sleep(sleep_time)
                else:
                    return LineResponse(
                        success=False,
                        status_code=e.code,
                        error_message=error_msg,
                    )

            except URLError as e:
                error_msg = f"URL Error: {e.reason}"
                self._logger.warning(f"Attempt {attempt + 1} failed: {error_msg}")

                if attempt < self.retry_attempts - 1:
                    sleep_time = self.retry_delay * (2**attempt)
                    time.sleep(sleep_time)
                else:
                    return LineResponse(
                        success=False,
                        error_message=error_msg,
                    )

            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self._logger.error(error_msg)
                return LineResponse(
                    success=False,
                    error_message=error_msg,
                )

        return LineResponse(success=False, error_message="Max retries exceeded")

    def send_message(self, text: str) -> LineResponse:
        """Send a text message."""
        data = {"messages": [{"type": "text", "text": text}]}
        headers = self._get_headers()

        response = self._send_with_retry("message/broadcast", data, headers)

        if response.success:
            self._last_notification_time = time.time()
            self._notification_count += 1

        return response

    def send_training_update(
        self,
        metrics: "TrainingMetrics",
        phase_name: str = "",
        include_emoji: bool = True,
    ) -> LineResponse:
        """
        Send training update with detailed metrics.

        Format B (Detailed):
        ```
        ━━━━━━━━━━━━━━━━━━━━━━━━
        🚀 SFT Training - Phase Complete
        ━━━━━━━━━━━━━━━━━━━━━━━━
        📊 Metrics:
          • Loss: 0.0234 ↓
          • LR: 2.0e-5
          • Step: 150/500 (30%)
          • Epoch: 1/3

        ⏱️ Time:
          • Elapsed: 00:05:23
          • ETA: 00:12:34

        💾 Resources:
          • GPU Memory: 8.2/12.0 GB (68%)
        ━━━━━━━━━━━━━━━━━━━━━━━━
        ```
        """
        if include_emoji:
            message = self._format_detailed_message(metrics, phase_name)
        else:
            message = self._format_simple_message(metrics, phase_name)

        return self.send_message(message)

    def _format_detailed_message(
        self,
        metrics: "TrainingMetrics",
        phase_name: str,
    ) -> str:
        """Format detailed message with emojis."""
        # Phase header
        phase_display = phase_name or metrics.phase_name or "Training"
        emoji = self._get_phase_emoji(phase_display)

        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"{emoji} {phase_display} - Status Update",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            "",
        ]

        # Metrics section
        lines.append("📊 Metrics:")
        if metrics.loss is not None:
            loss_trend = self._get_loss_trend_emoji(metrics.loss)
            lines.append(f"  • Loss: {metrics.loss:.4f} {loss_trend}")
        if metrics.learning_rate is not None:
            lines.append(f"  • LR: {metrics.learning_rate:.2e}")
        if metrics.total_steps > 0:
            progress = (metrics.step / metrics.total_steps) * 100
            lines.append(
                f"  • Step: {metrics.step}/{metrics.total_steps} ({progress:.1f}%)"
            )
        if metrics.total_epochs > 0:
            lines.append(f"  • Epoch: {metrics.epoch}/{metrics.total_epochs}")
        if metrics.batch_size > 0:
            lines.append(f"  • Batch Size: {metrics.batch_size}")
        lines.append("")

        # Time section
        lines.append("⏱️ Time:")
        lines.append(f"  • Elapsed: {metrics.elapsed_formatted}")
        if metrics.eta_seconds is not None and metrics.eta_seconds > 0:
            lines.append(f"  • ETA: {metrics.eta_formatted}")
        else:
            lines.append(f"  • ETA: Calculating...")
        lines.append("")

        # Resources section
        if metrics.gpu_memory_total_gb > 0:
            gpu_percent = metrics.gpu_memory_percentage
            gpu_status = self._get_gpu_status_emoji(gpu_percent)
            lines.append("💾 Resources:")
            lines.append(
                f"  • GPU Memory: {metrics.gpu_memory_used_gb:.1f}/{metrics.gpu_memory_total_gb:.1f} GB ({gpu_percent:.1f}%)"
            )
            if metrics.gpu_utilization is not None:
                lines.append(f"  • GPU Util: {metrics.gpu_utilization:.1f}%")
            lines.append("")

        # Data progress
        if metrics.data_progress > 0:
            data_percent = metrics.data_progress * 100
            lines.append(f"📁 Data Progress: {data_percent:.1f}%")
            lines.append("")

        # Timestamp
        if metrics.timestamp:
            timestamp = metrics.timestamp.strftime("%H:%M:%S")
            lines.append(f"🕐 {timestamp}")

        # Footer
        if metrics.run_id:
            lines.append(f"ID: {metrics.run_id[:16]}...")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")

        return "\n".join(lines)

    def _format_simple_message(
        self,
        metrics: "TrainingMetrics",
        phase_name: str,
    ) -> str:
        """Format simple message without emojis."""
        lines = [f"[{phase_name or metrics.phase_name}]"]

        if metrics.loss is not None:
            lines.append(f"Loss: {metrics.loss:.4f}")
        if metrics.learning_rate is not None:
            lines.append(f"LR: {metrics.learning_rate:.2e}")
        if metrics.total_steps > 0:
            lines.append(f"Step: {metrics.step}/{metrics.total_steps}")
        if metrics.eta_seconds is not None:
            lines.append(f"ETA: {metrics.eta_formatted}")
        if metrics.gpu_memory_total_gb > 0:
            lines.append(
                f"GPU: {metrics.gpu_memory_used_gb:.1f}/{metrics.gpu_memory_total_gb:.1f} GB"
            )

        return " | ".join(lines)

    def _get_phase_emoji(self, phase_name: str) -> str:
        """Get emoji for phase."""
        phase_emojis = {
            "setup": "🔧",
            "data": "📊",
            "sft": "🚀",
            "grpo": "🎯",
            "benchmark": "📈",
            "statistics": "📉",
            "visualize": "🎨",
            "release": "✅",
            "error": "❌",
            "complete": "🎉",
        }

        phase_lower = phase_name.lower()
        for key, emoji in phase_emojis.items():
            if key in phase_lower:
                return emoji
        return "📌"

    def _get_loss_trend_emoji(self, loss: float) -> str:
        """Get trend indicator for loss."""
        if loss < 0.1:
            return "⬇️⬇️"  # Very low
        elif loss < 0.5:
            return "⬇️"  # Good
        elif loss < 1.0:
            return "➡️"  # Stable
        elif loss < 2.0:
            return "⬆️"  # High
        else:
            return "⬆️⬆️"  # Very high

    def _get_gpu_status_emoji(self, percentage: float) -> str:
        """Get status indicator for GPU memory."""
        if percentage < 50:
            return "🟢"
        elif percentage < 80:
            return "🟡"
        else:
            return "🔴"

    def send_phase_complete(
        self,
        phase_name: str,
        metrics: Optional["TrainingMetrics"] = None,
    ) -> LineResponse:
        """Send phase completion notification."""
        emoji = self._get_phase_emoji(phase_name)
        message = f"{emoji} Phase Complete: {phase_name}"

        if metrics is not None:
            message += f"\nFinal Loss: {metrics.loss:.4f}" if metrics.loss else ""
            message += f"\nTotal Time: {metrics.elapsed_formatted}"

        return self.send_message(message)

    def send_error_alert(
        self,
        error_message: str,
        phase_name: str = "",
    ) -> LineResponse:
        """Send error alert notification."""
        emoji = self._get_phase_emoji("error")
        message = f"{emoji} Error in {phase_name or 'Training'}\n{error_message}"
        return self.send_message(message)

    def send_training_complete(
        self,
        final_metrics: "TrainingMetrics",
        total_time: str,
    ) -> LineResponse:
        """Send training completion notification."""
        emoji = self._get_phase_emoji("complete")
        lines = [
            f"{emoji} Training Complete!",
            f"Total Time: {total_time}",
            "",
            "📊 Final Metrics:",
        ]

        if final_metrics.loss is not None:
            lines.append(f"  • Final Loss: {final_metrics.loss:.4f}")
        if final_metrics.epoch > 1:
            lines.append(f"  • Epochs: {final_metrics.epoch}")

        lines.append("")
        lines.append(f"Run ID: {final_metrics.run_id}")

        return self.send_message("\n".join(lines))

    def health_check(self) -> bool:
        """Check LINE connectivity."""
        if not self.access_token:
            self._logger.warning("No access token configured")
            return False

        try:
            # Simple check - try to send a test message
            response = self.send_message("🔍 Health Check")
            return response.success
        except Exception as e:
            self._logger.error(f"Health check failed: {e}")
            return False

    @property
    def notification_count(self) -> int:
        """Get number of notifications sent."""
        return self._notification_count


# Import TrainingMetrics for type hints
if False:
    from scripts.monitoring.modules.metrics_collector import TrainingMetrics
