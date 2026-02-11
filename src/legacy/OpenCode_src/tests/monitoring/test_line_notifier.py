#!/usr/bin/env python3
"""
Unit Tests for LINE Notifier Module.

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


class TestLineMessage:
    """Test cases for LineMessage dataclass."""

    def test_default_creation(self):
        """Test LineMessage creation with defaults."""
        from scripts.monitoring.modules.line_notifier import LineMessage

        message = LineMessage(text="Hello, LINE!")

        assert message.text == "Hello, LINE!"
        assert message.message_type == "text"

    def test_custom_creation(self):
        """Test LineMessage creation with custom type."""
        from scripts.monitoring.modules.line_notifier import LineMessage

        message = LineMessage(text="Test message", message_type="image")

        assert message.text == "Test message"
        assert message.message_type == "image"


class TestLineResponse:
    """Test cases for LineResponse dataclass."""

    def test_success_response(self):
        """Test successful LineResponse."""
        from scripts.monitoring.modules.line_notifier import LineResponse

        response = LineResponse(
            success=True,
            status_code=200,
            response_body='{"status":"ok"}',
        )

        assert response.success is True
        assert response.status_code == 200
        assert response.response_body == '{"status":"ok"}'

    def test_error_response(self):
        """Test error LineResponse."""
        from scripts.monitoring.modules.line_notifier import LineResponse

        response = LineResponse(
            success=False,
            status_code=500,
            error_message="Internal Server Error",
        )

        assert response.success is False
        assert response.status_code == 500
        assert response.error_message == "Internal Server Error"


class TestMockHttpClient:
    """Test cases for MockHttpClient."""

    def test_successful_request(self):
        """Test mock client with successful request."""
        from scripts.monitoring.modules.line_notifier import MockHttpClient

        client = MockHttpClient(should_fail=False)

        status_code, body = client.post(
            url="https://api.line.me/v2/bot/message/broadcast",
            data={"messages": [{"type": "text", "text": "Test"}]},
            headers={"Authorization": "Bearer token"},
        )

        assert status_code == 200
        assert body == b'{"status":"ok"}'
        assert len(client.requests) == 1

    def test_failed_request(self):
        """Test mock client with failed request."""
        from scripts.monitoring.modules.line_notifier import MockHttpClient
        from urllib.error import HTTPError

        client = MockHttpClient(should_fail=True, fail_status=500)

        with pytest.raises(HTTPError):
            client.post(
                url="https://api.line.me/v2/bot/message/broadcast",
                data={},
                headers={},
            )


class TestLineNotifier:
    """Test cases for LineNotifier."""

    def test_initialization(self):
        """Test LineNotifier initialization."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(
            access_token="test_token_12345",
            channel_secret="test_secret",
            gateway_url="http://localhost:18789",
            retry_attempts=3,
            retry_delay=1.0,
            message_format="detailed",
        )

        assert notifier.access_token == "test_token_12345"
        assert notifier.channel_secret == "test_secret"
        assert notifier.gateway_url == "http://localhost:18789"
        assert notifier.retry_attempts == 3
        assert notifier.message_format == "detailed"

    def test_get_headers(self):
        """Test header generation."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")

        headers = notifier._get_headers()

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer token123"
        assert "Content-Type" in headers
        assert headers["Content-Type"] == "application/json"

    def test_should_throttle_first_message(self):
        """Test throttling for first message."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")

        assert notifier._should_throttle() is False

    def test_should_throttle_after_interval(self):
        """Test throttling after time interval."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")
        notifier._last_notification_time = time.time() - 120  # 2 minutes ago

        assert notifier._should_throttle(min_interval=60) is False

    def test_should_throttle_within_interval(self):
        """Test throttling within interval."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")
        notifier._last_notification_time = time.time() - 10  # 10 seconds ago

        assert notifier._should_throttle(min_interval=60) is True

    def test_send_message_success(self):
        """Test sending message successfully."""
        from scripts.monitoring.modules.line_notifier import (
            LineNotifier,
            MockHttpClient,
        )

        client = MockHttpClient(should_fail=False)
        notifier = LineNotifier(
            access_token="token123",
            http_client=client,
        )

        response = notifier.send_message("Test message")

        assert response.success is True
        assert response.status_code == 200
        assert notifier._last_notification_time is not None
        assert notifier._notification_count == 1

    def test_send_message_failure(self):
        """Test sending message with failure."""
        from scripts.monitoring.modules.line_notifier import (
            LineNotifier,
            MockHttpClient,
        )

        client = MockHttpClient(should_fail=True, fail_status=500)
        notifier = LineNotifier(
            access_token="token123",
            http_client=client,
            retry_attempts=1,
        )

        response = notifier.send_message("Test message")

        assert response.success is False
        assert response.status_code == 500

    def test_notification_count(self):
        """Test notification count tracking."""
        from scripts.monitoring.modules.line_notifier import (
            LineNotifier,
            MockHttpClient,
        )

        client = MockHttpClient(should_fail=False)
        notifier = LineNotifier(
            access_token="token123",
            http_client=client,
        )

        notifier.send_message("Message 1")
        notifier.send_message("Message 2")
        notifier.send_message("Message 3")

        assert notifier._notification_count == 3
        assert notifier.notification_count == 3


class TestLineNotifierMessageFormatting:
    """Test cases for LINE message formatting."""

    def test_format_detailed_message(self):
        """Test detailed message formatting."""
        from scripts.monitoring.modules.line_notifier import LineNotifier
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        notifier = LineNotifier(
            access_token="token123",
            message_format="detailed",
        )

        metrics = TrainingMetrics(
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
            timestamp=datetime(2026, 2, 2, 12, 30, 45),
        )

        message = notifier._format_detailed_message(metrics, "SFT Training")

        # Check key elements
        assert "SFT Training" in message
        assert "Loss: 0.0234" in message
        assert "LR: 2.0e-05" in message
        assert "Step: 150/500 (30.0%)" in message
        assert "Epoch: 1/3" in message
        assert "Elapsed: 5m 23s" in message
        assert "ETA: 12m 34s" in message
        assert "GPU Memory: 8.2/12.0 GB (68.3%)" in message

    def test_format_simple_message(self):
        """Test simple message formatting."""
        from scripts.monitoring.modules.line_notifier import LineNotifier
        from scripts.monitoring.modules.metrics_collector import TrainingMetrics

        notifier = LineNotifier(
            access_token="token123",
            message_format="simple",
        )

        metrics = TrainingMetrics(
            run_id="test_run",
            step=100,
            total_steps=500,
            loss=0.5,
            eta_seconds=1200,
            gpu_memory_used_gb=8.0,
            gpu_memory_total_gb=12.0,
            phase_name="sft",
        )

        message = notifier._format_simple_message(metrics, "SFT")

        assert "[SFT]" in message
        assert "Loss: 0.5" in message
        assert "Step: 100/500" in message
        assert "ETA: 20m 0s" in message

    def test_get_phase_emoji(self):
        """Test phase emoji mapping."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")

        assert notifier._get_phase_emoji("setup") == "🔧"
        assert notifier._get_phase_emoji("sft") == "🚀"
        assert notifier._get_phase_emoji("grpo") == "🎯"
        assert notifier._get_phase_emoji("benchmark") == "📈"
        assert notifier._get_phase_emoji("release") == "✅"
        assert notifier._get_phase_emoji("error") == "❌"
        assert notifier._get_phase_emoji("complete") == "🎉"

    def test_get_loss_trend_emoji(self):
        """Test loss trend emoji."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")

        assert notifier._get_loss_trend_emoji(0.05) == "⬇️⬇️"
        assert notifier._get_loss_trend_emoji(0.3) == "⬇️"
        assert notifier._get_loss_trend_emoji(0.7) == "➡️"
        assert notifier._get_loss_trend_emoji(1.5) == "⬆️"
        assert notifier._get_loss_trend_emoji(3.0) == "⬆️⬆️"

    def test_get_gpu_status_emoji(self):
        """Test GPU status emoji."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="token123")

        assert notifier._get_gpu_status_emoji(30.0) == "🟢"
        assert notifier._get_gpu_status_emoji(65.0) == "🟡"
        assert notifier._get_gpu_status_emoji(85.0) == "🔴"


class TestLineNotifierRetryLogic:
    """Test cases for LINE notifier retry logic."""

    def test_retry_on_failure(self):
        """Test retry on transient failure."""

        class FailThenSucceedClient:
            def __init__(self):
                self.call_count = 0

            def post(self, url, data, headers):
                self.call_count += 1
                if self.call_count == 1:
                    from urllib.error import HTTPError

                    raise HTTPError(url, 503, "Service Unavailable", {}, None)
                return 200, b'{"status":"ok"}'

        from scripts.monitoring.modules.line_notifier import LineNotifier

        client = FailThenSucceedClient()
        notifier = LineNotifier(
            access_token="token123",
            http_client=client,
            retry_attempts=3,
            retry_delay=0.01,
        )

        response = notifier.send_message("Test")

        assert response.success is True
        assert client.call_count == 2

    def test_max_retries_exceeded(self):
        """Test max retries exceeded."""

        class AlwaysFailClient:
            def post(self, url, data, headers):
                from urllib.error import HTTPError

                raise HTTPError(url, 500, "Server Error", {}, None)

        from scripts.monitoring.modules.line_notifier import LineNotifier

        client = AlwaysFailClient()
        notifier = LineNotifier(
            access_token="token123",
            http_client=client,
            retry_attempts=3,
            retry_delay=0.01,
        )

        response = notifier.send_message("Test")

        assert response.success is False
        assert response.status_code == 500


class TestLineNotifierHealthCheck:
    """Test cases for LINE notifier health check."""

    def test_health_check_no_token(self):
        """Test health check without token."""
        from scripts.monitoring.modules.line_notifier import LineNotifier

        notifier = LineNotifier(access_token="")

        assert notifier.health_check() is False

    def test_health_check_success(self):
        """Test successful health check."""
        from scripts.monitoring.modules.line_notifier import (
            LineNotifier,
            MockHttpClient,
        )

        client = MockHttpClient(should_fail=False)
        notifier = LineNotifier(
            access_token="token123",
            http_client=client,
        )

        assert notifier.health_check() is True
