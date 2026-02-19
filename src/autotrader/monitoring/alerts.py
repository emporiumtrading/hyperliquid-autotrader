"""Alert routing via webhooks and structured logging."""

from __future__ import annotations

import json
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError

import structlog

from autotrader.monitoring.logger import get_logger


class AlertManager:
    """Sends alerts to a webhook URL and logs them via structlog.

    If no *webhook_url* is configured, alerts are logged only.
    """

    def __init__(self, webhook_url: str | None = None) -> None:
        self._webhook_url = webhook_url
        self._logger: structlog.stdlib.BoundLogger = get_logger("alerts")

    def send(
        self,
        severity: str,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send an alert.

        * Always logs the alert via structlog.
        * If *webhook_url* is configured, posts a JSON payload to it.

        Returns ``True`` if the alert was successfully dispatched (or if no
        webhook is configured).  Returns ``False`` only when the webhook POST
        fails.
        """
        log_data: dict[str, Any] = {
            "severity": severity,
            "alert_title": title,
            "alert_message": message,
        }
        if data:
            log_data["alert_data"] = data

        # Log at appropriate level
        if severity == "critical":
            self._logger.error("alert_fired", **log_data)
        elif severity == "warning":
            self._logger.warning("alert_fired", **log_data)
        else:
            self._logger.info("alert_fired", **log_data)

        # If no webhook configured, consider it a success (logged only)
        if not self._webhook_url:
            return True

        # POST to webhook
        payload = {
            "severity": severity,
            "title": title,
            "message": message,
            "data": data or {},
        }

        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib_request.Request(
                self._webhook_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=10) as resp:
                _ = resp.read()
            return True
        except (URLError, OSError, ValueError) as exc:
            self._logger.error(
                "alert_webhook_failed",
                webhook_url=self._webhook_url,
                error=str(exc),
            )
            return False

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def critical(
        self,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send a critical-severity alert."""
        return self.send("critical", title, message, data)

    def warning(
        self,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send a warning-severity alert."""
        return self.send("warning", title, message, data)

    def info(
        self,
        title: str,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> bool:
        """Send an info-severity alert."""
        return self.send("info", title, message, data)


# ---------------------------------------------------------------------------
# Module-level convenience API
# ---------------------------------------------------------------------------

_manager: AlertManager | None = None


def init(webhook_url: str | None = None) -> None:
    """Initialise the module-level AlertManager singleton."""
    global _manager  # noqa: PLW0603
    _manager = AlertManager(webhook_url=webhook_url)


def send(
    severity: str,
    title: str,
    message: str,
    data: dict[str, Any] | None = None,
) -> bool:
    """Send an alert via the module-level manager.

    Lazily initialises a log-only AlertManager if :func:`init` has not been
    called.
    """
    global _manager  # noqa: PLW0603
    if _manager is None:
        _manager = AlertManager()
    return _manager.send(severity, title, message, data)
