"""Phase 5 — alert dispatch on restricted-zone violations / unknown persons."""
from __future__ import annotations

import logging
from typing import Protocol

from visionstack.common.types import Alert

logger = logging.getLogger("visionstack.alerts")


class AlertDispatcher(Protocol):
    def dispatch(self, alert: Alert) -> None:
        """Deliver an alert to security/HR (push notification, dashboard flag, etc.)."""
        ...


class LoggingAlertDispatcher:
    """Real-enough-for-now: logs alerts to stdout. TODO: real push-notification/dashboard channels."""

    def dispatch(self, alert: Alert) -> None:
        logger.warning("ALERT [%s/%s] %s", alert.alert_type, alert.severity, alert.message)
