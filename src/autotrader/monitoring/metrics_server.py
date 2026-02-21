"""Lightweight HTTP server for Prometheus metrics exposition.

Serves the ``/metrics`` endpoint on the configured Prometheus port
(default 9109) using the standard library ``http.server`` to avoid
additional dependencies.  Runs in a daemon thread so it does not
block the main trading loop.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import structlog

from autotrader.monitoring.dashboards import export_prometheus
from autotrader.monitoring.metrics import metrics

logger = structlog.get_logger(__name__)


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler that serves Prometheus text exposition."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/metrics":
            snapshot = metrics.snapshot()
            body = export_prometheus(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default stderr logging; use structlog instead."""
        pass


class MetricsServer:
    """Prometheus metrics HTTP server running in a background thread.

    Parameters
    ----------
    port:
        TCP port to bind to (default 9109).
    host:
        Address to bind to (default ``"0.0.0.0"``).
    """

    def __init__(self, port: int = 9109, host: str = "0.0.0.0") -> None:
        self.port = port
        self.host = host
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start serving metrics in a background daemon thread."""
        if self._server is not None:
            return

        self._server = HTTPServer((self.host, self.port), _MetricsHandler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="metrics-http",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "metrics_server.started",
            host=self.host,
            port=self.port,
        )

    def stop(self) -> None:
        """Shut down the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None
            logger.info("metrics_server.stopped")

    @property
    def is_running(self) -> bool:
        """Return True if the server is currently running."""
        return self._server is not None and self._thread is not None and self._thread.is_alive()
