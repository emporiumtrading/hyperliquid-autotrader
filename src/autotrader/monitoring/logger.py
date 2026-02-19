"""Structured logging setup using structlog."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog with JSON or console rendering.

    * If the ``ENVIRONMENT`` env var is set to ``"production"`` (case-insensitive)
      a JSON renderer is used; otherwise the coloured console renderer is used.
    * Adds timestamp, log level, and caller information processors.
    * Also sets the Python standard-library root logger to *level* so that
      third-party libraries respect the same threshold.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Determine renderer based on environment
    is_production = os.environ.get("ENVIRONMENT", "").lower() == "production"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
            ]
        ),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if is_production:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging so structlog formatters take effect
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger.

    If *name* is provided it will be set as the ``logger_name`` context
    variable; otherwise a nameless logger is returned.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


def log_trade(
    logger: structlog.stdlib.BoundLogger,
    side: str,
    symbol: str,
    size: float,
    price: float,
    **extra: object,
) -> None:
    """Log an executed trade with structured data."""
    logger.info(
        "trade_executed",
        side=side,
        symbol=symbol,
        size=size,
        price=price,
        notional=size * price,
        **extra,
    )


def log_signal(
    logger: structlog.stdlib.BoundLogger,
    symbol: str,
    side: str,
    confidence: float,
    **extra: object,
) -> None:
    """Log a generated trading signal."""
    logger.info(
        "signal_generated",
        symbol=symbol,
        side=side,
        confidence=confidence,
        **extra,
    )


def log_risk_event(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    **extra: object,
) -> None:
    """Log a risk-related event (breaches, margin warnings, etc.)."""
    logger.warning(
        "risk_event",
        risk_event=event,
        **extra,
    )
