"""Structured logging for the SmartShop API.

- JSON log lines in production (``SMARTSHOP_JSON_LOGS`` overrides), plain text
  in dev so local output stays readable.
- A request-ID contextvar that the API middleware sets per request; every log
  record emitted while handling that request carries the same ``request_id``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_DEV_ENVIRONMENTS = {"dev", "local", "test"}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key in ("action", "reason", "tool", "latency_seconds", "model"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    json_logs: bool | None = None,
    level: str | None = None,
) -> None:
    """Configure root logging once; safe to call multiple times."""
    environment = (os.getenv("SMARTSHOP_ENV") or "dev").strip().lower()
    if json_logs is None:
        json_logs = _env_flag(
            "SMARTSHOP_JSON_LOGS", environment not in _DEV_ENVIRONMENTS
        )
    level_name = (level or os.getenv("SMARTSHOP_LOG_LEVEL") or "INFO").upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())
    if json_logs:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
            )
        )

    root = logging.getLogger()
    root.setLevel(level_name)
    # Replace only handlers we installed before, keep e.g. pytest capture intact.
    for existing in list(root.handlers):
        if getattr(existing, "_smartshop_handler", False):
            root.removeHandler(existing)
    handler._smartshop_handler = True  # type: ignore[attr-defined]
    root.addHandler(handler)
