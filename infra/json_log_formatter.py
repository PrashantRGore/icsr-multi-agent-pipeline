"""
infra/json_log_formatter.py
============================
Structured JSON log formatter for production observability.

Emits one JSON object per log line, suitable for Grafana Loki, ELK Stack,
AWS CloudWatch, or any JSON-aware log aggregator.

Design:
  - Each LogRecord becomes a flat JSON dict (no nested objects except `extra`)
  - Standard fields always present: timestamp, level, logger, message, pid
  - Correlation fields propagated automatically if present in LogRecord:
      trace_id, case_id, review_id, agent_id  (set via logger.info(..., extra={...}))
  - Exception tracebacks serialized as a single `exception` string field
  - If JSON serialization itself fails, emits a safe plaintext fallback line
  - Thread-safe: uses no shared state

Usage:
  from infra.json_log_formatter import JSONFormatter

  handler = logging.StreamHandler()
  handler.setFormatter(JSONFormatter())
  logging.getLogger().addHandler(handler)

  # With correlation fields:
  logger.info("Processing case", extra={"trace_id": "ICSR-001", "agent_id": "TRIAGE"})
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import Any

# Correlation fields that are lifted to top-level JSON keys if present in extra
_CORRELATION_FIELDS = frozenset(
    {"trace_id", "case_id", "review_id", "agent_id", "reviewer_id"}
)

# Fields that exist on every LogRecord and must NOT be double-serialized into `extra`
_STDLIB_FIELDS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
})


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Parameters
    ----------
    service_name : str
        Added as `service` field to every log line (default: "icsr-hitl")
    include_pid : bool
        Include process ID in output (default: True)
    """

    def __init__(
        self,
        service_name: str = "icsr-hitl",
        include_pid: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._service   = service_name
        self._pid       = os.getpid() if include_pid else None

    def format(self, record: logging.LogRecord) -> str:
        # Ensure record.message is populated
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.message,
            "service":   self._service,
        }

        if self._pid is not None:
            payload["pid"] = self._pid

        # Lift correlation fields to top level
        for field in _CORRELATION_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                payload[field] = val

        # Remaining user-supplied extra fields → nested `extra` dict
        extra: dict[str, Any] = {}
        for key, val in record.__dict__.items():
            if key in _STDLIB_FIELDS or key in _CORRELATION_FIELDS:
                continue
            if key.startswith("_"):
                continue
            extra[key] = val
        if extra:
            payload["extra"] = extra

        # Exception info
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        elif record.exc_text:
            payload["exception"] = record.exc_text

        # Stack info (Python 3.2+)
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception as exc:
            # Safe fallback — must never raise
            return json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level":     "ERROR",
                "logger":    __name__,
                "message":   f"JSONFormatter serialization failed: {exc}. Original: {record.message}",
                "service":   self._service,
            })
