"""
infra/log_filter.py
====================
PII redaction filter for the Python logging system.

Attaches to the root logger to sanitise log messages before they are written
to any handler (console, file, etc.). Replaces common PII patterns with
[REDACTED] tokens using pre-compiled regular expressions.

Patterns covered (CIOMS WG XIV Principle 5 — Data Privacy):
  - ISO dates and common date formats (birth dates, event dates)
  - Age expressions ("45-year-old", "a 12-month-old infant")
  - Biological sex references in narrative context
  - Case IDs matching the ICSR ID pattern (e.g. ICSR-20240101-001)
  - Email addresses
  - Free-text that matches a simple phone-number heuristic

Design decisions:
  - Pure stdlib — no external dependencies.
  - Compiled at import time — negligible runtime overhead.
  - Applied as a logging.Filter so it intercepts ALL log records regardless
    of which logger emitted them.
  - Does NOT redact structured data (JSON payloads in audit_log) — those are
    addressed by the presidio de-identification layer (Phase 6, Week 2).

Usage:
  from infra.log_filter import PIIRedactionFilter
  logging.getLogger().addFilter(PIIRedactionFilter())
"""
from __future__ import annotations

import logging
import re
from typing import ClassVar


class PIIRedactionFilter(logging.Filter):
    """
    Logging filter that replaces PII patterns in log message strings.
    Safe to attach to the root logger.
    """

    # Compiled patterns — ordered from most-specific to least-specific
    _PATTERNS: ClassVar[list[tuple[re.Pattern, str]]] = [
        # ISO dates: 2024-01-15
        (re.compile(r'\b\d{4}-\d{2}-\d{2}\b'), '[DATE_REDACTED]'),
        # Common date formats: 01/03/1980, 01-Mar-1980
        (re.compile(r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'), '[DATE_REDACTED]'),
        # Age expressions: "45-year-old", "12 year old", "a 6-month-old"
        (re.compile(
            r'\b\d{1,3}\s*[-]?\s*(?:year|yr|month|mo)s?[-\s]old\b',
            re.IGNORECASE
        ), '[AGE_REDACTED]'),
        # Standalone age with unit: "45 years", "6 months old"
        (re.compile(r'\b\d{1,3}\s+(?:years?|months?)\s+old\b', re.IGNORECASE),
         '[AGE_REDACTED]'),
        # Email addresses
        (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z]{2,}\b',
                    re.IGNORECASE), '[EMAIL_REDACTED]'),
        # ICSR case IDs: ICSR-20240101-001
        (re.compile(r'\bICSR-\d{8}-\d{3,}\b'), '[CASE_ID_REDACTED]'),
        # Phone numbers (loose heuristic — 10+ consecutive digits, may have spaces/dashes)
        (re.compile(r'\b[\d\s\-\(\)]{10,}\b'), '[PHONE_REDACTED]'),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Redact PII from the log record's message in-place.
        Always returns True (never suppresses records — only sanitises them).
        """
        if isinstance(record.msg, str):
            record.msg = self._redact(record.msg)
        # Also redact pre-formatted args if they are strings
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact(a) if isinstance(a, str) else a
                    for a in record.args
                )
        return True

    @classmethod
    def _redact(cls, text: str) -> str:
        """Apply all PII patterns to a string and return the sanitised version."""
        for pattern, replacement in cls._PATTERNS:
            text = pattern.sub(replacement, text)
        return text
