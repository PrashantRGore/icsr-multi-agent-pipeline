"""
tests/unit/test_json_formatter.py
===================================
Unit tests for infra/json_log_formatter.JSONFormatter.
"""
from __future__ import annotations

import json
import logging

import pytest

from infra.json_log_formatter import JSONFormatter


def _make_record(
    msg: str = "test message",
    level: int = logging.INFO,
    name: str = "test.logger",
    **extra,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name, level=level, pathname="", lineno=0,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


def _format(record: logging.LogRecord, **kwargs) -> dict:
    fmt = JSONFormatter(**kwargs)
    line = fmt.format(record)
    return json.loads(line)


class TestBasicFields:
    def test_always_has_required_keys(self) -> None:
        data = _format(_make_record())
        for key in ("timestamp", "level", "logger", "message", "service"):
            assert key in data

    def test_message_content(self) -> None:
        data = _format(_make_record("hello world"))
        assert data["message"] == "hello world"

    def test_level_name(self) -> None:
        data = _format(_make_record(level=logging.WARNING))
        assert data["level"] == "WARNING"

    def test_default_service_name(self) -> None:
        data = _format(_make_record())
        assert data["service"] == "icsr-hitl"

    def test_custom_service_name(self) -> None:
        data = _format(_make_record(), service_name="custom-svc")
        assert data["service"] == "custom-svc"

    def test_pid_present_by_default(self) -> None:
        data = _format(_make_record())
        assert "pid" in data
        assert isinstance(data["pid"], int)

    def test_pid_excluded_when_disabled(self) -> None:
        data = _format(_make_record(), include_pid=False)
        assert "pid" not in data

    def test_timestamp_is_iso8601_utc(self) -> None:
        data = _format(_make_record())
        ts = data["timestamp"]
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_output_is_single_line(self) -> None:
        fmt = JSONFormatter()
        line = fmt.format(_make_record("line\nwith\nnewlines"))
        assert "\n" not in line


class TestCorrelationFields:
    def test_trace_id_promoted_to_top_level(self) -> None:
        data = _format(_make_record(trace_id="ICSR-001"))
        assert data["trace_id"] == "ICSR-001"
        assert "trace_id" not in data.get("extra", {})

    def test_case_id_promoted(self) -> None:
        data = _format(_make_record(case_id="C-001"))
        assert data["case_id"] == "C-001"

    def test_agent_id_promoted(self) -> None:
        data = _format(_make_record(agent_id="TRIAGE"))
        assert data["agent_id"] == "TRIAGE"

    def test_review_id_promoted(self) -> None:
        data = _format(_make_record(review_id="rev-abc"))
        assert data["review_id"] == "rev-abc"

    def test_unknown_extra_goes_to_extra_dict(self) -> None:
        data = _format(_make_record(custom_key="custom_val"))
        assert "extra" in data
        assert data["extra"]["custom_key"] == "custom_val"

    def test_no_extra_dict_when_no_extra(self) -> None:
        # Record with only stdlib fields — no user extras
        data = _format(_make_record())
        assert "extra" not in data


class TestExceptionHandling:
    def test_exception_field_present_on_exc_info(self) -> None:
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = _make_record("something failed")
        record.exc_info = exc_info
        data = _format(record)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_no_exception_field_when_no_error(self) -> None:
        data = _format(_make_record())
        assert "exception" not in data


class TestSerializationSafety:
    def test_unserializable_extra_uses_str_fallback(self) -> None:
        """Non-serializable objects should not crash the formatter."""
        class Unserializable:
            def __repr__(self): return "<Unserializable>"

        data = _format(_make_record(obj=Unserializable()))
        # Should still produce valid JSON with str() fallback
        assert "extra" in data

    def test_unicode_message_survives(self) -> None:
        data = _format(_make_record("Patiënt: François Müller"))
        assert "François" in data["message"]
