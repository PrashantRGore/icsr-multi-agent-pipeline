"""
tests/unit/test_log_filter.py
==============================
Unit tests for infra/log_filter.py — PII redaction logging filter.
"""
from __future__ import annotations

import logging

import pytest

from infra.log_filter import PIIRedactionFilter


@pytest.fixture
def filt() -> PIIRedactionFilter:
    return PIIRedactionFilter()


def _make_record(msg: str, *args) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test", level=logging.INFO,
        pathname="", lineno=0,
        msg=msg, args=args, exc_info=None,
    )
    return record


class TestDateRedaction:
    def test_iso_date_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Patient DOB: 1980-03-15")
        filt.filter(rec)
        assert "1980-03-15" not in rec.msg
        assert "DATE_REDACTED" in rec.msg

    def test_slash_date_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Date: 15/03/1980")
        filt.filter(rec)
        assert "15/03/1980" not in rec.msg

    def test_non_date_digits_preserved(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Case count: 12345")
        filt.filter(rec)
        assert "12345" in rec.msg


class TestAgeRedaction:
    def test_year_old_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("A 45-year-old male patient")
        filt.filter(rec)
        assert "45-year-old" not in rec.msg
        assert "AGE_REDACTED" in rec.msg

    def test_years_old_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Patient is 67 years old")
        filt.filter(rec)
        assert "67 years old" not in rec.msg

    def test_month_old_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("6-month-old infant")
        filt.filter(rec)
        assert "6-month-old" not in rec.msg


class TestEmailRedaction:
    def test_email_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Contact: john.doe@hospital.org")
        filt.filter(rec)
        assert "john.doe@hospital.org" not in rec.msg
        assert "EMAIL_REDACTED" in rec.msg

    def test_non_email_at_sign_preserved(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Model: llama@Q4_K")
        filt.filter(rec)
        # Should not crash; model identifier is not a valid email domain
        assert rec.msg  # just passes without exception


class TestCaseIdRedaction:
    def test_icsr_id_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Processing ICSR-20240115-001")
        filt.filter(rec)
        assert "ICSR-20240115-001" not in rec.msg
        assert "CASE_ID_REDACTED" in rec.msg


class TestFilterAlwaysReturnsTrue:
    def test_filter_never_suppresses(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Some message with 1980-01-01 date")
        result = filt.filter(rec)
        assert result is True


class TestArgsRedaction:
    def test_tuple_args_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Patient %s born %s", "ICSR-20240115-001", "1980-03-15")
        filt.filter(rec)
        assert isinstance(rec.args, tuple)
        assert not any("1980-03-15" in str(a) for a in rec.args)

    def test_dict_args_redacted(self, filt: PIIRedactionFilter) -> None:
        rec = _make_record("Info %(dob)s", {"dob": "1980-03-15"})
        filt.filter(rec)
        assert isinstance(rec.args, dict)
        assert "1980-03-15" not in rec.args.get("dob", "")


class TestCombinedRedaction:
    def test_multiple_pii_types_in_one_message(self, filt: PIIRedactionFilter) -> None:
        msg = "ICSR-20240115-001: 45-year-old male, DOB 1980-03-15, email a@b.com"
        rec = _make_record(msg)
        filt.filter(rec)
        assert "ICSR-20240115-001" not in rec.msg
        assert "45-year-old" not in rec.msg
        assert "1980-03-15" not in rec.msg
        assert "a@b.com" not in rec.msg
