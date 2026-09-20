"""
tests/unit/test_learning_pipeline.py
======================================
Unit tests for Phase 5: Learning Pipeline + PSMF Report Generation.

Coverage
--------
SecondaryQCAuditor (12 tests)
  - Field classification rules (triage, extraction sub-fields, causality,
    coding, listedness, unknown fallback)
  - learning_signal=False -> skipped, no DB writes
  - Multiple changed fields -> multiple DB rows
  - Demographic fields passed through to LearningDB
  - AuditLogEntry written on every call
  - Signal count increments in a real in-memory LearningDB

PSMF generator (6 tests)
  - All {{PLACEHOLDER}} tokens replaced in output
  - Agent pass-rate calculation from AuditDB stats
  - Output file written to correct directory with timestamp name
  - Empty AuditDB -> graceful "N/A" placeholders
  - --owner CLI argument substituted into report
  - LearningDB top_patterns appear in report
"""
from __future__ import annotations

import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# -- Project root on path so imports resolve ----------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.secondary_qc_auditor import (
    LearningSignalResult,
    SecondaryQCAuditor,
    _classify_field,
    _derive_age_group,
    _values_equal,
)
from infra.audit_db import AuditDB
from infra.learning_db import LearningDB
from schemas.audit import HITLReviewRecord, HITLStage
from schemas.qc import ErrorClassification


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_review_record(
    corrected_value: object,
    original_value: object = None,
    stage: HITLStage = HITLStage.TRIAGE,
    learning_signal: bool = True,
    approved: bool = True,
) -> HITLReviewRecord:
    return HITLReviewRecord(
        review_id            = str(uuid.uuid4()),
        trace_id             = str(uuid.uuid4()),
        stage                = stage,
        reviewer_id          = "TEST-REVIEWER",
        original_value       = original_value or {},
        corrected_value      = corrected_value,
        correction_rationale = "Unit test correction rationale — sufficient length",
        approved             = approved,
        learning_signal      = learning_signal,
    )


def _make_dbs() -> tuple[AuditDB, LearningDB]:
    """Use temp files not :memory: — SQLite :memory: is connection-scoped."""
    import tempfile
    a = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    a.close()
    b = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    b.close()
    return AuditDB(db_path=a.name), LearningDB(db_path=b.name)


# ─────────────────────────────────────────────────────────────────────────────
# SecondaryQCAuditor — field classification
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldClassification:

    def test_triage_correction_classification(self):
        path, cls = _classify_field("triage_output", {}, {"status": "VALID"})
        assert path == "triage_output"
        assert cls == ErrorClassification.CONFIDENCE_INSUFFICIENT

    def test_causality_correction_classification(self):
        path, cls = _classify_field("causality_matrix", {}, {"terms": []})
        assert path == "causality_matrix"
        assert cls == ErrorClassification.CAUSALITY_UNSUPPORTED

    def test_coding_correction_classification(self):
        path, cls = _classify_field("coded_events", [], [{"code": "X"}])
        assert path == "coded_events"
        assert cls == ErrorClassification.CODING_MISMATCH

    def test_listedness_correction_classification(self):
        path, cls = _classify_field("listedness_evaluations", {}, {"status": "LISTED"})
        assert path == "listedness_evaluations"
        assert cls == ErrorClassification.LISTEDNESS_UNCERTAIN

    def test_extraction_drug_sub_field_classification(self):
        original  = {"suspect_drugs": [{"drug_name": "Aspirin"}]}
        corrected = {"suspect_drugs": [{"drug_name": "Aspirin"}, {"drug_name": "Ibuprofen"}]}
        path, cls = _classify_field("extracted_entities", original, corrected)
        assert path == "extracted_entities.suspect_drugs"
        assert cls == ErrorClassification.OMISSION

    def test_extraction_ae_sub_field_classification(self):
        original  = {"adverse_events": [{"verbatim_term": "rash"}]}
        corrected = {"adverse_events": [{"verbatim_term": "severe rash"}]}
        path, cls = _classify_field("extracted_entities", original, corrected)
        assert path == "extracted_entities.adverse_events"
        assert cls == ErrorClassification.OMISSION

    def test_extraction_date_sub_field_classification(self):
        original  = {"onset_date": {"year": 2024, "month": 1}}
        corrected = {"onset_date": {"year": 2024, "month": 2}}
        path, cls = _classify_field("extracted_entities", original, corrected)
        assert path == "extracted_entities.onset_date"
        assert cls == ErrorClassification.DATE_ORDER_VIOLATION

    def test_extraction_rechallenge_polarity_flip(self):
        original  = {"rechallenge": "NO"}
        corrected = {"rechallenge": "YES"}
        path, cls = _classify_field("extracted_entities", original, corrected)
        assert path == "extracted_entities.rechallenge"
        assert cls == ErrorClassification.POLARITY_FLIP

    def test_unknown_field_fallback_classification(self):
        path, cls = _classify_field("unknown_field_xyz", None, "new_value")
        assert path == "unknown_field_xyz"
        assert cls == ErrorClassification.SCHEMA_VIOLATION


class TestValuesEqual:

    def test_equal_dicts(self):
        assert _values_equal({"a": 1}, {"a": 1}) is True

    def test_json_string_vs_dict(self):
        assert _values_equal('{"a": 1}', {"a": 1}) is True

    def test_different_values(self):
        assert _values_equal("VALID", "INVALID") is False

    def test_none_vs_none(self):
        assert _values_equal(None, None) is True


class TestDeriveAgeGroup:

    def test_pediatric(self):
        assert _derive_age_group({"patient": {"age_at_onset": 10}}) == "PEDIATRIC"

    def test_adult(self):
        assert _derive_age_group({"patient": {"age_at_onset": 35}}) == "ADULT"

    def test_elderly(self):
        assert _derive_age_group({"patient": {"age_at_onset": 70}}) == "ELDERLY"

    def test_no_age_returns_none(self):
        assert _derive_age_group({}) is None


# ─────────────────────────────────────────────────────────────────────────────
# SecondaryQCAuditor — audit_correction behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestSecondaryQCAuditor:

    def test_learning_signal_false_skips_db(self):
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        record = _make_review_record(
            corrected_value={"triage_output": {"status": "VALID"}},
            original_value={"triage_output": {"status": "INVALID"}},
            learning_signal=False,
        )
        result = auditor.audit_correction(record, {})
        assert result.skipped is True
        assert result.signals_written == 0
        # LearningDB should have no signals
        assert learning_db.count_signals() == 0

    def test_triage_correction_writes_signal(self):
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        record = _make_review_record(
            corrected_value={"triage_output": {"status": "VALID"}},
            original_value={"triage_output": {"status": "INVALID"}},
            stage=HITLStage.TRIAGE,
        )
        result = auditor.audit_correction(record, {})
        assert result.signals_written == 1
        assert "triage_output" in result.field_paths
        assert ErrorClassification.CONFIDENCE_INSUFFICIENT.value in result.error_classes
        assert learning_db.count_signals() == 1

    def test_multiple_corrected_fields_multiple_signals(self):
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        record = _make_review_record(
            original_value={
                "triage_output":      {"status": "INVALID"},
                "causality_matrix":   {"terms": []},
                "coded_events":       [],
            },
            corrected_value={
                "triage_output":      {"status": "VALID"},
                "causality_matrix":   {"terms": [{"term": "PROBABLE"}]},
                "coded_events":       [{"code": "OAE:001"}],
            },
            stage=HITLStage.TRIAGE,
        )
        result = auditor.audit_correction(record, {})
        assert result.signals_written == 3
        assert learning_db.count_signals() == 3

    def test_unchanged_fields_not_written(self):
        """If corrected_value == original_value for a field, skip it."""
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        record = _make_review_record(
            original_value={
                "triage_output": {"status": "INVALID"},
                "next_stage":    "extraction",   # unchanged
            },
            corrected_value={
                "triage_output": {"status": "VALID"},
                "next_stage":    "extraction",   # same -> skip
            },
        )
        result = auditor.audit_correction(record, {})
        assert result.signals_written == 1   # only triage_output

    def test_demographic_fields_in_signal(self):
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        state_snap = {
            "extracted_entities": {
                "patient_ethnicity": "CAUCASIAN",
                "patient": {"sex": "FEMALE", "age_at_onset": 72},
            }
        }
        record = _make_review_record(
            corrected_value={"triage_output": {"status": "VALID"}},
            original_value={"triage_output": {"status": "INVALID"}},
        )
        result = auditor.audit_correction(record, state_snap)
        assert result.signals_written == 1
        # Verify demographic data stored in LearningDB
        with learning_db._connect() as conn:
            row = conn.execute("SELECT * FROM learning_signals LIMIT 1").fetchone()
        assert row["patient_ethnicity"] == "CAUCASIAN"
        assert row["patient_sex"]       == "FEMALE"
        assert row["patient_age_group"] == "ELDERLY"

    def test_audit_entry_written_on_success(self):
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        before = audit_db.count_entries()
        record = _make_review_record(
            corrected_value={"causality_matrix": {"terms": []}},
            original_value={"causality_matrix": {}},
        )
        auditor.audit_correction(record, {})
        # One AuditLogEntry should have been written by the auditor
        assert audit_db.count_entries() == before + 1

    def test_audit_entry_written_when_skipped(self):
        """Even when skipped, an AuditLogEntry is written for traceability."""
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        before = audit_db.count_entries()
        record = _make_review_record(
            corrected_value={},
            learning_signal=False,
        )
        auditor.audit_correction(record, {})
        assert audit_db.count_entries() == before + 1

    def test_signal_count_increments_across_multiple_calls(self):
        audit_db, learning_db = _make_dbs()
        auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
        for i in range(3):
            record = _make_review_record(
                corrected_value={"triage_output": {"status": f"VALID_{i}"}},
                original_value={"triage_output": {"status": "INVALID"}},
            )
            auditor.audit_correction(record, {})
        assert learning_db.count_signals() == 3


# ─────────────────────────────────────────────────────────────────────────────
# PSMF Generator
# ─────────────────────────────────────────────────────────────────────────────

class TestPSMFGenerator:
    """Tests for scripts/generate_psmf_report.py render logic."""

    @pytest.fixture
    def dbs(self, tmp_path):
        """Real in-memory DBs (not :memory: so PSMF can receive real Path args)."""
        audit_path    = tmp_path / "audit.db"
        learning_path = tmp_path / "learning.db"
        return AuditDB(db_path=str(audit_path)), LearningDB(db_path=str(learning_path))

    @pytest.fixture
    def template_path(self):
        return Path(__file__).resolve().parents[2] / "governance" / "psmf_template.md"

    def test_placeholder_replacement_all_filled(self, dbs, template_path):
        """After render_psmf(), no {{...}} tokens should remain."""
        from scripts.generate_psmf_report import render_psmf
        audit_db, learning_db = dbs
        now = datetime.now(timezone.utc)
        report = render_psmf(
            template_path=template_path,
            audit_db=audit_db,
            learning_db=learning_db,
            owner="TEST-QPPV",
            now=now,
        )
        import re
        # Strip the intentional footer sentinel {{PLACEHOLDER}} before asserting
        _raw = re.findall(r"\{\{([^}]+)\}\}", report)
        remaining = [t for t in _raw if t != "PLACEHOLDER"]
        assert remaining == [], f"Unreplaced placeholders: {remaining}"

    def test_empty_audit_db_graceful(self, dbs, template_path):
        """Empty AuditDB should produce N/A values, not an exception."""
        from scripts.generate_psmf_report import render_psmf
        audit_db, learning_db = dbs
        now = datetime.now(timezone.utc)
        report = render_psmf(
            template_path=template_path,
            audit_db=audit_db,
            learning_db=learning_db,
            owner="TEST-QPPV",
            now=now,
        )
        assert "N/A" in report

    def test_agent_stats_pass_rate_calculation(self, dbs):
        """get_agent_stats: 8 SUCCESS / 10 total = 80.0%"""
        from infra.audit_db import AuditDB as _ADB
        from schemas.audit import AuditLogEntry, AuditStatus
        import hashlib
        audit_db, _ = dbs
        agent_id = "triage-agent-v1"
        for i in range(8):
            e = AuditLogEntry(
                trace_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()),
                review_id=str(uuid.uuid4()), agent_id=agent_id,
                prompt_version="v1", status=AuditStatus.SUCCESS,
                content_hash=hashlib.sha256(f"ok{i}".encode()).hexdigest(),
            )
            audit_db.insert_entry(e)
        for i in range(2):
            e = AuditLogEntry(
                trace_id=str(uuid.uuid4()), run_id=str(uuid.uuid4()),
                review_id=str(uuid.uuid4()), agent_id=agent_id,
                prompt_version="v1", status=AuditStatus.HITL_QUEUED,
                content_hash=hashlib.sha256(f"hq{i}".encode()).hexdigest(),
            )
            audit_db.insert_entry(e)
        stats = audit_db.get_agent_stats(agent_id)
        assert stats["total"]     == 10
        assert stats["success"]   == 8
        assert stats["pass_rate"] == 80.0

    def test_output_file_written_to_correct_directory(self, dbs, template_path, tmp_path):
        """CLI main() should create a PSMF_*.md file in --output-dir."""
        from scripts.generate_psmf_report import main
        audit_db, learning_db = dbs
        out_dir = tmp_path / "psmf_out"
        argv = [
            "--audit-db",    str(audit_db.db_path),
            "--learning-db", str(learning_db.db_path),
            "--template",    str(template_path),
            "--output-dir",  str(out_dir),
            "--owner",       "TEST-QPPV",
        ]
        rc = main(argv)
        assert rc == 0
        files = list(out_dir.glob("PSMF_*.md"))
        assert len(files) == 1
        content = files[0].read_text(encoding="utf-8")
        assert len(content) > 500   # non-trivial content

    def test_cli_owner_arg_substituted(self, dbs, template_path, tmp_path):
        """--owner value must appear in the rendered report."""
        from scripts.generate_psmf_report import main
        audit_db, learning_db = dbs
        out_dir = tmp_path / "psmf_out"
        owner = "Dr. Ada Lovelace, QPPV"
        rc = main([
            "--audit-db",    str(audit_db.db_path),
            "--learning-db", str(learning_db.db_path),
            "--template",    str(template_path),
            "--output-dir",  str(out_dir),
            "--owner",       owner,
        ])
        assert rc == 0
        content = next(out_dir.glob("PSMF_*.md")).read_text(encoding="utf-8")
        assert owner in content

    def test_learning_db_top_patterns_in_report(self, dbs, template_path, tmp_path):
        """When LearningDB has signals, top_patterns should appear in report."""
        from scripts.generate_psmf_report import render_psmf
        audit_db, learning_db = dbs
        # Insert a signal so top_patterns is non-empty
        learning_db.insert_signal(
            trace_id="t1", review_id="r1", stage="TRIAGE",
            agent_id="secondary-qc-auditor-v1",
            prompt_version="v1",
            error_classification="CONFIDENCE_INSUFFICIENT",
            field_path="triage_output",
            original_value="INVALID", corrected_value="VALID",
            correction_rationale="Test", reviewer_id="TEST",
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
        now = datetime.now(timezone.utc)
        report = render_psmf(
            template_path=template_path,
            audit_db=audit_db,
            learning_db=learning_db,
            owner="TEST-QPPV",
            now=now,
        )
        assert "CONFIDENCE_INSUFFICIENT" in report
        assert "triage_output" in report
