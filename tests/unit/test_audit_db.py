"""
tests/unit/test_audit_db.py
============================
Unit tests for infra/audit_db.py and infra/learning_db.py

Tests:
  1.  AuditDB: tables created on init
  2.  AuditDB: insert_entry succeeds
  3.  AuditDB: immutability — UPDATE raises sqlite3.OperationalError
  4.  AuditDB: immutability — DELETE raises sqlite3.OperationalError
  5.  AuditDB: get_entries_for_case returns correct rows
  6.  AuditDB: content_hash length enforced (not 64 chars → insert fails)
  7.  AuditDB: insert_hitl_review succeeds
  8.  AuditDB: hitl_review immutability — UPDATE raises
  9.  LearningDB: tables and views created on init
  10. LearningDB: insert_signal succeeds + pattern upserted
  11. LearningDB: agent_accuracy returns correct count
  12. LearningDB: demographic_bias_report returns rows when data present
  13. LearningDB: immutability — UPDATE raises
"""
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from infra.audit_db import AuditDB
from infra.learning_db import LearningDB
from schemas.audit import AuditLogEntry, AuditStatus, HITLReviewRecord, HITLStage


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def audit_db(tmp_path) -> AuditDB:
    return AuditDB(db_path=tmp_path / "test_audit.db")


@pytest.fixture()
def learning_db(tmp_path) -> LearningDB:
    return LearningDB(db_path=tmp_path / "test_learning.db")


def _make_entry(trace_id="trace-001", agent_id="triage-agent-v1") -> AuditLogEntry:
    return AuditLogEntry(
        trace_id=trace_id,
        run_id="run-001",
        review_id=str(uuid.uuid4()),
        agent_id=agent_id,
        prompt_version="triage-prompt-v1.0",
        status=AuditStatus.SUCCESS,
        content_hash="a" * 64,
        processing_ms=120,
    )


def _make_hitl_review(trace_id="trace-001") -> HITLReviewRecord:
    return HITLReviewRecord(
        review_id=str(uuid.uuid4()),
        trace_id=trace_id,
        stage=HITLStage.TRIAGE,
        reviewer_id="MED-REVIEWER-01",
        original_value={"status": "VALID"},
        corrected_value={"status": "INVALID"},
        correction_rationale="Reporter information is not identifiable in the narrative text.",
        approved=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. AuditDB tables created
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_db_tables_created(audit_db, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "audit_log" in tables
    assert "hitl_review_log" in tables


# ─────────────────────────────────────────────────────────────────────────────
# 2. insert_entry succeeds
# ─────────────────────────────────────────────────────────────────────────────

def test_insert_entry_succeeds(audit_db):
    entry = _make_entry()
    audit_db.insert_entry(entry)
    assert audit_db.count_entries() == 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. Immutability: UPDATE raises
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_immutability_update_raises(audit_db, tmp_path):
    entry = _make_entry()
    audit_db.insert_entry(entry)

    conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
    with pytest.raises(sqlite3.IntegrityError, match="21CFR11"):
        conn.execute(
            "UPDATE audit_log SET agent_id='tampered' WHERE entry_id=?",
            (entry.entry_id,)
        )
        conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Immutability: DELETE raises
# ─────────────────────────────────────────────────────────────────────────────

def test_audit_immutability_delete_raises(audit_db, tmp_path):
    entry = _make_entry()
    audit_db.insert_entry(entry)

    conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
    with pytest.raises(sqlite3.IntegrityError, match="21CFR11"):
        conn.execute(
            "DELETE FROM audit_log WHERE entry_id=?",
            (entry.entry_id,)
        )
        conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 5. get_entries_for_case returns correct rows
# ─────────────────────────────────────────────────────────────────────────────

def test_get_entries_for_case(audit_db):
    e1 = _make_entry(trace_id="trace-001", agent_id="triage-agent-v1")
    e2 = _make_entry(trace_id="trace-001", agent_id="extraction-agent-v1")
    e3 = _make_entry(trace_id="trace-999", agent_id="triage-agent-v1")
    for e in [e1, e2, e3]:
        audit_db.insert_entry(e)

    entries = audit_db.get_entries_for_case("trace-001")
    assert len(entries) == 2
    agent_ids = {e["agent_id"] for e in entries}
    assert agent_ids == {"triage-agent-v1", "extraction-agent-v1"}


# ─────────────────────────────────────────────────────────────────────────────
# 6. content_hash length CHECK constraint enforced
# ─────────────────────────────────────────────────────────────────────────────

def test_content_hash_check_constraint(audit_db, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """INSERT INTO audit_log
               (entry_id, trace_id, run_id, review_id, agent_id, prompt_version,
                model_name, model_temp, status, content_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            ("bad-id", "t1", "r1", "rv1", "agent-v1", "prompt-v1",
             "llama3", 0.0, "SUCCESS", "tooshort")  # < 64 chars
        )
        conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 7. insert_hitl_review succeeds
# ─────────────────────────────────────────────────────────────────────────────

def test_insert_hitl_review_succeeds(audit_db):
    review = _make_hitl_review()
    audit_db.insert_hitl_review(review)
    reviews = audit_db.get_hitl_reviews_for_case("trace-001")
    assert len(reviews) == 1
    assert reviews[0]["stage"] == "TRIAGE"


# ─────────────────────────────────────────────────────────────────────────────
# 8. HITL review immutability: UPDATE raises
# ─────────────────────────────────────────────────────────────────────────────

def test_hitl_review_immutability_raises(audit_db, tmp_path):
    review = _make_hitl_review()
    audit_db.insert_hitl_review(review)

    conn = sqlite3.connect(str(tmp_path / "test_audit.db"))
    with pytest.raises(sqlite3.IntegrityError, match="21CFR11"):
        conn.execute(
            "UPDATE hitl_review_log SET approved=1 WHERE review_id=?",
            (review.review_id,)
        )
        conn.commit()
    conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 9. LearningDB tables created
# ─────────────────────────────────────────────────────────────────────────────

def test_learning_db_tables_created(learning_db, tmp_path):
    conn = sqlite3.connect(str(tmp_path / "test_learning.db"))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "learning_signals" in tables
    assert "signal_patterns" in tables


# ─────────────────────────────────────────────────────────────────────────────
# 10. LearningDB insert_signal + pattern upsert
# ─────────────────────────────────────────────────────────────────────────────

def test_learning_db_insert_signal_and_pattern(learning_db):
    for _ in range(3):
        learning_db.insert_signal(
            trace_id="trace-001",
            review_id="rv-001",
            stage="QC",
            agent_id="qc-agent-v1",
            prompt_version="qc-prompt-v1.0",
            error_classification="HALLUCINATION",
            field_path="suspect_drugs[0].drug_name",
            original_value="Placebix",
            corrected_value="Amoxicillin",
            correction_rationale="Drug name hallucinated; correct drug is Amoxicillin per narrative.",
            reviewer_id="MED-REVIEWER-01",
            reviewed_at=datetime.now(timezone.utc).isoformat(),
        )
    assert learning_db.count_signals() == 3
    patterns = learning_db.top_patterns(limit=5)
    assert any(p["error_classification"] == "HALLUCINATION" for p in patterns)
    assert patterns[0]["count"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 11. agent_accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test_learning_db_agent_accuracy(learning_db):
    learning_db.insert_signal(
        trace_id="t1", review_id="rv1", stage="EXTRACTION",
        agent_id="extraction-agent-v1", prompt_version="extraction-prompt-v1.0",
        error_classification="OMISSION", field_path="verbatim_events[0]",
        original_value=None, corrected_value="Rash",
        correction_rationale="Event omitted from extraction despite being in narrative.",
        reviewer_id="MED-REVIEWER-01",
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    acc = learning_db.agent_accuracy("extraction-agent-v1")
    assert acc["total_corrections"] == 1
    assert acc["by_classification"]["OMISSION"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# 12. demographic_bias_report returns rows when data present
# ─────────────────────────────────────────────────────────────────────────────

def test_demographic_bias_report(learning_db):
    learning_db.insert_signal(
        trace_id="t1", review_id="rv1", stage="CAUSALITY",
        agent_id="causality-agent-v1", prompt_version="causality-prompt-v1.0",
        error_classification="POLARITY_FLIP", field_path="causality_term",
        original_value="Related", corrected_value="Not related",
        correction_rationale="Alternative etiology (latex allergy) is primary cause.",
        reviewer_id="MED-REVIEWER-01",
        reviewed_at=datetime.now(timezone.utc).isoformat(),
        patient_ethnicity="Asian",
        patient_sex="Female",
        patient_age_group="ADULT",
    )
    report = learning_db.demographic_bias_report()
    assert len(report) >= 1
    assert report[0]["patient_ethnicity"] == "Asian"


# ─────────────────────────────────────────────────────────────────────────────
# 13. LearningDB immutability: UPDATE raises
# ─────────────────────────────────────────────────────────────────────────────

def test_learning_db_immutability_raises(learning_db, tmp_path):
    learning_db.insert_signal(
        trace_id="t1", review_id="rv1", stage="QC",
        agent_id="qc-agent-v1", prompt_version="qc-prompt-v1.0",
        error_classification="HALLUCINATION", field_path=None,
        original_value="x", corrected_value="y",
        correction_rationale="Test correction.",
        reviewer_id="MED-REVIEWER-01",
        reviewed_at=datetime.now(timezone.utc).isoformat(),
    )
    conn = sqlite3.connect(str(tmp_path / "test_learning.db"))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE learning_signals SET reviewer_id='hacked'")
        conn.commit()
    conn.close()
