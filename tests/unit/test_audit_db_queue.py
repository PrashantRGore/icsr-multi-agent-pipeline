"""
tests/unit/test_audit_db_queue.py
===================================
Unit tests for the hitl_queue persistence methods added to AuditDB in Week 2.

Existing audit_log and hitl_review_log tests are in test_audit_db.py.
These tests cover:
  - enqueue_hitl()
  - update_hitl_status()
  - get_pending_hitl() (including restart-reload simulation)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from infra.audit_db import AuditDB


SAMPLE_STATE = {
    "case_id":        "ICSR-20240115-001",
    "trace_id":       "thread-abc",
    "hitl_stage":     "QC",
    "pipeline_halted": True,
    "raw_narrative":  "Patient reported adverse event.",
}


@pytest.fixture
def db(tmp_path: Path) -> AuditDB:
    """Fresh AuditDB backed by a temp file."""
    return AuditDB(db_path=tmp_path / "audit.db")


def _review_id() -> str:
    return str(uuid.uuid4())


class TestEnqueueHitl:
    def test_enqueue_creates_pending_row(self, db: AuditDB) -> None:
        rid = _review_id()
        db.enqueue_hitl(
            review_id  = rid,
            thread_id  = "thread-001",
            case_id    = "CASE-001",
            hitl_stage = "QC",
            state_snap = SAMPLE_STATE,
        )
        rows = db.get_pending_hitl()
        assert any(r["review_id"] == rid for r in rows)

    def test_enqueued_state_snap_roundtrips(self, db: AuditDB) -> None:
        rid = _review_id()
        db.enqueue_hitl(rid, "t-001", "C-001", "QC", SAMPLE_STATE)
        rows = db.get_pending_hitl()
        match = next(r for r in rows if r["review_id"] == rid)
        assert match["state_snap"]["case_id"] == SAMPLE_STATE["case_id"]
        assert match["state_snap"]["hitl_stage"] == SAMPLE_STATE["hitl_stage"]

    def test_enqueue_replace_on_duplicate(self, db: AuditDB) -> None:
        """INSERT OR REPLACE: re-enqueueing same review_id updates the row."""
        rid = _review_id()
        db.enqueue_hitl(rid, "t-001", "C-001", "QC", SAMPLE_STATE)
        updated_state = {**SAMPLE_STATE, "case_id": "CASE-UPDATED"}
        db.enqueue_hitl(rid, "t-001", "C-001", "QC", updated_state)
        rows = db.get_pending_hitl()
        matches = [r for r in rows if r["review_id"] == rid]
        assert len(matches) == 1
        assert matches[0]["state_snap"]["case_id"] == "CASE-UPDATED"


class TestUpdateHitlStatus:
    def test_update_to_dequeued(self, db: AuditDB) -> None:
        rid = _review_id()
        db.enqueue_hitl(rid, "t-002", "C-002", "CAUSALITY", SAMPLE_STATE)
        db.update_hitl_status(rid, "DEQUEUED")
        # Should no longer appear in PENDING list
        pending = [r["review_id"] for r in db.get_pending_hitl()]
        assert rid not in pending

    def test_update_to_resumed(self, db: AuditDB) -> None:
        rid = _review_id()
        db.enqueue_hitl(rid, "t-003", "C-003", "NARRATIVE", SAMPLE_STATE)
        db.update_hitl_status(rid, "RESUMED")
        pending = [r["review_id"] for r in db.get_pending_hitl()]
        assert rid not in pending

    def test_update_to_rejected(self, db: AuditDB) -> None:
        rid = _review_id()
        db.enqueue_hitl(rid, "t-004", "C-004", "TRIAGE", SAMPLE_STATE)
        db.update_hitl_status(rid, "REJECTED")
        pending = [r["review_id"] for r in db.get_pending_hitl()]
        assert rid not in pending


class TestGetPendingHitl:
    def test_empty_db_returns_empty_list(self, db: AuditDB) -> None:
        assert db.get_pending_hitl() == []

    def test_multiple_pending_all_returned(self, db: AuditDB) -> None:
        ids = [_review_id() for _ in range(3)]
        for i, rid in enumerate(ids):
            db.enqueue_hitl(rid, f"t-{i}", f"C-{i}", "QC", SAMPLE_STATE)
        pending_ids = [r["review_id"] for r in db.get_pending_hitl()]
        for rid in ids:
            assert rid in pending_ids

    def test_only_pending_status_returned(self, db: AuditDB) -> None:
        rid_pending  = _review_id()
        rid_dequeued = _review_id()
        db.enqueue_hitl(rid_pending,  "t-p", "C-P", "QC", SAMPLE_STATE)
        db.enqueue_hitl(rid_dequeued, "t-d", "C-D", "QC", SAMPLE_STATE)
        db.update_hitl_status(rid_dequeued, "DEQUEUED")
        pending_ids = [r["review_id"] for r in db.get_pending_hitl()]
        assert rid_pending  in pending_ids
        assert rid_dequeued not in pending_ids

    def test_restart_reload_simulation(self, tmp_path: Path) -> None:
        """
        Simulate server restart: write cases to AuditDB with one connection,
        open a NEW AuditDB instance (new connection), verify cases survive.
        """
        db_path = tmp_path / "restart_test.db"
        rid = _review_id()

        # First connection: enqueue a case
        db1 = AuditDB(db_path=db_path)
        db1.enqueue_hitl(rid, "thread-X", "CASE-X", "CODING", SAMPLE_STATE)

        # Second connection: reload (simulates new server process)
        db2 = AuditDB(db_path=db_path)
        pending = db2.get_pending_hitl()
        assert any(r["review_id"] == rid for r in pending)

    def test_result_contains_required_keys(self, db: AuditDB) -> None:
        rid = _review_id()
        db.enqueue_hitl(rid, "t-k", "C-K", "QC", SAMPLE_STATE)
        rows = db.get_pending_hitl()
        row = next(r for r in rows if r["review_id"] == rid)
        for key in ("review_id", "thread_id", "case_id", "hitl_stage", "state_snap", "enqueued_at"):
            assert key in row
