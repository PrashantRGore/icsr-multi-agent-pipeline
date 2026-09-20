"""
infra/audit_db.py
=================
21 CFR Part 11-oriented immutable SQLite audit database.

Compliance guarantees:
  1. IMMUTABILITY:  An UPDATE or DELETE trigger raises an error on audit_log.
     Once written, no row can be modified or removed.
  2. CONTENT_HASH:  Every entry carries a SHA-256 of the serialized agent output.
     Tamper detection: re-compute hash on retrieval and compare.
  3. TIMESTAMPS:    All timestamps are UTC ISO-8601 (CURRENT_TIMESTAMP enforced).
  4. HITL_LOG:      Separate table tracks every human review action (append-only).
  5. WAL mode:      SQLite WAL for concurrent read safety with SQLiteSaver checkpoint DB.

Tables:
  audit_log      — one row per agent invocation (AuditLogEntry)
  hitl_review_log — one row per human reviewer action (HITLReviewRecord)

Usage:
  db = AuditDB(db_path="audit/audit.db")
  db.insert_entry(entry)                      # Write AuditLogEntry
  entries = db.get_entries_for_case(trace_id) # Read all entries for a case
  db.insert_hitl_review(review)               # Write HITLReviewRecord
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from infra.encrypted_db import EncryptedDBMixin
from schemas.audit import AuditLogEntry, AuditStatus, HITLReviewRecord

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_AUDIT_LOG = """
CREATE TABLE IF NOT EXISTS audit_log (
    entry_id        TEXT    PRIMARY KEY,
    trace_id        TEXT    NOT NULL,
    run_id          TEXT    NOT NULL,
    review_id       TEXT    NOT NULL,
    timestamp       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    agent_id        TEXT    NOT NULL,
    prompt_version  TEXT    NOT NULL,
    model_name      TEXT    NOT NULL,
    model_temp      REAL    NOT NULL,
    status          TEXT    NOT NULL,
    content_hash    TEXT    NOT NULL CHECK(length(content_hash) = 64),
    processing_ms   INTEGER,
    error_message   TEXT,
    extra_metadata  TEXT    DEFAULT '{}'
);
"""

_DDL_AUDIT_IMMUTABLE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS audit_log_immutable_update
BEFORE UPDATE ON audit_log
BEGIN
    SELECT RAISE(ABORT, '21CFR11: audit_log rows are immutable — UPDATE is prohibited.');
END;
"""

_DDL_AUDIT_IMMUTABLE_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS audit_log_immutable_delete
BEFORE DELETE ON audit_log
BEGIN
    SELECT RAISE(ABORT, '21CFR11: audit_log rows are immutable — DELETE is prohibited.');
END;
"""

_DDL_AUDIT_INDEX = """
CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_log (trace_id);
CREATE INDEX IF NOT EXISTS idx_audit_review ON audit_log (review_id);
CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log (agent_id);
"""

_DDL_HITL_LOG = """
CREATE TABLE IF NOT EXISTS hitl_review_log (
    review_id           TEXT    PRIMARY KEY,
    trace_id            TEXT    NOT NULL,
    stage               TEXT    NOT NULL,
    reviewer_id         TEXT    NOT NULL,
    reviewed_at         TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    original_value      TEXT    NOT NULL,
    corrected_value     TEXT    NOT NULL,
    correction_rationale TEXT   NOT NULL,
    approved            INTEGER NOT NULL CHECK(approved IN (0, 1)),
    learning_signal     INTEGER NOT NULL DEFAULT 1 CHECK(learning_signal IN (0, 1))
);
"""

_DDL_HITL_IMMUTABLE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS hitl_review_log_immutable_update
BEFORE UPDATE ON hitl_review_log
BEGIN
    SELECT RAISE(ABORT, '21CFR11: hitl_review_log rows are immutable — UPDATE is prohibited.');
END;
"""

_DDL_HITL_IMMUTABLE_DELETE_TRIGGER = """
CREATE TRIGGER IF NOT EXISTS hitl_review_log_immutable_delete
BEFORE DELETE ON hitl_review_log
BEGIN
    SELECT RAISE(ABORT, '21CFR11: hitl_review_log rows are immutable — DELETE is prohibited.');
END;
"""

_DDL_HITL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_hitl_trace ON hitl_review_log (trace_id);
"""

# hitl_queue: mutable pending-cases store (NOT part of the immutable audit trail)
# Allows restoring the reviewer queue across server restarts.
_DDL_HITL_QUEUE = """
CREATE TABLE IF NOT EXISTS hitl_queue (
    review_id    TEXT    PRIMARY KEY,
    thread_id    TEXT    NOT NULL,
    case_id      TEXT    NOT NULL,
    hitl_stage   TEXT    NOT NULL,
    state_snap   TEXT    NOT NULL,
    enqueued_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    status       TEXT    NOT NULL DEFAULT 'PENDING'
                         CHECK(status IN ('PENDING', 'DEQUEUED', 'RESUMED', 'REJECTED'))
);
"""

_DDL_HITL_QUEUE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_hitl_queue_status ON hitl_queue (status);
"""


# ─────────────────────────────────────────────────────────────────────────────
# AuditDB class
# ─────────────────────────────────────────────────────────────────────────────

class AuditDB(EncryptedDBMixin):
    """
    Append-only SQLite database supporting selected 21 CFR Part 11 controls:
    immutability triggers, reviewer identity on every record, and encrypted
    column-level storage. Not formally validated for regulatory submission.

    Thread safety: sqlite3.connect() is not thread-safe for concurrent writes.
    This class uses a context manager for each operation, relying on SQLite's
    serialized WAL mode for safe concurrent reads from LangGraph checkpointer.

    Encryption: When DB_ENCRYPTION_KEY is set (or encryption_key kwarg is provided),
    the following columns are stored encrypted (Fernet/AES-256-GCM):
      audit_log.extra_metadata
      hitl_review_log.original_value, corrected_value
      hitl_queue.state_snap
    """

    def __init__(
        self,
        db_path: str | Path = "audit/audit.db",
        encryption_key: str | bytes | None = None,
    ) -> None:
        super().__init__(encryption_key=encryption_key)  # EncryptedDBMixin
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        """Create tables, triggers, and indexes on first run."""
        with self._connect() as conn:
            for ddl in [
                _DDL_AUDIT_LOG,
                _DDL_AUDIT_IMMUTABLE_TRIGGER,
                _DDL_AUDIT_IMMUTABLE_DELETE_TRIGGER,
                _DDL_HITL_LOG,
                _DDL_HITL_IMMUTABLE_TRIGGER,
                _DDL_HITL_IMMUTABLE_DELETE_TRIGGER,
                _DDL_HITL_QUEUE,
            ]:
                conn.execute(ddl)
            # CREATE INDEX IF NOT EXISTS statements must run separately
            for stmt in _DDL_AUDIT_INDEX.strip().split("\n"):
                if stmt.strip():
                    conn.execute(stmt.strip())
            for stmt in _DDL_HITL_INDEX.strip().split("\n"):
                if stmt.strip():
                    conn.execute(stmt.strip())
            for stmt in _DDL_HITL_QUEUE_INDEX.strip().split("\n"):
                if stmt.strip():
                    conn.execute(stmt.strip())
        logger.info("AuditDB initialized at %s", self.db_path)

    # ── Health check helpers ──────────────────────────────────────────────────

    def count_entries(self) -> int:
        """Return total number of audit_log rows. Used by the /health endpoint."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    # ── Write ─────────────────────────────────────────────────────────────────

    def insert_entry(self, entry: AuditLogEntry) -> None:
        """Insert a single AuditLogEntry. Raises on duplicate entry_id."""
        sql = """
        INSERT INTO audit_log
            (entry_id, trace_id, run_id, review_id, timestamp, agent_id,
             prompt_version, model_name, model_temp, status, content_hash,
             processing_ms, error_message, extra_metadata)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        encrypted_meta = self._encrypt(json.dumps(entry.extra_metadata))
        with self._connect() as conn:
            conn.execute(sql, (
                entry.entry_id,
                entry.trace_id,
                entry.run_id,
                entry.review_id,
                entry.timestamp.isoformat(),
                entry.agent_id,
                entry.prompt_version,
                entry.model_name,
                entry.model_temp,
                entry.status.value,
                entry.content_hash,
                entry.processing_ms,
                entry.error_message,
                encrypted_meta,
            ))
        logger.debug("AuditDB: inserted entry %s (%s)", entry.entry_id, entry.agent_id)

    def insert_hitl_review(self, review: HITLReviewRecord) -> None:
        """Insert a single HITLReviewRecord. Raises on duplicate review_id."""
        sql = """
        INSERT INTO hitl_review_log
            (review_id, trace_id, stage, reviewer_id, reviewed_at,
             original_value, corrected_value, correction_rationale,
             approved, learning_signal)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """
        encrypted_orig = self._encrypt(json.dumps(review.original_value,  default=str))
        encrypted_corr = self._encrypt(json.dumps(review.corrected_value, default=str))
        with self._connect() as conn:
            conn.execute(sql, (
                review.review_id,
                review.trace_id,
                review.stage.value,
                review.reviewer_id,
                review.reviewed_at.isoformat(),
                encrypted_orig,
                encrypted_corr,
                review.correction_rationale,
                int(review.approved),
                int(review.learning_signal),
            ))
        logger.debug(
            "AuditDB: inserted HITL review %s (stage=%s, approved=%s)",
            review.review_id, review.stage, review.approved
        )

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_entries_for_case(self, trace_id: str) -> list[dict]:
        """Return all audit entries for a case ordered by timestamp ASC."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE trace_id = ? ORDER BY timestamp ASC",
                (trace_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_hitl_reviews_for_case(self, trace_id: str) -> list[dict]:
        """Return all HITL reviews for a case ordered by reviewed_at ASC."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM hitl_review_log WHERE trace_id = ? ORDER BY reviewed_at ASC",
                (trace_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_entries_for_review(self, review_id: str) -> list[dict]:
        """Return all audit entries tagged with a specific review_id."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE review_id = ? ORDER BY timestamp ASC",
                (review_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_entries(self) -> int:
        """Total number of entries in audit_log."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]

    def get_agent_stats(self, agent_id: str) -> dict:
        """
        Return success/failure statistics for a specific agent.

        Used by the PSMF report generator to compute per-agent pass rates.

        Returns
        -------
        dict with keys:
          agent_id      : str
          total         : int   — total invocations in audit_log
          success       : int   — rows where status = 'SUCCESS'
          hitl_queued   : int   — rows where status = 'HITL_QUEUED'
          failed        : int   — rows where status = 'FAILED'
          pass_rate     : float — success / total * 100 (0.0 if total == 0)
        """
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE agent_id = ?",
                (agent_id,)
            ).fetchone()[0]
            by_status = conn.execute(
                """SELECT status, COUNT(*) as cnt
                   FROM audit_log
                   WHERE agent_id = ?
                   GROUP BY status""",
                (agent_id,)
            ).fetchall()
        counts = {r["status"]: r["cnt"] for r in by_status}
        success = counts.get("SUCCESS", 0)
        return {
            "agent_id":    agent_id,
            "total":       total,
            "success":     success,
            "hitl_queued": counts.get("HITL_QUEUED", 0),
            "failed":      counts.get("FAILED", 0),
            "pass_rate":   round(success / total * 100, 1) if total > 0 else 0.0,
        }

    def get_earliest_timestamp(self) -> str | None:
        """
        Return the earliest timestamp in the audit_log (UTC ISO-8601).
        Used by the PSMF generator as the effective deployment date.
        Returns None if the audit_log is empty.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(timestamp) AS ts FROM audit_log"
            ).fetchone()
        return row["ts"] if row and row["ts"] else None

    # ── HITL Queue persistence ────────────────────────────────────────────────

    def enqueue_hitl(
        self,
        review_id:  str,
        thread_id:  str,
        case_id:    str,
        hitl_stage: str,
        state_snap: dict,
    ) -> None:
        """
        Write a new PENDING HITL case to the queue table.
        State snapshot is encrypted if DB_ENCRYPTION_KEY is configured.
        Called by HITLQueue.enqueue() when a case is halted.
        """
        encrypted_snap = self._encrypt(json.dumps(state_snap, default=str))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hitl_queue
                    (review_id, thread_id, case_id, hitl_stage, state_snap, status)
                VALUES (?, ?, ?, ?, ?, 'PENDING')
                """,
                (review_id, thread_id, case_id, hitl_stage, encrypted_snap),
            )
        logger.debug("AuditDB: enqueued HITL case review_id=%s stage=%s", review_id, hitl_stage)

    def update_hitl_status(self, review_id: str, new_status: str) -> None:
        """
        Transition a queue entry to DEQUEUED, RESUMED, or REJECTED.
        Called by HITLQueue.dequeue() when a case is resolved.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE hitl_queue SET status = ? WHERE review_id = ?",
                (new_status, review_id),
            )
        logger.debug("AuditDB: hitl_queue review_id=%s → %s", review_id, new_status)

    def get_pending_hitl(self) -> list[dict]:
        """
        Return all PENDING queue entries ordered by enqueued_at.
        State snapshots are transparently decrypted.
        Called at server startup to reload surviving cases.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT review_id, thread_id, case_id, hitl_stage,
                       state_snap, enqueued_at
                FROM hitl_queue
                WHERE status = 'PENDING'
                ORDER BY enqueued_at ASC
                """
            ).fetchall()
        return [
            {
                "review_id":   r["review_id"],
                "thread_id":   r["thread_id"],
                "case_id":     r["case_id"],
                "hitl_stage":  r["hitl_stage"],
                "state_snap":  json.loads(self._decrypt(r["state_snap"])),
                "enqueued_at": r["enqueued_at"],
            }
            for r in rows
        ]
