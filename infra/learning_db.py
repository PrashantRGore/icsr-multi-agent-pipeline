"""
infra/learning_db.py
====================
HITL Learning Pipeline database.

Stores correction signals from human reviewers for pattern analysis and
bias monitoring (CIOMS WG XIV Principle 6 — Fairness and Equity).

Tables:
  learning_signals  — one row per agent field corrected by a reviewer
  signal_patterns   — aggregated correction patterns by agent + field + classification

Demographic columns (patient_ethnicity, patient_sex, patient_age_group) enable
stratified analysis to detect systematic bias (e.g., lower confidence for
pediatric cases or specific ethnic groups).

Design:
  - Append-only (no UPDATE/DELETE on learning_signals — immutability preserved)
  - pattern_summary view enables fast dashboard queries
  - agent_accuracy() returns per-agent accuracy from correction history
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_LEARNING_SIGNALS = """
CREATE TABLE IF NOT EXISTS learning_signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id            TEXT    NOT NULL,
    review_id           TEXT    NOT NULL,
    stage               TEXT    NOT NULL,
    agent_id            TEXT    NOT NULL,
    prompt_version      TEXT    NOT NULL,
    error_classification TEXT   NOT NULL,
    field_path          TEXT,
    original_value      TEXT,
    corrected_value     TEXT,
    correction_rationale TEXT,
    reviewer_id         TEXT    NOT NULL,
    reviewed_at         TEXT    NOT NULL,
    -- CIOMS WG XIV Principle 6: demographic stratification for bias monitoring
    patient_ethnicity   TEXT,
    patient_sex         TEXT,
    patient_age_group   TEXT CHECK(patient_age_group IN ('PEDIATRIC', 'ADULT', 'ELDERLY', NULL))
);
"""

_DDL_LEARNING_IMMUTABLE_UPDATE = """
CREATE TRIGGER IF NOT EXISTS learning_signals_immutable_update
BEFORE UPDATE ON learning_signals
BEGIN
    SELECT RAISE(ABORT, 'learning_signals rows are immutable — UPDATE is prohibited.');
END;
"""

_DDL_LEARNING_IMMUTABLE_DELETE = """
CREATE TRIGGER IF NOT EXISTS learning_signals_immutable_delete
BEFORE DELETE ON learning_signals
BEGIN
    SELECT RAISE(ABORT, 'learning_signals rows are immutable — DELETE is prohibited.');
END;
"""

_DDL_SIGNAL_PATTERNS = """
CREATE TABLE IF NOT EXISTS signal_patterns (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id             TEXT    NOT NULL,
    error_classification TEXT    NOT NULL,
    field_path           TEXT,
    count                INTEGER NOT NULL DEFAULT 1,
    last_seen            TEXT    NOT NULL,
    UNIQUE(agent_id, error_classification, field_path)
);
"""

_DDL_PATTERN_VIEW = """
CREATE VIEW IF NOT EXISTS pattern_summary AS
SELECT
    agent_id,
    error_classification,
    field_path,
    count,
    last_seen
FROM signal_patterns
ORDER BY count DESC;
"""

_DDL_DEMOGRAPHIC_VIEW = """
CREATE VIEW IF NOT EXISTS demographic_bias_summary AS
SELECT
    agent_id,
    patient_sex,
    patient_ethnicity,
    patient_age_group,
    error_classification,
    COUNT(*) as correction_count
FROM learning_signals
WHERE patient_sex IS NOT NULL OR patient_ethnicity IS NOT NULL OR patient_age_group IS NOT NULL
GROUP BY agent_id, patient_sex, patient_ethnicity, patient_age_group, error_classification
ORDER BY correction_count DESC;
"""

_DDL_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_ls_trace ON learning_signals (trace_id);
CREATE INDEX IF NOT EXISTS idx_ls_agent ON learning_signals (agent_id);
CREATE INDEX IF NOT EXISTS idx_ls_error ON learning_signals (error_classification);
CREATE INDEX IF NOT EXISTS idx_ls_ethnicity ON learning_signals (patient_ethnicity);
"""


# ─────────────────────────────────────────────────────────────────────────────
# LearningDB class
# ─────────────────────────────────────────────────────────────────────────────

class LearningDB:
    """HITL Learning Pipeline database — append-only correction signal store."""

    def __init__(self, db_path: str | Path = "audit/learning.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            for ddl in [
                _DDL_LEARNING_SIGNALS,
                _DDL_LEARNING_IMMUTABLE_UPDATE,
                _DDL_LEARNING_IMMUTABLE_DELETE,
                _DDL_SIGNAL_PATTERNS,
            ]:
                conn.execute(ddl)
            # Views are idempotent
            try:
                conn.execute(_DDL_PATTERN_VIEW)
            except sqlite3.OperationalError:
                pass
            try:
                conn.execute(_DDL_DEMOGRAPHIC_VIEW)
            except sqlite3.OperationalError:
                pass
            for stmt in _DDL_INDEXES.strip().split("\n"):
                if stmt.strip():
                    conn.execute(stmt.strip())
        logger.info("LearningDB initialized at %s", self.db_path)

    # ── Write ────────────────────────────────────────────────────────────────

    def insert_signal(
        self,
        trace_id:             str,
        review_id:            str,
        stage:                str,
        agent_id:             str,
        prompt_version:       str,
        error_classification: str,
        field_path:           Optional[str],
        original_value:       object,
        corrected_value:      object,
        correction_rationale: str,
        reviewer_id:          str,
        reviewed_at:          str,
        patient_ethnicity:    Optional[str] = None,
        patient_sex:          Optional[str] = None,
        patient_age_group:    Optional[str] = None,
    ) -> None:
        """Insert one learning signal. Updates signal_patterns aggregate."""
        sql_signal = """
        INSERT INTO learning_signals
            (trace_id, review_id, stage, agent_id, prompt_version,
             error_classification, field_path, original_value, corrected_value,
             correction_rationale, reviewer_id, reviewed_at,
             patient_ethnicity, patient_sex, patient_age_group)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """
        sql_upsert_pattern = """
        INSERT INTO signal_patterns (agent_id, error_classification, field_path, count, last_seen)
        VALUES (?, ?, ?, 1, ?)
        ON CONFLICT(agent_id, error_classification, field_path)
        DO UPDATE SET count = count + 1, last_seen = excluded.last_seen
        """
        with self._connect() as conn:
            conn.execute(sql_signal, (
                trace_id, review_id, stage, agent_id, prompt_version,
                error_classification, field_path,
                json.dumps(original_value, default=str),
                json.dumps(corrected_value, default=str),
                correction_rationale, reviewer_id, reviewed_at,
                patient_ethnicity, patient_sex, patient_age_group,
            ))
            conn.execute(sql_upsert_pattern, (
                agent_id, error_classification, field_path, reviewed_at,
            ))
        logger.debug(
            "LearningDB: signal inserted for agent=%s error=%s",
            agent_id, error_classification
        )

    # ── Read ─────────────────────────────────────────────────────────────────

    def agent_accuracy(self, agent_id: str) -> dict:
        """
        Returns correction statistics for a specific agent.
        Used by the secondary QC auditor and PSMF report generator.
        """
        with self._connect() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM learning_signals WHERE agent_id = ?",
                (agent_id,)
            ).fetchone()[0]
            by_class = conn.execute(
                """SELECT error_classification, COUNT(*) as cnt
                   FROM learning_signals WHERE agent_id = ?
                   GROUP BY error_classification ORDER BY cnt DESC""",
                (agent_id,)
            ).fetchall()
        return {
            "agent_id": agent_id,
            "total_corrections": total,
            "by_classification": {r["error_classification"]: r["cnt"] for r in by_class},
        }

    def top_patterns(self, limit: int = 20) -> list[dict]:
        """Return top recurring error patterns across all agents."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pattern_summary LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def demographic_bias_report(self) -> list[dict]:
        """
        Returns stratified correction counts by demographic group.
        CIOMS WG XIV Principle 6 compliance.
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM demographic_bias_summary").fetchall()
        return [dict(r) for r in rows]

    def count_signals(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM learning_signals").fetchone()[0]
