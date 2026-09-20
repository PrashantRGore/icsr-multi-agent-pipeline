"""
infra/auth_db.py
================
Reviewer API-key store supporting 21 CFR Part 11 identity controls.

Design decisions:
  - Separate SQLite DB (audit/auth.db) — never mixed with the immutable audit trail.
    This allows key rotation (UPDATE/DELETE on reviewers table) without risking
    compliance violations on the audit_log table.
  - Keys are stored as bcrypt hashes (cost factor 12) — raw keys are never persisted.
  - reviewer_id is role-based (e.g. "QPPV-01"), not a personal name (data minimisation).
  - last_used_at is updated on every successful authentication for anomaly detection.
  - active flag allows deactivation without deleting records (audit trail of who existed).

21 CFR Part 11 alignment:
  - §11.50: Signed corrections require verified reviewer identity (enforced via API key)
  - §11.300: Access controls — only active key holders can submit corrections

Usage:
  db = AuthDB("audit/auth.db")
  raw_key = db.create_reviewer("QPPV-01", role="QPPV")
  # → Give raw_key to the reviewer; it is never stored again

  reviewer = db.authenticate(raw_key)
  # → ReviewerRecord | None
"""
from __future__ import annotations

import logging
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

import bcrypt

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_REVIEWERS = """
CREATE TABLE IF NOT EXISTS reviewers (
    reviewer_id   TEXT    PRIMARY KEY,
    key_hash      TEXT    NOT NULL UNIQUE,
    role          TEXT    NOT NULL DEFAULT 'REVIEWER',
    active        INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0, 1)),
    created_at    TEXT    NOT NULL,
    last_used_at  TEXT
);
"""

_DDL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_reviewers_active ON reviewers (active);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReviewerRecord:
    """Authenticated reviewer identity (no raw key or hash exposed)."""
    reviewer_id:  str
    role:         str
    active:       bool
    created_at:   str
    last_used_at: str | None


# ─────────────────────────────────────────────────────────────────────────────
# AuthDB
# ─────────────────────────────────────────────────────────────────────────────

class AuthDB:
    """
    Reviewer API-key store backed by SQLite.

    Parameters
    ----------
    db_path : Path or str — location of the auth SQLite file.
    """

    BCRYPT_COST = 12  # ~300ms per hash on modest hardware — brute-force resistant

    def __init__(self, db_path: str | Path = "audit/auth.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info("AuthDB: initialised at %s", self._path)

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL_REVIEWERS + _DDL_INDEX)

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
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

    # ── Write API ─────────────────────────────────────────────────────────────

    def create_reviewer(
        self,
        reviewer_id: str,
        role: str = "REVIEWER",
    ) -> str:
        """
        Create a new reviewer and return the raw API key (shown exactly once).

        The raw key is never stored — only its bcrypt hash is persisted.
        The caller MUST save the returned key securely.

        Parameters
        ----------
        reviewer_id : Role-based identifier (e.g. "QPPV-01", "MED-REVIEWER-02")
        role        : One of REVIEWER, QPPV, ADMIN

        Returns
        -------
        Raw API key string (URL-safe base64, 43 chars)
        """
        raw_key  = secrets.token_urlsafe(32)          # 256-bit entropy
        key_hash = bcrypt.hashpw(
            raw_key.encode(), bcrypt.gensalt(rounds=self.BCRYPT_COST)
        ).decode()
        now = _utcnow()

        with self._connect() as conn:
            conn.execute(
                """INSERT INTO reviewers
                   (reviewer_id, key_hash, role, active, created_at)
                   VALUES (?, ?, ?, 1, ?)""",
                (reviewer_id, key_hash, role, now),
            )

        logger.info("AuthDB: created reviewer_id=%s role=%s", reviewer_id, role)
        return raw_key

    def deactivate(self, reviewer_id: str) -> None:
        """Deactivate a reviewer — they can no longer authenticate."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE reviewers SET active = 0 WHERE reviewer_id = ?",
                (reviewer_id,),
            )
        logger.info("AuthDB: deactivated reviewer_id=%s", reviewer_id)

    def rotate_key(self, reviewer_id: str) -> str:
        """
        Generate a new API key for an existing reviewer (invalidates old key).
        Returns the new raw key.
        """
        raw_key  = secrets.token_urlsafe(32)
        key_hash = bcrypt.hashpw(
            raw_key.encode(), bcrypt.gensalt(rounds=self.BCRYPT_COST)
        ).decode()

        with self._connect() as conn:
            conn.execute(
                "UPDATE reviewers SET key_hash = ? WHERE reviewer_id = ? AND active = 1",
                (key_hash, reviewer_id),
            )
        logger.info("AuthDB: rotated key for reviewer_id=%s", reviewer_id)
        return raw_key

    # ── Read / Auth API ───────────────────────────────────────────────────────

    def authenticate(self, raw_key: str) -> ReviewerRecord | None:
        """
        Validate a raw API key.  Returns None on failure (wrong key, inactive reviewer).

        On success, updates last_used_at for anomaly detection.
        bcrypt.checkpw is intentionally constant-time.
        """
        with self._connect() as conn:
            # Fetch all active reviewers and check bcrypt against each.
            # In practice there will be <=50 reviewers, so this is fast enough.
            rows = conn.execute(
                "SELECT reviewer_id, key_hash, role, created_at, last_used_at "
                "FROM reviewers WHERE active = 1"
            ).fetchall()

        for row in rows:
            try:
                match = bcrypt.checkpw(raw_key.encode(), row["key_hash"].encode())
            except Exception:
                continue
            if match:
                now = _utcnow()
                try:
                    with self._connect() as conn:
                        conn.execute(
                            "UPDATE reviewers SET last_used_at = ? WHERE reviewer_id = ?",
                            (now, row["reviewer_id"]),
                        )
                except Exception as exc:
                    logger.warning("AuthDB: failed to update last_used_at: %s", exc)

                return ReviewerRecord(
                    reviewer_id  = row["reviewer_id"],
                    role         = row["role"],
                    active       = True,
                    created_at   = row["created_at"],
                    last_used_at = now,
                )

        logger.warning("AuthDB: authentication failed — no matching active key")
        return None

    def list_active(self) -> list[ReviewerRecord]:
        """Return all active reviewer records (no key hashes exposed)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT reviewer_id, role, created_at, last_used_at "
                "FROM reviewers WHERE active = 1 ORDER BY created_at"
            ).fetchall()
        return [
            ReviewerRecord(
                reviewer_id  = r["reviewer_id"],
                role         = r["role"],
                active       = True,
                created_at   = r["created_at"],
                last_used_at = r["last_used_at"],
            )
            for r in rows
        ]

    def count_reviewers(self) -> int:
        """Return total number of active reviewers. Used by the /health endpoint."""
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM reviewers WHERE active = 1"
            ).fetchone()[0]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()
