"""
tests/unit/test_auth_db.py
===========================
Unit tests for infra/auth_db.py — reviewer API-key store.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from infra.auth_db import AuthDB, ReviewerRecord


@pytest.fixture
def db(tmp_path: Path) -> AuthDB:
    """Fresh AuthDB backed by a temporary file (not :memory: — bcrypt needs persistence)."""
    return AuthDB(db_path=tmp_path / "auth.db")


class TestCreateReviewer:
    def test_returns_nonempty_key(self, db: AuthDB) -> None:
        key = db.create_reviewer("QPPV-01", role="QPPV")
        assert isinstance(key, str)
        assert len(key) >= 40

    def test_key_not_stored_in_plaintext(self, db: AuthDB) -> None:
        key = db.create_reviewer("QPPV-01")
        # The raw key must not appear in the DB (only its hash should)
        import sqlite3
        conn = sqlite3.connect(str(db._path))
        rows = conn.execute("SELECT key_hash FROM reviewers").fetchall()
        conn.close()
        assert not any(key in str(row) for row in rows)

    def test_duplicate_reviewer_id_raises(self, db: AuthDB) -> None:
        db.create_reviewer("DUP-01")
        with pytest.raises(Exception):
            db.create_reviewer("DUP-01")

    def test_role_stored_correctly(self, db: AuthDB) -> None:
        db.create_reviewer("ADMIN-01", role="ADMIN")
        reviewers = db.list_active()
        assert any(r.reviewer_id == "ADMIN-01" and r.role == "ADMIN" for r in reviewers)


class TestAuthenticate:
    def test_valid_key_returns_record(self, db: AuthDB) -> None:
        key = db.create_reviewer("MED-01")
        result = db.authenticate(key)
        assert result is not None
        assert isinstance(result, ReviewerRecord)
        assert result.reviewer_id == "MED-01"
        assert result.active is True

    def test_wrong_key_returns_none(self, db: AuthDB) -> None:
        db.create_reviewer("MED-02")
        result = db.authenticate("totally-wrong-key")
        assert result is None

    def test_empty_key_returns_none(self, db: AuthDB) -> None:
        db.create_reviewer("MED-03")
        assert db.authenticate("") is None

    def test_last_used_at_updated_on_auth(self, db: AuthDB) -> None:
        key = db.create_reviewer("MED-04")
        result = db.authenticate(key)
        assert result is not None
        assert result.last_used_at is not None

    def test_deactivated_key_returns_none(self, db: AuthDB) -> None:
        key = db.create_reviewer("MED-05")
        db.deactivate("MED-05")
        assert db.authenticate(key) is None


class TestDeactivate:
    def test_deactivate_removes_from_active(self, db: AuthDB) -> None:
        db.create_reviewer("EX-01")
        db.deactivate("EX-01")
        active = [r.reviewer_id for r in db.list_active()]
        assert "EX-01" not in active

    def test_deactivate_does_not_delete_row(self, db: AuthDB) -> None:
        """Row must remain for audit trail — only active=0 is set."""
        db.create_reviewer("EX-02")
        db.deactivate("EX-02")
        import sqlite3
        conn = sqlite3.connect(str(db._path))
        row = conn.execute(
            "SELECT active FROM reviewers WHERE reviewer_id = ?", ("EX-02",)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 0  # deactivated, not deleted


class TestRotateKey:
    def test_old_key_invalid_after_rotate(self, db: AuthDB) -> None:
        old_key = db.create_reviewer("ROT-01")
        db.rotate_key("ROT-01")
        assert db.authenticate(old_key) is None

    def test_new_key_valid_after_rotate(self, db: AuthDB) -> None:
        db.create_reviewer("ROT-02")
        new_key = db.rotate_key("ROT-02")
        result = db.authenticate(new_key)
        assert result is not None
        assert result.reviewer_id == "ROT-02"


class TestListActive:
    def test_empty_db_returns_empty_list(self, db: AuthDB) -> None:
        assert db.list_active() == []

    def test_multiple_reviewers_all_listed(self, db: AuthDB) -> None:
        db.create_reviewer("R-01")
        db.create_reviewer("R-02")
        ids = [r.reviewer_id for r in db.list_active()]
        assert "R-01" in ids and "R-02" in ids

    def test_deactivated_not_in_list(self, db: AuthDB) -> None:
        db.create_reviewer("R-03")
        db.create_reviewer("R-04")
        db.deactivate("R-03")
        ids = [r.reviewer_id for r in db.list_active()]
        assert "R-03" not in ids
        assert "R-04" in ids
