"""
tests/unit/test_encrypted_db.py
=================================
Unit tests for infra/encrypted_db.py (EncryptedDBMixin).

Tests cover:
  - plaintext pass-through when no key is configured
  - Fernet encrypt / decrypt round-trip
  - _CIPHER_PREFIX detection
  - Legacy plaintext migration (value without prefix decrypts cleanly)
  - encryption_active property
  - Invalid key → graceful fallback to plaintext
  - None value handling in _decrypt()
"""
from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from infra.encrypted_db import EncryptedDBMixin, _CIPHER_PREFIX


# ─────────────────────────────────────────────────────────────────────────────
# Minimal concrete class for testing the mixin
# ─────────────────────────────────────────────────────────────────────────────

class _TestDB(EncryptedDBMixin):
    """Thin concrete subclass — no real DB, just the encryption logic."""
    def __init__(self, encryption_key=None):
        super().__init__(encryption_key=encryption_key)


@pytest.fixture
def key() -> str:
    return Fernet.generate_key().decode()


@pytest.fixture
def db_plain() -> _TestDB:
    """No encryption key — plaintext mode."""
    return _TestDB(encryption_key=None)


@pytest.fixture
def db_encrypted(key: str) -> _TestDB:
    """Fernet encryption enabled."""
    return _TestDB(encryption_key=key)


# ─────────────────────────────────────────────────────────────────────────────
# Plaintext mode (no key)
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaintextMode:
    def test_encrypt_returns_bytes(self, db_plain: _TestDB) -> None:
        result = db_plain._encrypt("hello")
        assert isinstance(result, bytes)

    def test_encrypt_no_cipher_prefix(self, db_plain: _TestDB) -> None:
        result = db_plain._encrypt("hello")
        assert not result.startswith(_CIPHER_PREFIX)

    def test_decrypt_roundtrip(self, db_plain: _TestDB) -> None:
        raw = db_plain._encrypt("patient narrative text")
        assert db_plain._decrypt(raw) == "patient narrative text"

    def test_encryption_active_false(self, db_plain: _TestDB) -> None:
        assert db_plain.encryption_active is False

    def test_decrypt_none_returns_empty(self, db_plain: _TestDB) -> None:
        assert db_plain._decrypt(None) == ""

    def test_decrypt_plain_string_input(self, db_plain: _TestDB) -> None:
        assert db_plain._decrypt("already plain") == "already plain"


# ─────────────────────────────────────────────────────────────────────────────
# Encrypted mode (Fernet key provided)
# ─────────────────────────────────────────────────────────────────────────────

class TestEncryptedMode:
    def test_encryption_active_true(self, db_encrypted: _TestDB) -> None:
        assert db_encrypted.encryption_active is True

    def test_encrypt_has_cipher_prefix(self, db_encrypted: _TestDB) -> None:
        ct = db_encrypted._encrypt("sensitive")
        assert ct.startswith(_CIPHER_PREFIX)

    def test_encrypt_decrypt_roundtrip(self, db_encrypted: _TestDB) -> None:
        original = '{"patient_name": "John Smith", "dob": "1980-01-01"}'
        ct = db_encrypted._encrypt(original)
        assert db_encrypted._decrypt(ct) == original

    def test_different_encryptions_of_same_value(self, db_encrypted: _TestDB) -> None:
        """Fernet uses a random IV — same plaintext produces different ciphertext each time."""
        ct1 = db_encrypted._encrypt("hello")
        ct2 = db_encrypted._encrypt("hello")
        assert ct1 != ct2

    def test_unicode_text_survives_encryption(self, db_encrypted: _TestDB) -> None:
        text = "Patiënt: François Müller, né le 12/03/1975"
        assert db_encrypted._decrypt(db_encrypted._encrypt(text)) == text

    def test_json_payload_survives_encryption(self, db_encrypted: _TestDB) -> None:
        import json
        payload = {"case_id": "ICSR-20240115-001", "hitl_stage": "QC", "nested": [1, 2, 3]}
        serialized = json.dumps(payload)
        decrypted = db_encrypted._decrypt(db_encrypted._encrypt(serialized))
        assert json.loads(decrypted) == payload

    def test_decrypt_none_returns_empty(self, db_encrypted: _TestDB) -> None:
        assert db_encrypted._decrypt(None) == ""

    def test_decrypt_legacy_plaintext(self, db_encrypted: _TestDB) -> None:
        """
        Values written before encryption was enabled (no _CIPHER_PREFIX)
        must still be readable — migration safety.
        """
        legacy_bytes = b'{"old": "value"}'    # No _CIPHER_PREFIX
        result = db_encrypted._decrypt(legacy_bytes)
        assert result == '{"old": "value"}'


# ─────────────────────────────────────────────────────────────────────────────
# Key rotation / invalid key
# ─────────────────────────────────────────────────────────────────────────────

class TestKeyHandling:
    def test_invalid_key_falls_back_to_plaintext(self) -> None:
        """
        If the provided key is malformed, EncryptedDBMixin should log an error
        and fall back to plaintext mode rather than crashing.
        """
        db = _TestDB(encryption_key="not-a-valid-fernet-key")
        assert db.encryption_active is False
        # Should still encrypt/decrypt (plaintext mode)
        ct = db._encrypt("hello")
        assert db._decrypt(ct) == "hello"

    def test_bytes_key_accepted(self) -> None:
        key_bytes = Fernet.generate_key()
        db = _TestDB(encryption_key=key_bytes)
        assert db.encryption_active is True
        assert db._decrypt(db._encrypt("works")) == "works"


# ─────────────────────────────────────────────────────────────────────────────
# AuditDB integration — encrypted write-through
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditDBEncryption:
    """
    Smoke-tests that AuditDB correctly encrypts/decrypts sensitive columns.
    Uses a real temp SQLite file.
    """

    def test_enqueue_and_reload_with_encryption(self, tmp_path, key: str) -> None:
        """
        enqueue_hitl() must encrypt state_snap and get_pending_hitl() must
        decrypt it transparently, including across a DB reconnect.
        """
        import json
        import uuid
        from infra.audit_db import AuditDB

        db_path = tmp_path / "enc_audit.db"
        snap    = {"case_id": "ICSR-001", "patient": "Jane Doe"}
        rid     = str(uuid.uuid4())

        # Write with encryption
        db_write = AuditDB(db_path=db_path, encryption_key=key)
        db_write.enqueue_hitl(rid, "thread-1", "ICSR-001", "QC", snap)

        # Verify raw bytes in DB are NOT plaintext JSON
        import sqlite3
        with sqlite3.connect(str(db_path)) as conn:
            raw = conn.execute(
                "SELECT state_snap FROM hitl_queue WHERE review_id=?", (rid,)
            ).fetchone()[0]
        raw_bytes = raw if isinstance(raw, bytes) else raw.encode()
        assert b"Jane Doe" not in raw_bytes, "PII must not appear in plaintext"

        # Read back and verify decryption
        db_read = AuditDB(db_path=db_path, encryption_key=key)
        pending = db_read.get_pending_hitl()
        match = next(r for r in pending if r["review_id"] == rid)
        assert match["state_snap"]["patient"] == "Jane Doe"

    def test_insert_entry_encrypts_extra_metadata(self, tmp_path, key: str) -> None:
        """extra_metadata must be encrypted on disk."""
        import hashlib
        import json
        import sqlite3
        import uuid
        from datetime import datetime, timezone
        from infra.audit_db import AuditDB
        from schemas.audit import AuditLogEntry, AuditStatus

        db_path = tmp_path / "enc_meta.db"
        db = AuditDB(db_path=db_path, encryption_key=key)

        entry = AuditLogEntry(
            entry_id       = str(uuid.uuid4()),
            trace_id       = "trace-001",
            run_id         = str(uuid.uuid4()),
            review_id      = str(uuid.uuid4()),   # must be a string
            timestamp      = datetime.now(timezone.utc),
            agent_id       = "TEST_AGENT",
            prompt_version = "v1",
            model_name     = "test-model",
            model_temp     = 0.0,
            status         = AuditStatus.SUCCESS,
            content_hash   = hashlib.sha256(b"payload").hexdigest(),  # 64-char hex
            processing_ms  = 100,
            error_message  = None,
            extra_metadata = {"patient_id": "P-SECRET-001"},
        )
        db.insert_entry(entry)

        # Raw DB should not contain the secret
        with sqlite3.connect(str(db_path)) as conn:
            raw = conn.execute(
                "SELECT extra_metadata FROM audit_log WHERE entry_id=?",
                (entry.entry_id,)
            ).fetchone()[0]
        raw_bytes = raw if isinstance(raw, bytes) else raw.encode()
        assert b"P-SECRET-001" not in raw_bytes, "patient_id must be encrypted"
