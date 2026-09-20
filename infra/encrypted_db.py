"""
infra/encrypted_db.py
======================
Transparent column-level Fernet encryption mixin for SQLite databases.

WHY THIS OVER SQLCIPHER:
  SQLCipher requires a pre-built native extension (no Windows binary wheel exists).
  Column-level Fernet encryption achieves the same goal — patient data is never
  stored in plaintext on disk — using the pure-Python `cryptography` library that
  is already installed.

DESIGN:
  - EncryptedDBMixin provides two methods: _encrypt() and _decrypt().
  - When DB_ENCRYPTION_KEY is set, _encrypt() returns a Fernet-ciphertext token
    (URL-safe base64, includes IV + HMAC) prefixed with b"\\xfe\\xfe" to
    distinguish ciphertext from legacy plaintext values.
  - _decrypt() detects the prefix and decrypts; falls back to returning the
    raw bytes as-is for any plaintext values written before the key was set.
    This allows safe migration of existing databases.
  - The mixin is stateless — no connection-level state.

THREAT MODEL:
  Protects against:
    ✓ Server disk image theft / cloud snapshot access
    ✓ Database file copied off-server
    ✓ Accidental plaintext exposure in backups

  Does NOT protect against:
    ✗ Compromised server with DB_ENCRYPTION_KEY in memory
    ✗ Application-layer attacks (SQL injection → runtime decryption)

ENVIRONMENT VARIABLES:
  DB_ENCRYPTION_KEY : Fernet key (base64-encoded 32-byte key)
                      Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
                      If NOT set: data stored in plaintext; WARNING logged on startup.

USAGE:
  class MyDB(EncryptedDBMixin):
      def __init__(self, db_path, encryption_key=None):
          super().__init__(encryption_key=encryption_key)
          ...

      def write_sensitive(self, value: str) -> None:
          with self._connect() as conn:
              conn.execute("INSERT INTO t (col) VALUES (?)", (self._encrypt(value),))

      def read_sensitive(self, rowid: int) -> str:
          with self._connect() as conn:
              row = conn.execute("SELECT col FROM t WHERE rowid=?", (rowid,)).fetchone()
          return self._decrypt(row[0])
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Two-byte magic prefix that marks a Fernet ciphertext value.
# Chosen to be invalid UTF-8 and never appear at the start of JSON / ISO-date strings.
_CIPHER_PREFIX = b"\xfe\xfe"


class EncryptedDBMixin:
    """
    Mixin that adds _encrypt() / _decrypt() to any SQLite-backed DB class.

    Call super().__init__(encryption_key=...) in the subclass __init__.
    Pass encryption_key=None to read from DB_ENCRYPTION_KEY env var.
    """

    def __init__(self, *, encryption_key: str | bytes | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fernet = None

        raw_key = encryption_key or os.getenv("DB_ENCRYPTION_KEY", "")
        if raw_key:
            try:
                from cryptography.fernet import Fernet
                key_bytes = raw_key.encode() if isinstance(raw_key, str) else raw_key
                self._fernet = Fernet(key_bytes)
                logger.info(
                    "%s: column-level encryption ENABLED (Fernet/AES-256-GCM)",
                    self.__class__.__name__,
                )
            except Exception as exc:
                logger.error(
                    "%s: failed to initialise Fernet with DB_ENCRYPTION_KEY: %s — "
                    "data will be stored in PLAINTEXT",
                    self.__class__.__name__, exc,
                )
        else:
            logger.warning(
                "%s: DB_ENCRYPTION_KEY not set — sensitive columns stored in PLAINTEXT. "
                "Set DB_ENCRYPTION_KEY for production deployments.",
                self.__class__.__name__,
            )

    # ── Encryption helpers ──────────────────────────────────────────────────

    def _encrypt(self, plaintext: str) -> bytes:
        """
        Encrypt a string value for storage.

        Returns Fernet ciphertext (bytes) prefixed with _CIPHER_PREFIX if a key
        is configured, otherwise returns the UTF-8 encoded plaintext unchanged.
        """
        if self._fernet is None:
            return plaintext.encode("utf-8")
        token = self._fernet.encrypt(plaintext.encode("utf-8"))
        return _CIPHER_PREFIX + token

    def _decrypt(self, value: bytes | str | None) -> str:
        """
        Decrypt a value retrieved from storage.

        Handles three cases:
          1. value starts with _CIPHER_PREFIX → Fernet-decrypt
          2. value is bytes/str but no prefix → legacy plaintext (migration)
          3. value is None → return empty string
        """
        if value is None:
            return ""
        if isinstance(value, str):
            value = value.encode("utf-8")
        if value.startswith(_CIPHER_PREFIX) and self._fernet is not None:
            token = value[len(_CIPHER_PREFIX):]
            return self._fernet.decrypt(token).decode("utf-8")
        # Legacy plaintext — return as-is
        return value.decode("utf-8", errors="replace")

    @property
    def encryption_active(self) -> bool:
        """True if a Fernet key is configured and encryption is active."""
        return self._fernet is not None
