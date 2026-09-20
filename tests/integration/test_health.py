"""
tests/integration/test_health.py
==================================
Integration tests for the enriched /health endpoint.

Tests the full response structure, sub-check keys, and HTTP status codes
without requiring a live Ollama instance (Ollama is mocked).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import Response as HttpxResponse

from hitl.main import create_app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    Build a TestClient with fully mocked app.state (no real DBs, no Ollama).
    Scoped to module for speed — state is read-only for these tests.
    """
    tmp = tmp_path_factory.mktemp("health_test")

    app = create_app()

    # Patch lifespan so we control app.state manually
    from infra.audit_db import AuditDB
    from infra.auth_db import AuthDB

    audit_db = AuditDB(db_path=tmp / "audit.db")
    auth_db  = AuthDB(db_path=tmp / "auth.db")

    # Mock pipeline with an empty HITL queue
    mock_pipeline = MagicMock()
    mock_pipeline.hitl_queue.__len__ = MagicMock(return_value=0)

    # Inject state directly (bypass lifespan for unit-isolation)
    app.state.audit_db     = audit_db
    app.state.auth_db      = auth_db
    app.state.pipeline     = mock_pipeline
    app.state.deidentifier = None
    app.state._ollama_host  = "http://localhost:11434"
    app.state._ollama_model = "llama3.1:8b-instruct-q4_K_M"

    return TestClient(app, raise_server_exceptions=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthStructure:
    def test_health_returns_200_when_dbs_ok(self, client) -> None:
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            resp = client.get("/health")

        assert resp.status_code == 200

    def test_health_response_has_required_fields(self, client) -> None:
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            body = client.get("/health").json()

        for field in ("status", "version", "checks", "encryption_active", "pii_deidentifier_active"):
            assert field in body, f"Missing field: {field}"

    def test_health_checks_has_expected_subsystems(self, client) -> None:
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            checks = client.get("/health").json()["checks"]

        for subsystem in ("audit_db", "auth_db", "hitl_queue", "ollama"):
            assert subsystem in checks, f"Missing subsystem: {subsystem}"

    def test_version_is_string(self, client) -> None:
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            body = client.get("/health").json()

        assert isinstance(body["version"], str)
        assert len(body["version"]) > 0

    def test_encryption_active_is_bool(self, client) -> None:
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            body = client.get("/health").json()

        assert isinstance(body["encryption_active"], bool)

    def test_pii_deidentifier_false_when_none(self, client) -> None:
        """deidentifier=None means pii_deidentifier_active should be False."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            body = client.get("/health").json()

        assert body["pii_deidentifier_active"] is False


class TestOllamaDegradation:
    def test_ollama_timeout_does_not_cause_503(self, client) -> None:
        """
        Ollama unavailability should result in status=degraded, NOT 503.
        The HITL server must stay operational even if Ollama is unreachable.
        """
        import httpx

        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value = mock_http

            resp = client.get("/health")

        assert resp.status_code == 200   # Still 200 — Ollama is non-fatal
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert body["checks"]["ollama"]["status"] == "degraded"

    def test_audit_db_checks_pass(self, client) -> None:
        """audit_db check should always be ok with a fresh in-memory DB."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            checks = client.get("/health").json()["checks"]

        assert checks["audit_db"]["status"] == "ok"
        assert "entry_count" in checks["audit_db"]

    def test_auth_db_checks_pass(self, client) -> None:
        """auth_db check should always be ok with a fresh DB."""
        with patch("httpx.AsyncClient") as mock_cls:
            mock_http = AsyncMock()
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__  = AsyncMock(return_value=None)
            mock_http.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_cls.return_value = mock_http

            checks = client.get("/health").json()["checks"]

        assert checks["auth_db"]["status"] == "ok"
        assert "reviewer_count" in checks["auth_db"]
