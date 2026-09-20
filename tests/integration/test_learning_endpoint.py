"""
tests/integration/test_learning_endpoint.py
=============================================
Integration tests for GET /api/v1/learning/* endpoints.

Uses the same pattern as test_health.py: create_app() without triggering
the full lifespan, inject app.state directly with real (tmp) DBs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    TestClient with a real LearningDB (1 signal) injected into app.state.
    Bypasses the full lifespan to avoid requiring Ollama / FAISS.
    """
    from infra.audit_db import AuditDB
    from infra.auth_db import AuthDB, ReviewerRecord
    from infra.learning_db import LearningDB
    from hitl.auth import require_reviewer
    from hitl.main import create_app

    tmp = tmp_path_factory.mktemp("learning_test")

    # Build real DBs
    audit_db    = AuditDB(db_path=tmp / "audit.db")
    auth_db     = AuthDB(db_path=tmp / "auth.db")
    learning_db = LearningDB(db_path=tmp / "learning.db")

    # Seed one learning signal
    learning_db.insert_signal(
        trace_id="T1", review_id="R1", stage="extraction",
        agent_id="extraction-agent-v1", prompt_version="v1",
        error_classification="OMISSION",
        field_path="extracted_entities.suspect_drugs",
        original_value="A", corrected_value="B",
        correction_rationale="test",
        reviewer_id="QPPV-01",
        reviewed_at="2026-09-10T10:00:00+00:00",
        patient_sex="FEMALE", patient_age_group="ADULT", patient_ethnicity=None,
    )

    mock_reviewer = ReviewerRecord(
        reviewer_id="QPPV-01", role="QPPV",
        active=True,
        created_at="2026-01-01T00:00:00+00:00",
        last_used_at=None,
    )

    app = create_app()

    # Override auth dependency BEFORE lifespan — dependency_overrides survive lifespan
    app.dependency_overrides[require_reviewer] = lambda: mock_reviewer

    with TestClient(app, raise_server_exceptions=True) as c:
        # Lifespan has now completed and set up its own app.state.*
        # Re-inject our fixture DB so tests see the seeded signal
        c.app.state.learning_db = learning_db
        yield c

    app.dependency_overrides.clear()


# ── /api/v1/learning/report ───────────────────────────────────────────────────

class TestLearningReport:
    def test_status_200(self, client):
        r = client.get("/api/v1/learning/report",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 200

    def test_response_has_total_signals(self, client):
        data = client.get("/api/v1/learning/report",
                          headers={"X-API-Key": "any-key"}).json()
        assert "total_signals" in data
        assert data["total_signals"] == 1

    def test_response_has_agent_accuracy(self, client):
        data = client.get("/api/v1/learning/report",
                          headers={"X-API-Key": "any-key"}).json()
        assert "agent_accuracy" in data
        assert isinstance(data["agent_accuracy"], list)

    def test_response_has_top_patterns(self, client):
        data = client.get("/api/v1/learning/report",
                          headers={"X-API-Key": "any-key"}).json()
        assert "top_patterns" in data

    def test_response_has_bias_report(self, client):
        data = client.get("/api/v1/learning/report",
                          headers={"X-API-Key": "any-key"}).json()
        assert "bias_report" in data


# ── /api/v1/learning/agents ───────────────────────────────────────────────────

class TestAgentsEndpoint:
    def test_agents_status_200(self, client):
        r = client.get("/api/v1/learning/agents",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 200

    def test_agents_list_in_response(self, client):
        data = client.get("/api/v1/learning/agents",
                          headers={"X-API-Key": "any-key"}).json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        agent_ids = {a["agent_id"] for a in data["agents"]}
        assert "extraction-agent-v1" in agent_ids


# ── /api/v1/learning/bias ─────────────────────────────────────────────────────

class TestBiasEndpoint:
    def test_bias_status_200(self, client):
        r = client.get("/api/v1/learning/bias",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 200

    def test_bias_table_in_response(self, client):
        data = client.get("/api/v1/learning/bias",
                          headers={"X-API-Key": "any-key"}).json()
        assert "bias_table" in data
