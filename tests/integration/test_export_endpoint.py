"""
tests/integration/test_export_endpoint.py
==========================================
Integration tests for:
  GET /api/v1/cases/{case_id}/export/e2b
  GET /api/v1/cases/{case_id}/export/json

Injects app.state directly without triggering the full lifespan
(same pattern as test_health.py) to avoid requiring Ollama / FAISS.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

_FAKE_STATE = json.dumps({
    "triage_output":      {"risk_tier": "TIER_1"},
    "extracted_entities": {
        "patient_sex":   "FEMALE",
        "patient_age":   "55",
        "suspect_drugs": [{"name": "penicillin", "rxnorm_cui": "7980"}],
    },
    "causality_matrix": {"penicillin": "CERTAIN"},
    "coded_events":     [{"preferred_term": "Urticaria", "ctcae_code": "10046735"}],
    "final_narrative":  "Patient developed urticaria after penicillin.",
})


# ── Client fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """
    TestClient with real AuditDB (1 completed case) injected into app.state.
    Bypasses the full lifespan to avoid requiring Ollama / FAISS.
    """
    from infra.audit_db import AuditDB
    from infra.auth_db import AuthDB, ReviewerRecord
    from infra.learning_db import LearningDB
    from hitl.auth import require_reviewer
    from hitl.main import create_app

    tmp      = tmp_path_factory.mktemp("export_test")
    audit_db = AuditDB(db_path=tmp / "audit.db")

    # Insert a completed audit log entry using the correct schema
    with audit_db._connect() as conn:
        conn.execute(
            """
            INSERT INTO audit_log
               (entry_id, trace_id, run_id, review_id,
                agent_id, prompt_version, model_name, model_temp,
                status, content_hash, timestamp, extra_metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "ENTRY-001",           # entry_id
                "ICSR-EXPORT-001",     # trace_id
                "RUN-001",             # run_id
                "REV-001",             # review_id
                "narrative-agent-v1", # agent_id
                "v1",                  # prompt_version
                "llama3.1:8b",         # model_name
                0.0,                   # model_temp
                "SUCCESS",             # status
                "a" * 64,              # content_hash (64 hex chars)
                "2026-09-10T10:00:00Z", # timestamp
                _FAKE_STATE,           # extra_metadata
            ),
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
        # Lifespan has now completed and created its own app.state.audit_db.
        # Re-inject our fixture DB (which has the seeded ICSR-EXPORT-001 row)
        c.app.state.audit_db = audit_db
        yield c

    app.dependency_overrides.clear()


# ── E2B export tests ──────────────────────────────────────────────────────────

class TestE2BExport:
    def test_e2b_status_200(self, client):
        r = client.get("/api/v1/cases/ICSR-EXPORT-001/export/e2b",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 200

    def test_e2b_content_type_xml(self, client):
        r = client.get("/api/v1/cases/ICSR-EXPORT-001/export/e2b",
                       headers={"X-API-Key": "any-key"})
        assert "application/xml" in r.headers["content-type"]

    def test_e2b_content_disposition_filename(self, client):
        r  = client.get("/api/v1/cases/ICSR-EXPORT-001/export/e2b",
                        headers={"X-API-Key": "any-key"})
        cd = r.headers.get("content-disposition", "")
        assert "ICSR-EXPORT-001.xml" in cd

    def test_e2b_xml_is_well_formed(self, client):
        r = client.get("/api/v1/cases/ICSR-EXPORT-001/export/e2b",
                       headers={"X-API-Key": "any-key"})
        ET.fromstring(r.content)   # bytes — handles XML declaration

    def test_e2b_contains_case_id(self, client):
        r = client.get("/api/v1/cases/ICSR-EXPORT-001/export/e2b",
                       headers={"X-API-Key": "any-key"})
        assert "ICSR-EXPORT-001" in r.text

    def test_e2b_contains_adr001_comment(self, client):
        r = client.get("/api/v1/cases/ICSR-EXPORT-001/export/e2b",
                       headers={"X-API-Key": "any-key"})
        assert "MedDRA LLT/PT" in r.text or "ADR-001" in r.text

    def test_e2b_404_for_unknown_case(self, client):
        r = client.get("/api/v1/cases/DOES-NOT-EXIST/export/e2b",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 404


# ── JSON export tests ─────────────────────────────────────────────────────────

class TestJSONExport:
    def test_json_status_200(self, client):
        r = client.get("/api/v1/cases/ICSR-EXPORT-001/export/json",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 200

    def test_json_case_id_present(self, client):
        data = client.get("/api/v1/cases/ICSR-EXPORT-001/export/json",
                          headers={"X-API-Key": "any-key"}).json()
        assert data["case_id"] == "ICSR-EXPORT-001"

    def test_json_has_meddra_note(self, client):
        data = client.get("/api/v1/cases/ICSR-EXPORT-001/export/json",
                          headers={"X-API-Key": "any-key"}).json()
        assert "meddra_note" in data
        assert "ADR-001" in data["meddra_note"]

    def test_json_has_patient_block(self, client):
        data = client.get("/api/v1/cases/ICSR-EXPORT-001/export/json",
                          headers={"X-API-Key": "any-key"}).json()
        assert "patient" in data

    def test_json_404_for_unknown_case(self, client):
        r = client.get("/api/v1/cases/GHOST-CASE/export/json",
                       headers={"X-API-Key": "any-key"})
        assert r.status_code == 404
