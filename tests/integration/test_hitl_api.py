"""
tests/integration/test_hitl_api.py
=====================================
Integration tests for the HITL FastAPI endpoints.

Uses FastAPI's TestClient (httpx-based) with a fully mocked pipeline
attached to app.state. No real Ollama or DailyMed calls.

Strategy: create a bare FastAPI app (no lifespan) and manually inject
mocked pipeline/audit_db into app.state, then include the review router.
This avoids triggering the real lifespan which requires Ollama.
"""
from __future__ import annotations

import hashlib
import uuid
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from graph.hitl_interrupt import HITLQueue
from graph.state import initial_state
from hitl.routes.review import router as review_router
from infra.auth_db import ReviewerRecord
from schemas.audit import HITLStage


# ─────────────────────────────────────────────────────────────────────────────
# Auth constants
# ─────────────────────────────────────────────────────────────────────────────

TEST_API_KEY  = "test-api-key-for-integration-tests"
REVIEWER_ID   = "TEST-REVIEWER-01"
AUTH_HEADERS  = {"X-API-Key": TEST_API_KEY}


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVE = (
    "A 45-year-old male received Aspirin 100mg daily. "
    "He developed gastrointestinal bleeding. "
    "This was a serious adverse event."
)
CASE_ID = "ICSR-API-TEST-001"


def _make_complete_state() -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    return {
        **initial_state(case_id=CASE_ID, raw_narrative=NARRATIVE, narrative_hash=h),
        "pipeline_complete": True,
        "pipeline_halted":   False,
        "final_narrative":   "A 45-year-old male patient received Aspirin and developed GI bleeding.",
        "partial_e2b":       {"H.1": "GI bleeding narrative."},
        "current_stage":     "narrative",
        "next_stage":        "complete",
    }


def _make_halted_state(review_id: str) -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    return {
        **initial_state(case_id=CASE_ID, raw_narrative=NARRATIVE, narrative_hash=h),
        "pipeline_complete": False,
        "pipeline_halted":   True,
        "hitl_stage":        HITLStage.CAUSALITY,
        "review_id":         review_id,
        "current_stage":     "causality",
        "next_stage":        "hitl_review",
        "error_log":         ["Confidence 0.60 < threshold 0.85"],
    }


def _make_mock_pipeline(run_returns: dict, resume_returns: dict | None = None):
    hitl_queue = HITLQueue()
    pipeline   = MagicMock()
    pipeline.hitl_queue              = hitl_queue
    pipeline.run_case.return_value   = run_returns
    pipeline.resume_case.return_value = resume_returns or run_returns
    pipeline.get_state.return_value  = run_returns
    return pipeline, hitl_queue


def _mock_audit_db() -> MagicMock:
    db = MagicMock()
    db.insert_hitl_review.return_value = None
    db.insert_entry.return_value       = None
    return db


def _mock_learning_db() -> MagicMock:
    db = MagicMock()
    db.insert_signal.return_value = None
    return db


def _mock_auth_db() -> MagicMock:
    """Mock AuthDB that authenticates TEST_API_KEY immediately (no bcrypt)."""
    reviewer = ReviewerRecord(
        reviewer_id  = REVIEWER_ID,
        role         = "REVIEWER",
        active       = True,
        created_at   = "2026-01-01T00:00:00+00:00",
        last_used_at = None,
    )
    db = MagicMock()
    db.authenticate.return_value = reviewer
    return db


def _make_test_app(pipeline, audit_db, learning_db=None, auth_db=None) -> FastAPI:
    """
    Build a bare FastAPI app (no lifespan) with the review router.
    Injects mocked pipeline/audit_db/learning_db/auth_db into app.state.
    """
    app = FastAPI(title="ICSR HITL Test App")
    app.include_router(review_router)

    @app.get("/health")
    def health():
        return {"status": "ok", "service": "ICSR HITL Server", "version": "0.5.0"}

    app.state.pipeline    = pipeline
    app.state.audit_db    = audit_db
    app.state.learning_db = learning_db or _mock_learning_db()
    app.state.auth_db     = auth_db or _mock_auth_db()
    return app


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def app_complete():
    pipeline, hitl_queue = _make_mock_pipeline(run_returns=_make_complete_state())
    app = _make_test_app(pipeline, _mock_audit_db(), _mock_learning_db())
    return app, hitl_queue



@pytest.fixture
def app_halted():
    review_id = str(uuid.uuid4())
    halted    = _make_halted_state(review_id)
    pipeline, hitl_queue = _make_mock_pipeline(
        run_returns    = halted,
        resume_returns = _make_complete_state(),
    )
    hitl_queue.enqueue(
        review_id  = review_id,
        thread_id  = halted["trace_id"],
        state_snap = halted,
    )
    pipeline.hitl_queue = hitl_queue
    app = _make_test_app(pipeline, _mock_audit_db())
    return app, review_id, hitl_queue


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self, app_complete):
        app, _ = app_complete
        resp = TestClient(app).get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestQueueEndpoint:
    def test_empty_queue(self, app_complete):
        """Empty queue returns 0 pending entries."""
        app, _ = app_complete
        resp = TestClient(app).get("/api/v1/review/queue", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pending"] == 0
        assert data["entries"] == []

    def test_populated_queue(self, app_halted):
        """After a halt, queue has 1 entry."""
        app, review_id, _ = app_halted
        resp = TestClient(app).get("/api/v1/review/queue", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pending"] == 1
        assert data["entries"][0]["review_id"] == review_id

    def test_no_auth_key_returns_401(self, app_complete):
        """Request without X-API-Key header → 401."""
        app, _ = app_complete
        # Override auth_db so authenticate() returns None (simulates bad/missing key)
        app.state.auth_db.authenticate.return_value = None
        resp = TestClient(app).get("/api/v1/review/queue")  # No headers
        assert resp.status_code in (401, 422)  # 422 = missing header field


class TestGetCaseEndpoint:
    def test_404_unknown_review_id(self, app_complete):
        app, _ = app_complete
        resp = TestClient(app).get(f"/api/v1/review/{uuid.uuid4()}", headers=AUTH_HEADERS)
        assert resp.status_code == 404

    def test_200_known_review_id(self, app_halted):
        app, review_id, _ = app_halted
        resp = TestClient(app).get(f"/api/v1/review/{review_id}", headers=AUTH_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["review_id"] == review_id
        assert data["case_id"]   == CASE_ID


class TestSubmitCorrectionEndpoint:
    def test_submit_approved_correction_resumes(self, app_halted):
        """POST correction with approved=True → pipeline resumes → 200."""
        app, review_id, _ = app_halted
        resp = TestClient(app).post(
            f"/api/v1/review/{review_id}/submit",
            headers=AUTH_HEADERS,
            json={
                "reviewer_id":           "QPPV-01",
                "correction":            {"causality_matrix": {}},
                "correction_rationale":  "Causality re-assessed as Possibly Related.",
                "approved":              True,
                "learning_signal":       True,
            }
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["review_id"] == review_id
        assert data["status"] in ("complete", "resumed", "still_halted")

    def test_submit_rejected_correction(self, app_halted):
        """POST correction with approved=False → status=rejected."""
        app, review_id, _ = app_halted
        resp = TestClient(app).post(
            f"/api/v1/review/{review_id}/submit",
            headers=AUTH_HEADERS,
            json={
                "reviewer_id":           "MED-REVIEWER-02",
                "correction":            {},
                "correction_rationale":  "Case narrative is incomplete and requires rework.",
                "approved":              False,
                "learning_signal":       False,
            }
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_submit_unknown_review_id_404(self, app_complete):
        app, _ = app_complete
        resp = TestClient(app).post(
            f"/api/v1/review/{uuid.uuid4()}/submit",
            headers=AUTH_HEADERS,
            json={
                "reviewer_id":          "QPPV-01",
                "correction":           {},
                "correction_rationale": "Test correction.",
                "approved":             True,
            }
        )
        assert resp.status_code == 404

    def test_submit_validation_error_422(self, app_halted):
        """POST with missing required fields → 422."""
        app, review_id, _ = app_halted
        resp = TestClient(app).post(
            f"/api/v1/review/{review_id}/submit",
            headers=AUTH_HEADERS,
            json={"correction": {}}  # Missing reviewer_id, rationale, approved
        )
        assert resp.status_code == 422

    def test_submit_invalid_api_key_401(self, app_halted):
        """POST with bad API key → 401."""
        app, review_id, _ = app_halted
        app.state.auth_db.authenticate.return_value = None  # Simulate bad key
        resp = TestClient(app).post(
            f"/api/v1/review/{review_id}/submit",
            headers={"X-API-Key": "bad-key"},
            json={
                "reviewer_id":          "QPPV-01",
                "correction":           {},
                "correction_rationale": "Test.",
                "approved":             True,
            }
        )
        assert resp.status_code == 401


class TestCasesRunEndpoint:
    def test_run_complete_case(self, app_complete):
        """POST /api/v1/cases/run → complete → status=complete."""
        app, _ = app_complete
        resp = TestClient(app).post(
            "/api/v1/cases/run",
            headers=AUTH_HEADERS,
            json={
                "case_id":       CASE_ID,
                "raw_narrative": NARRATIVE,
                "country":       "US",
            }
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "complete"
        assert data["pipeline_complete"] is True
        assert data["final_narrative"] is not None

    def test_run_halted_case_returns_review_id(self, app_halted):
        """POST → pipeline halts → status=hitl_pending + review_id present."""
        app, expected_review_id, _ = app_halted
        halted = _make_halted_state(expected_review_id)
        app.state.pipeline.run_case.return_value = halted

        resp = TestClient(app).post(
            "/api/v1/cases/run",
            headers=AUTH_HEADERS,
            json={"case_id": "ICSR-HALTED-001", "raw_narrative": NARRATIVE}
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["pipeline_halted"] is True
        assert data["status"] == "hitl_pending"
        assert data["review_id"] is not None

    def test_run_short_narrative_422(self, app_complete):
        """Narrative < 50 chars → 422 validation error."""
        app, _ = app_complete
        resp = TestClient(app).post(
            "/api/v1/cases/run",
            json={"case_id": "TEST-001", "raw_narrative": "Too short"}
        )
        assert resp.status_code == 422
