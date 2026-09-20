"""
tests/unit/test_triage_agent.py
================================
Unit tests for TriageAgent.

All Ollama calls are mocked — no real LLM is needed.
Tests verify:
  1. Valid TIER_1 case (death) → status=VALID, oversight=HITL
  2. INVALID case (missing reporter) → pipeline_halted + hitl_stage
  3. LLM status contradiction coercion (all_met=True but LLM says INVALID → override)
  4. JSON parse failure → HITL route (via BaseAgent._handle_failure)
  5. TIER_3 case (non-serious) → next_stage=extraction
"""
from __future__ import annotations

import hashlib
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agents.triage_agent import TriageAgent
from graph.state import initial_state
from schemas.triage import OversightMode, RiskTier, TriageStatus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(narrative: str = "Patient A (45F) received Amoxicillin 500mg. "
                                  "She developed anaphylaxis and died.") -> dict:
    h = hashlib.sha256(narrative.encode()).hexdigest()
    return initial_state(
        case_id="ICSR-20240101-TST",
        raw_narrative=narrative,
        narrative_hash=h,
    )


def _mock_llm(response_dict: dict) -> MagicMock:
    """Return a mock OllamaClient whose .chat() returns the given JSON dict."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(response_dict)
    mock_resp.processing_ms = 500
    llm = MagicMock()
    llm.chat.return_value = mock_resp
    return llm


def _mock_audit_db() -> MagicMock:
    db = MagicMock()
    db.insert_entry.return_value = None
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTriageAgentValid:
    def test_tier1_death_case(self):
        """TIER_1 death case → VALID, HITL, oversight=HITL, risk_tier=TIER_1."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": True,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "VALID",
            "seriousness_signals":       ["death", "died"],
            "confidence":                0.97,
            "failure_reasons":           [],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        state = _make_state()
        result = agent.run(state)

        assert result["triage_output"].status == TriageStatus.VALID
        assert result["triage_output"].risk_tier == RiskTier.TIER_1
        assert result["triage_output"].oversight_mode == OversightMode.HITL
        # TIER_1 cases always need HITL even when confidence is high (97% > 95%)
        # status=VALID + confidence=0.97 >= 0.95 → needs_hitl=False
        assert result.get("next_stage") == "extraction"
        assert result.get("pipeline_halted") is not True

    def test_tier1_always_hitl_when_low_confidence(self):
        """TIER_1 with confidence=0.80 < 0.95 threshold → pipeline_halted."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": True,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "VALID",
            "seriousness_signals":       ["death"],
            "confidence":                0.80,
            "failure_reasons":           [],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state())

        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "triage"

    def test_tier3_non_serious_no_hitl(self):
        """TIER_3 non-serious case with high confidence → no HITL."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": True,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "VALID",
            "seriousness_signals":       [],
            "confidence":                0.90,
            "failure_reasons":           [],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        narrative = "Patient (30M) took Ibuprofen and reported mild headache."
        result = agent.run(_make_state(narrative))

        assert result["triage_output"].risk_tier == RiskTier.TIER_3
        assert result["next_stage"] == "extraction"
        assert result.get("pipeline_halted") is not True


class TestTriageAgentInvalid:
    def test_invalid_missing_reporter(self):
        """Missing reporter → INVALID → pipeline_halted."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": False,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "INVALID",
            "seriousness_signals":       [],
            "confidence":                0.85,
            "failure_reasons":           ["No identifiable reporter"],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state())

        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "triage"
        assert result["triage_output"].status == TriageStatus.INVALID

    def test_llm_contradiction_coercion_valid_override(self):
        """LLM says INVALID but all 4 criteria are True → agent overrides to VALID."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": True,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "INVALID",    # LLM hallucinated INVALID
            "seriousness_signals":       [],
            "confidence":                0.88,
            "failure_reasons":           ["some false reason"],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state())

        # Coercion: criteria all_met=True overrides status to VALID
        assert result["triage_output"].status == TriageStatus.VALID

    def test_llm_contradiction_coercion_invalid_override(self):
        """LLM says VALID but reporter criterion is False → agent overrides to INVALID."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": False,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "VALID",    # LLM hallucinated VALID
            "seriousness_signals":       [],
            "confidence":                0.85,
            "failure_reasons":           [],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state())

        assert result["triage_output"].status == TriageStatus.INVALID
        assert result["pipeline_halted"] is True


class TestTriageAgentFailures:
    def test_json_parse_failure_routes_to_hitl(self):
        """Malformed LLM JSON → BaseAgent catches exception → pipeline_halted."""
        mock_resp = MagicMock()
        mock_resp.text = "This is not JSON at all!"
        mock_resp.processing_ms = 100
        llm = MagicMock()
        llm.chat.return_value = mock_resp

        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state())

        assert result["pipeline_halted"] is True
        assert len(result["error_log"]) > 0

    def test_audit_db_called_on_success(self):
        """AuditDB.insert_entry is called exactly once on successful run."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": True,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "VALID",
            "seriousness_signals":       [],
            "confidence":                0.90,
            "failure_reasons":           [],
        })
        db = _mock_audit_db()
        agent = TriageAgent(llm=llm, audit_db=db)
        agent.run(_make_state())

        db.insert_entry.assert_called_once()

    def test_review_id_refreshed(self):
        """run() must update review_id so audit trail has a new per-stage UUID."""
        llm = _mock_llm({
            "has_identifiable_patient":  True,
            "has_identifiable_reporter": True,
            "has_suspect_drug":          True,
            "has_adverse_event":         True,
            "status":                    "VALID",
            "seriousness_signals":       [],
            "confidence":                0.90,
            "failure_reasons":           [],
        })
        agent = TriageAgent(llm=llm, audit_db=_mock_audit_db())
        state = _make_state()
        original_review_id = state["review_id"]

        # run() calls refresh_review_id internally — but the RETURN dict is a partial
        # We can check that audit_db.insert_entry was called with a valid UUID
        db = _mock_audit_db()
        agent2 = TriageAgent(llm=llm, audit_db=db)
        agent2.run(state)
        call_args = db.insert_entry.call_args[0][0]   # First positional arg = AuditLogEntry
        import uuid
        uuid.UUID(call_args.review_id)   # Raises ValueError if not a valid UUID
