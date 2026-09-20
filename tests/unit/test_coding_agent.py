"""
tests/unit/test_coding_agent.py
================================
Unit tests for CodingAgent.

Tests verify:
  1. OAE FAISS hit above threshold → AUTO_CODED
  2. OAE miss → CTCAE fallback hit → FALLBACK_CTCAE
  3. Both miss → LLM hint called → NEEDS_MANUAL → HITL
  4. Multiple events: one NEEDS_MANUAL → pipeline_halted
  5. Missing extracted_entities → error → pipeline_halted
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from agents.coding_agent import CodingAgent
from graph.state import initial_state
from schemas.coding import CodingStatus
from schemas.extraction import (
    Dechallenge, DrugRole, ExtractedCaseEntities,
    PartialDate, Rechallenge, SuspectDrug, VerbatimEvent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_entities(events: list[str] = None) -> ExtractedCaseEntities:
    events = events or ["anaphylaxis"]
    return ExtractedCaseEntities(
        case_id="ICSR-20240101-TST",
        narrative_hash="a" * 64,
        suspect_drugs=[
            SuspectDrug(drug_name="Amoxicillin", drug_role=DrugRole.SUSPECT,
                        dechallenge=Dechallenge.UNKNOWN, rechallenge=Rechallenge.NOT_REPORTED)
        ],
        verbatim_events=[
            VerbatimEvent(verbatim_term=e, serious=False) for e in events
        ],
        extraction_confidence=0.90,
    )


def _make_state(entities=None) -> dict:
    narrative = "Patient received Amoxicillin and developed anaphylaxis."
    h = hashlib.sha256(narrative.encode()).hexdigest()
    state = initial_state(
        case_id="ICSR-20240101-TST",
        raw_narrative=narrative,
        narrative_hash=h,
    )
    state["extracted_entities"] = entities or _make_entities()
    return state


def _mock_faiss(score: float, term: str = "anaphylaxis", oae_id: str = None) -> MagicMock:
    """FAISS index that returns one result with the given score."""
    idx = MagicMock()
    result = {"label": term, "text": term, "score": score}
    if oae_id:
        result["oae_id"] = oae_id
    idx.search.return_value = [result] if score > 0 else []
    return idx


def _mock_llm_hint(suggested: str = "Allergic reaction") -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = json.dumps({"suggested_term": suggested})
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

class TestCodingAgentOAE:
    def test_oae_hit_auto_coded(self):
        """OAE FAISS score ≥ 0.80 → CodingStatus.AUTO_CODED."""
        agent = CodingAgent(
            llm=_mock_llm_hint(),
            audit_db=_mock_audit_db(),
            oae_index=_mock_faiss(score=0.92, term="anaphylaxis"),
            ctcae_index=None,
        )
        result = agent.run(_make_state())
        assert result["coded_events"][0].coding_status == CodingStatus.AUTO_CODED
        assert result["next_stage"] == "causality"

    def test_oae_low_score_falls_back_to_ctcae(self):
        """OAE score=0.50 < 0.80 → tries CTCAE fallback."""
        agent = CodingAgent(
            llm=_mock_llm_hint(),
            audit_db=_mock_audit_db(),
            oae_index=_mock_faiss(score=0.50, term="anaphylaxis"),
            ctcae_index=_mock_faiss(score=0.75, term="Allergic reaction"),
        )
        result = agent.run(_make_state())
        assert result["coded_events"][0].coding_status == CodingStatus.FALLBACK_CTCAE


class TestCodingAgentFallbacks:
    def test_both_miss_needs_manual(self):
        """OAE+CTCAE both below threshold → NEEDS_MANUAL → pipeline_halted."""
        oae = MagicMock()
        oae.search.return_value = []
        ctcae = MagicMock()
        ctcae.search.return_value = []

        agent = CodingAgent(
            llm=_mock_llm_hint("Allergic Reaction"),
            audit_db=_mock_audit_db(),
            oae_index=oae,
            ctcae_index=ctcae,
        )
        result = agent.run(_make_state())
        assert result["coded_events"][0].coding_status == CodingStatus.NEEDS_MANUAL
        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "coding"

    def test_needs_manual_provides_llm_hint(self):
        """NEEDS_MANUAL case includes LLM suggestion in meddra_pt field."""
        oae = MagicMock()
        oae.search.return_value = []
        ctcae = MagicMock()
        ctcae.search.return_value = []

        agent = CodingAgent(
            llm=_mock_llm_hint("Anaphylactic Shock"),
            audit_db=_mock_audit_db(),
            oae_index=oae,
            ctcae_index=ctcae,
        )
        result = agent.run(_make_state())
        assert result["coded_events"][0].meddra_pt == "Anaphylactic Shock"

    def test_one_event_needs_manual_triggers_hitl(self):
        """One out of two events is NEEDS_MANUAL → whole case goes to HITL."""
        oae = MagicMock()
        # First event hits, second misses
        oae.search.side_effect = [
            [{"label": "anaphylaxis", "score": 0.92}],
            [],
        ]
        ctcae = MagicMock()
        ctcae.search.return_value = []

        entities = _make_entities(events=["anaphylaxis", "some rare obscure term"])
        agent = CodingAgent(
            llm=_mock_llm_hint(),
            audit_db=_mock_audit_db(),
            oae_index=oae,
            ctcae_index=ctcae,
        )
        result = agent.run(_make_state(entities))
        assert result["pipeline_halted"] is True
        coded = result["coded_events"]
        statuses = {ce.coding_status for ce in coded}
        assert CodingStatus.NEEDS_MANUAL in statuses


class TestCodingAgentEdgeCases:
    def test_partial_e2b_written(self):
        """E2B term codes written to partial_e2b for each event."""
        agent = CodingAgent(
            llm=_mock_llm_hint(),
            audit_db=_mock_audit_db(),
            oae_index=_mock_faiss(score=0.88, term="anaphylaxis"),
            ctcae_index=None,
        )
        result = agent.run(_make_state())
        # At least one E2B key should be in partial_e2b
        assert any("E.i.2" in k for k in result["partial_e2b"])

    def test_missing_entities_halts_pipeline(self):
        """Missing extracted_entities → error → pipeline_halted."""
        agent = CodingAgent(
            llm=MagicMock(),
            audit_db=_mock_audit_db(),
            oae_index=None,
            ctcae_index=None,
        )
        state = _make_state()
        state["extracted_entities"] = None

        result = agent.run(state)
        assert result["pipeline_halted"] is True
        assert len(result["error_log"]) > 0

    def test_no_indexes_needs_manual(self):
        """With both indexes=None, every event returns NEEDS_MANUAL."""
        agent = CodingAgent(
            llm=_mock_llm_hint(),
            audit_db=_mock_audit_db(),
            oae_index=None,
            ctcae_index=None,
        )
        result = agent.run(_make_state())
        assert result["coded_events"][0].coding_status == CodingStatus.NEEDS_MANUAL
