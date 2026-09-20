"""
tests/unit/test_narrative_agent.py
====================================
Unit tests for NarrativeAgent.

LLM .generate() is mocked — tests verify:
  1. Final narrative returned in state as 'final_narrative'
  2. SHA-256 narrative_hash updated
  3. E2B key H.1 written to partial_e2b
  4. pipeline_complete=True set
  5. Self-check warns when drug name absent from narrative (but does NOT block)
  6. Audit DB called on success
"""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import pytest

from agents.narrative_agent import NarrativeAgent
from graph.state import initial_state
from schemas.causality import CausalityAssessment, CausalityMatrix, CausalityTerm
from schemas.coding import CodedEvent, CodingStatus
from schemas.extraction import (
    Dechallenge, DrugRole, ExtractedCaseEntities,
    PartialDate, Rechallenge, SuspectDrug, VerbatimEvent,
    SeriousnessCriterion,
    HospitalizationCausalityFlag, HospitalizationDetails,
)
from schemas.listedness import ListednessEvaluation, ListednessStatus
from schemas.triage import (
    MinimumCriteria, OversightMode, RiskTier, TriageOutput, TriageStatus,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVE = "Patient (35M) received Amoxicillin 500mg PO. Developed anaphylaxis on Jan 10."
_MOCK_NARRATIVE_TEXT = (
    "A 35-year-old male patient received Amoxicillin 500mg orally for dental infection "
    "starting January 5, 2024. On January 10, 2024, the patient developed anaphylaxis "
    "requiring hospitalization. The drug was discontinued and the adverse event resolved. "
    "Causality was assessed as Related (WHO-UMC). The event was listed in the RSI. "
    "No expedited reporting obligation applies."
)


def _make_full_state() -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    state = initial_state(
        case_id="ICSR-20240110-TST",
        raw_narrative=NARRATIVE,
        narrative_hash=h,
    )

    # Triage
    criteria = MinimumCriteria(
        has_identifiable_patient=True,
        has_identifiable_reporter=True,
        has_suspect_drug=True,
        has_adverse_event=True,
    )
    state["triage_output"] = TriageOutput(
        case_id=state["case_id"],
        narrative_hash=h,
        criteria=criteria,
        status=TriageStatus.VALID,
        risk_tier=RiskTier.TIER_2,
        oversight_mode=OversightMode.HITL,
        seriousness_signals=["hospitalization"],
        confidence=0.90,
    )
    state["risk_tier"] = "TIER_2"

    # Entities
    drug = SuspectDrug(
        drug_name   = "Amoxicillin",
        drug_role   = DrugRole.SUSPECT,
        dose        = "500mg",
        route       = "PO",
        start_date  = PartialDate(year=2024, month=1, day=5),
        dechallenge = Dechallenge.YES,
        rechallenge = Rechallenge.NOT_REPORTED,
    )
    event = VerbatimEvent(
        verbatim_term         = "anaphylaxis",
        onset_date            = PartialDate(year=2024, month=1, day=10),
        serious               = True,
        seriousness_criteria  = [SeriousnessCriterion.HOSPITALIZATION],
        hospitalization_details = HospitalizationDetails(
            hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION,
        ),
    )
    entities = ExtractedCaseEntities(
        case_id="ICSR-20240110-TST",
        narrative_hash=h,
        suspect_drugs=[drug],
        verbatim_events=[event],
        extraction_confidence=0.95,
        patient_age="35",
        patient_sex="Male",
        reporter_type="HCP",
        country_of_occurrence="US",
    )
    state["extracted_entities"] = entities

    # Causality
    ca = CausalityAssessment(
        drug_node_id   = drug.node_id,
        event_node_id  = event.node_id,
        drug_name      = "Amoxicillin",
        verbatim_event = "anaphylaxis",
        causality_term = CausalityTerm.RELATED,
        rationale      = "Clear temporal association; dechallenge positive.",
        confidence     = 0.95,
    )
    state["causality_matrix"] = CausalityMatrix(
        case_id="ICSR-20240110-TST",
        assessments=[ca],
    )

    # Coded events
    state["coded_events"] = [
        CodedEvent(
            event_node_id = event.node_id,
            verbatim_term = "anaphylaxis",
            oae_term      = "anaphylaxis",
            coding_status = CodingStatus.AUTO_CODED,
        )
    ]

    # Listedness
    state["listedness_evaluations"] = [
        ListednessEvaluation(
            event_node_id     = event.node_id,
            drug_node_id      = drug.node_id,
            verbatim_term     = "anaphylaxis",
            coded_term        = "anaphylaxis",
            listedness_status = ListednessStatus.LISTED,
            confidence        = 0.90,
        )
    ]

    state["partial_e2b"] = {"H.1.preliminary": "draft"}
    return state


def _mock_llm(narrative: str = _MOCK_NARRATIVE_TEXT) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = narrative
    mock_resp.processing_ms = 2000
    llm = MagicMock()
    llm.generate.return_value = mock_resp
    return llm


def _mock_audit_db() -> MagicMock:
    db = MagicMock()
    db.insert_entry.return_value = None
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrativeAgentSuccess:
    def test_final_narrative_in_state(self):
        """NarrativeAgent places narrative text in state['final_narrative']."""
        agent = NarrativeAgent(llm=_mock_llm(), audit_db=_mock_audit_db())
        result = agent.run(_make_full_state())
        assert "final_narrative" in result
        assert "Amoxicillin" in result["final_narrative"]

    def test_pipeline_complete_set(self):
        """pipeline_complete=True is set after NarrativeAgent succeeds."""
        agent = NarrativeAgent(llm=_mock_llm(), audit_db=_mock_audit_db())
        result = agent.run(_make_full_state())
        assert result["pipeline_complete"] is True

    def test_next_stage_complete(self):
        """next_stage is set to 'complete'."""
        agent = NarrativeAgent(llm=_mock_llm(), audit_db=_mock_audit_db())
        result = agent.run(_make_full_state())
        assert result["next_stage"] == "complete"

    def test_e2b_h1_written(self):
        """partial_e2b key 'H.1' is populated with the narrative text."""
        agent = NarrativeAgent(llm=_mock_llm(), audit_db=_mock_audit_db())
        result = agent.run(_make_full_state())
        assert "H.1" in result["partial_e2b"]
        assert result["partial_e2b"]["H.1"] == _MOCK_NARRATIVE_TEXT

    def test_narrative_hash_updated(self):
        """narrative_hash in result is SHA-256 of the new narrative text."""
        agent = NarrativeAgent(llm=_mock_llm(), audit_db=_mock_audit_db())
        result = agent.run(_make_full_state())
        expected_hash = hashlib.sha256(_MOCK_NARRATIVE_TEXT.encode()).hexdigest()
        assert result["narrative_hash"] == expected_hash

    def test_audit_db_called(self):
        """AuditDB.insert_entry is called once on success."""
        db = _mock_audit_db()
        agent = NarrativeAgent(llm=_mock_llm(), audit_db=db)
        agent.run(_make_full_state())
        db.insert_entry.assert_called_once()


class TestNarrativeAgentSelfCheck:
    def test_missing_drug_in_narrative_does_not_block(self):
        """Self-check warns about missing entity but does NOT block pipeline."""
        # Return narrative that does NOT mention Amoxicillin
        agent = NarrativeAgent(
            llm=_mock_llm("Patient developed anaphylaxis after taking medication."),
            audit_db=_mock_audit_db(),
        )
        result = agent.run(_make_full_state())
        # Should still succeed (self-check is advisory)
        assert result["pipeline_complete"] is True

    def test_llm_generate_called_not_chat(self):
        """NarrativeAgent uses .generate() not .chat() (free-form text, not JSON)."""
        llm = _mock_llm()
        agent = NarrativeAgent(llm=llm, audit_db=_mock_audit_db())
        agent.run(_make_full_state())
        llm.generate.assert_called_once()
        llm.chat.assert_not_called()
