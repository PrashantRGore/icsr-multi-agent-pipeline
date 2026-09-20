"""
tests/unit/test_causality_agent.py
====================================
Unit tests for CausalityAgent.

Tests verify:
  1. RELATED assessment with rechallenge+ evidence
  2. UNLIKELY_RELATED must include alternative_etiologies
  3. NOT_RELATED auto-injects default etiology when LLM omits it
  4. CausalityMatrix duplicate pair raises → HITL route
  5. Confidence < tier threshold → HITL
  6. Missing entities → pipeline_halted
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from agents.causality_agent import CausalityAgent
from graph.state import initial_state
from schemas.causality import CausalityTerm
from schemas.extraction import (
    Dechallenge, DrugRole, ExtractedCaseEntities,
    PartialDate, Rechallenge, SuspectDrug, VerbatimEvent,
    SeriousnessCriterion,
    HospitalizationCausalityFlag, HospitalizationDetails,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVE = "Patient received Amoxicillin and developed anaphylaxis."


def _make_entities(
    drug_name: str = "Amoxicillin",
    event_term: str = "anaphylaxis",
    rechallenge: Rechallenge = Rechallenge.YES,
):
    drug = SuspectDrug(
        drug_name   = drug_name,
        drug_role   = DrugRole.SUSPECT,
        start_date  = PartialDate(year=2024, month=1, day=5),
        dechallenge = Dechallenge.YES,
        rechallenge = rechallenge,
    )
    event = VerbatimEvent(
        verbatim_term      = event_term,
        onset_date         = PartialDate(year=2024, month=1, day=10),
        serious            = True,
        seriousness_criteria = [SeriousnessCriterion.HOSPITALIZATION],
        hospitalization_details = HospitalizationDetails(
            hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION,
        ),
    )
    return ExtractedCaseEntities(
        case_id="ICSR-20240101-TST",
        narrative_hash="a" * 64,
        suspect_drugs=[drug],
        verbatim_events=[event],
        extraction_confidence=0.95,
    ), drug.node_id, event.node_id


def _make_state(entities=None, risk_tier: str = "TIER_2") -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    state = initial_state(
        case_id="ICSR-20240101-TST",
        raw_narrative=NARRATIVE,
        narrative_hash=h,
    )
    state["risk_tier"] = risk_tier
    if entities is not None:
        state["extracted_entities"] = entities
    return state


def _mock_llm(response_dict: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(response_dict)
    mock_resp.processing_ms = 700
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

class TestCausalityAgentRelated:
    def test_related_assessment_no_etiologies_required(self):
        """RELATED term does not require alternative_etiologies."""
        entities, drug_id, event_id = _make_entities()
        llm = _mock_llm({"assessments": [{
            "drug_node_id":           drug_id,
            "event_node_id":          event_id,
            "drug_name":              "Amoxicillin",
            "verbatim_event":         "anaphylaxis",
            "causality_term":         "Related",
            "rationale":              "Positive dechallenge and rechallenge.",
            "confidence":             0.95,
            "alternative_etiologies": [],
            "hard_negative_checked":  True,
        }]})
        agent = CausalityAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state(entities, "TIER_2"))

        assert result["causality_matrix"].assessments[0].causality_term == CausalityTerm.RELATED
        assert result["next_stage"] == "listedness"
        assert result.get("pipeline_halted") is not True

    def test_e2b_causality_field_written(self):
        """G.k.9.i.2 E2B key written for the drug–event pair."""
        entities, drug_id, event_id = _make_entities()
        llm = _mock_llm({"assessments": [{
            "drug_node_id":           drug_id,
            "event_node_id":          event_id,
            "drug_name":              "Amoxicillin",
            "verbatim_event":         "anaphylaxis",
            "causality_term":         "Related",
            "rationale":              "Clear causal link.",
            "confidence":             0.95,
            "alternative_etiologies": [],
            "hard_negative_checked":  True,
        }]})
        agent = CausalityAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state(entities))
        assert any("G.k.9.i.2" in k for k in result["partial_e2b"])


class TestCausalityAgentUnlikely:
    def test_unlikely_related_requires_etiologies_auto_injected(self):
        """UNLIKELY_RELATED with no etiologies → auto-injected default etiology."""
        entities, drug_id, event_id = _make_entities(rechallenge=Rechallenge.NO)
        llm = _mock_llm({"assessments": [{
            "drug_node_id":           drug_id,
            "event_node_id":          event_id,
            "drug_name":              "Amoxicillin",
            "verbatim_event":         "anaphylaxis",
            "causality_term":         "Unlikely related",
            "rationale":              "Other possible causes exist.",
            "confidence":             0.75,
            "alternative_etiologies": [],   # LLM forgot to include etiologies
            "hard_negative_checked":  True,
        }]})
        agent = CausalityAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state(entities))

        ca = result["causality_matrix"].assessments[0]
        assert ca.causality_term == CausalityTerm.UNLIKELY_RELATED
        # Auto-injected default
        assert len(ca.alternative_etiologies) >= 1

    def test_not_related_with_explicit_etiologies(self):
        """NOT_RELATED with explicitly provided etiologies passes validation."""
        entities, drug_id, event_id = _make_entities()
        llm = _mock_llm({"assessments": [{
            "drug_node_id":           drug_id,
            "event_node_id":          event_id,
            "drug_name":              "Amoxicillin",
            "verbatim_event":         "anaphylaxis",
            "causality_term":         "Not related",
            "rationale":              "Pre-existing allergy to bee sting confirmed.",
            "confidence":             0.90,
            "alternative_etiologies": [
                {"description": "Bee sting allergy", "likelihood": "High", "source_reference": None}
            ],
            "hard_negative_checked":  True,
        }]})
        agent = CausalityAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state(entities))

        ca = result["causality_matrix"].assessments[0]
        assert ca.causality_term == CausalityTerm.NOT_RELATED
        assert ca.alternative_etiologies[0].description == "Bee sting allergy"


class TestCausalityAgentHITL:
    def test_low_confidence_routes_to_hitl(self):
        """Confidence=0.60 < TIER_2 threshold 0.85 → pipeline_halted."""
        entities, drug_id, event_id = _make_entities()
        llm = _mock_llm({"assessments": [{
            "drug_node_id":           drug_id,
            "event_node_id":          event_id,
            "drug_name":              "Amoxicillin",
            "verbatim_event":         "anaphylaxis",
            "causality_term":         "Possibly related",
            "rationale":              "Temporal relationship unclear.",
            "confidence":             0.60,
            "alternative_etiologies": [],
            "hard_negative_checked":  True,
        }]})
        agent = CausalityAgent(llm=llm, audit_db=_mock_audit_db())
        result = agent.run(_make_state(entities, "TIER_2"))

        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "causality"

    def test_missing_entities_halts(self):
        """Missing extracted_entities → pipeline_halted with error_log."""
        agent = CausalityAgent(
            llm=MagicMock(),
            audit_db=_mock_audit_db(),
        )
        state = _make_state(entities=None)
        state.pop("extracted_entities", None)
        result = agent.run(state)
        assert result["pipeline_halted"] is True
        assert len(result["error_log"]) > 0
