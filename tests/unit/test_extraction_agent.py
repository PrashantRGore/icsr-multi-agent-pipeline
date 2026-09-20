"""
tests/unit/test_extraction_agent.py
=====================================
Unit tests for ExtractionAgent.

Tests verify:
  1. Full extraction with RxNorm enrichment → ExtractedCaseEntities populated
  2. Partial date parsing (year-only, year+month, full)
  3. Hospitalization details auto-injected when criterion present
  4. HITL routing when extraction_confidence < tier threshold
  5. JSON parse failure → pipeline_halted
  6. at_least_one_suspect_drug enforced (SUSPECT role)
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock

import pytest

from agents.extraction_agent import ExtractionAgent
from graph.state import initial_state
from schemas.extraction import DrugRole, ExtractedCaseEntities, SeriousnessCriterion


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVE = (
    "A 35-year-old male physician from the US was prescribed Amoxicillin 500mg PO "
    "for dental infection starting January 5, 2024. On January 10, 2024, he developed "
    "anaphylaxis requiring hospitalization. The drug was stopped (dechallenge positive). "
    "No rechallenge was attempted."
)


def _make_state(risk_tier: str = "TIER_1") -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    state = initial_state(
        case_id="ICSR-20240110-TST",
        raw_narrative=NARRATIVE,
        narrative_hash=h,
    )
    state["risk_tier"] = risk_tier
    return state


def _mock_llm(response_dict: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(response_dict)
    mock_resp.processing_ms = 800
    llm = MagicMock()
    llm.chat.return_value = mock_resp
    return llm


def _mock_rxnorm(rxcui: str = "723") -> MagicMock:
    """Returns a mock RxNormClient with a single result."""
    result = MagicMock()
    result.rxcui = rxcui
    result.label = "amoxicillin"
    result.drug_class = "Aminopenicillin"
    rxnorm = MagicMock()
    rxnorm.batch_lookup.return_value = {"amoxicillin": result}
    return rxnorm


def _mock_audit_db() -> MagicMock:
    db = MagicMock()
    db.insert_entry.return_value = None
    return db


_FULL_RESPONSE = {
    "suspect_drugs": [
        {
            "drug_name":   "Amoxicillin",
            "dose":        "500mg",
            "route":       "PO",
            "indication":  "dental infection",
            "drug_role":   "SUSPECT",
            "start_date":  {"year": 2024, "month": 1, "day": 5},
            "stop_date":   None,
            "dechallenge": "YES",
            "rechallenge": "NOT_REPORTED",
        }
    ],
    "verbatim_events": [
        {
            "verbatim_term":        "anaphylaxis",
            "reporter_term":        "anaphylaxis",
            "onset_date":           {"year": 2024, "month": 1, "day": 10},
            "outcome":              "recovered",
            "serious":              True,
            "seriousness_criteria": ["Hospitalization"],
            "hospitalization_details": {
                "hospitalization_flag": "Event caused hospitalization",
                "date_of_admission":    {"year": 2024, "month": 1, "day": 10},
                "date_of_discharge":    {"year": 2024, "month": 1, "day": 12},
            },
        }
    ],
    "patient_age":           "35",
    "patient_sex":           "Male",
    "patient_ethnicity":     None,
    "patient_weight_kg":     None,
    "patient_height_cm":     None,
    "reporter_type":         "HCP",
    "country_of_occurrence": "US",
    "extraction_confidence": 0.95,
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionAgentSuccess:
    def test_full_extraction_returns_entities(self):
        """Full extraction produces a valid ExtractedCaseEntities in state."""
        agent = ExtractionAgent(
            llm=_mock_llm(_FULL_RESPONSE),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())

        assert "extracted_entities" in result
        entities = result["extracted_entities"]
        assert isinstance(entities, ExtractedCaseEntities)
        assert len(entities.suspect_drugs) == 1
        assert entities.suspect_drugs[0].drug_name == "Amoxicillin"
        assert len(entities.verbatim_events) == 1
        assert entities.verbatim_events[0].serious is True

    def test_rxnorm_enrichment_applied(self):
        """RxNorm RxCUI is filled in on the extracted drug."""
        agent = ExtractionAgent(
            llm=_mock_llm(_FULL_RESPONSE),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(rxcui="723"),
        )
        result = agent.run(_make_state())
        entities = result["extracted_entities"]
        assert entities.suspect_drugs[0].rxcui == "723"
        assert entities.suspect_drugs[0].rxnorm_label == "amoxicillin"

    def test_hospitalization_details_populated(self):
        """HOSPITALIZATION criterion triggers hospitalization_details."""
        agent = ExtractionAgent(
            llm=_mock_llm(_FULL_RESPONSE),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())
        event = result["extracted_entities"].verbatim_events[0]
        assert SeriousnessCriterion.HOSPITALIZATION in event.seriousness_criteria
        assert event.hospitalization_details is not None

    def test_partial_e2b_written(self):
        """partial_e2b contains demographic field codes."""
        agent = ExtractionAgent(
            llm=_mock_llm(_FULL_RESPONSE),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())
        assert "D.2.2" in result["partial_e2b"]   # patient_age
        assert result["partial_e2b"]["C.1.9"] == "US"

    def test_high_confidence_tier1_no_hitl(self):
        """Confidence=0.95 ≥ TIER_1 threshold 0.95 → next_stage=qc."""
        agent = ExtractionAgent(
            llm=_mock_llm(_FULL_RESPONSE),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state(risk_tier="TIER_1"))
        assert result.get("next_stage") == "qc"
        assert result.get("pipeline_halted") is not True

    def test_graph_payload_returned(self):
        """MAGMA graph JSON is written to state."""
        agent = ExtractionAgent(
            llm=_mock_llm(_FULL_RESPONSE),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())
        assert result["graph_payload"] is not None
        import json
        data = json.loads(result["graph_payload"])
        assert "nodes" in data or "directed" in data


class TestExtractionAgentHITL:
    def test_low_confidence_routes_to_hitl(self):
        """Confidence=0.70 < TIER_1 threshold 0.95 → pipeline_halted."""
        low_conf = {**_FULL_RESPONSE, "extraction_confidence": 0.70}
        agent = ExtractionAgent(
            llm=_mock_llm(low_conf),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state(risk_tier="TIER_1"))
        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "extraction"

    def test_json_parse_failure_routes_to_hitl(self):
        """Malformed JSON from LLM → pipeline_halted via BaseAgent."""
        mock_resp = MagicMock()
        mock_resp.text = "NOT VALID JSON"
        mock_resp.processing_ms = 100
        llm = MagicMock()
        llm.chat.return_value = mock_resp

        agent = ExtractionAgent(
            llm=llm,
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())
        assert result["pipeline_halted"] is True
        assert len(result["error_log"]) > 0


class TestExtractionAgentEdgeCases:
    def test_year_only_date_accepted(self):
        """Year-only partial date is accepted for start_date."""
        resp = {
            **_FULL_RESPONSE,
            "suspect_drugs": [{
                **_FULL_RESPONSE["suspect_drugs"][0],
                "start_date": {"year": 2023, "month": None, "day": None},
            }],
        }
        agent = ExtractionAgent(
            llm=_mock_llm(resp),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())
        drug = result["extracted_entities"].suspect_drugs[0]
        assert drug.start_date is not None
        assert drug.start_date.year == 2023
        assert drug.start_date.month is None

    def test_missing_hospitalization_details_auto_injected(self):
        """When HOSPITALIZATION criterion present but details missing, default details injected."""
        resp = {
            **_FULL_RESPONSE,
            "verbatim_events": [{
                **_FULL_RESPONSE["verbatim_events"][0],
                "hospitalization_details": None,   # Missing
            }],
        }
        agent = ExtractionAgent(
            llm=_mock_llm(resp),
            audit_db=_mock_audit_db(),
            rxnorm=_mock_rxnorm(),
        )
        result = agent.run(_make_state())
        event = result["extracted_entities"].verbatim_events[0]
        assert event.hospitalization_details is not None
