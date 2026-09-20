"""
tests/unit/test_qc_agent.py
============================
Unit tests for QCAgent.

Tests verify:
  1. Clean extraction → approved=True, empty critique_items
  2. LLM returns BLOCKER → approved=False → HITL route
  3. Hard-negative ontology blocks DOES_NOT_TREAT pair → BLOCKER added
  4. INCREASES_RISK_OF edges do NOT trigger violations
  5. Missing extracted_entities → pipeline_halted with error
  6. Audit DB called on success
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agents.qc_agent import QCAgent
from graph.state import initial_state
from schemas.extraction import (
    Dechallenge, DrugRole, ExtractedCaseEntities,
    HospitalizationCausalityFlag, HospitalizationDetails,
    PartialDate, Rechallenge, SeriousnessCriterion, SuspectDrug, VerbatimEvent,
)
from schemas.qc import CritiqueSeverity


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVE = "Patient (35M) received Amoxicillin. Developed anaphylaxis."


def _make_entities(drug_name: str = "Amoxicillin", event_term: str = "anaphylaxis"):
    return ExtractedCaseEntities(
        case_id="ICSR-20240101-TST",
        narrative_hash="a" * 64,
        suspect_drugs=[
            SuspectDrug(
                drug_name   = drug_name,
                drug_role   = DrugRole.SUSPECT,
                start_date  = PartialDate(year=2024, month=1, day=5),
                dechallenge = Dechallenge.YES,
                rechallenge = Rechallenge.NOT_REPORTED,
            )
        ],
        verbatim_events=[
            VerbatimEvent(
                verbatim_term = event_term,
                onset_date    = PartialDate(year=2024, month=1, day=10),
                serious       = True,
                seriousness_criteria=[SeriousnessCriterion.HOSPITALIZATION],
                hospitalization_details=HospitalizationDetails(
                    hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION,
                ),
            )
        ],
        extraction_confidence=0.95,
    )


def _make_state(entities=None) -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    state = initial_state(
        case_id="ICSR-20240101-TST",
        raw_narrative=NARRATIVE,
        narrative_hash=h,
    )
    if entities is not None:
        state["extracted_entities"] = entities
    return state


def _mock_llm(response_dict: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(response_dict)
    mock_resp.processing_ms = 600
    llm = MagicMock()
    llm.chat.return_value = mock_resp
    return llm


def _mock_audit_db() -> MagicMock:
    db = MagicMock()
    db.insert_entry.return_value = None
    return db


def _write_negatives(rules: list) -> Path:
    """Write a clinical_negatives.json to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(rules, tmp)
    tmp.close()
    return Path(tmp.name)


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestQCAgentApproved:
    def test_clean_extraction_approved(self):
        """LLM finds no errors → approved=True, next_stage=coding."""
        llm = _mock_llm({
            "approved":       True,
            "critique_items": [],
            "qc_confidence":  0.95,
        })
        agent = QCAgent(llm=llm, audit_db=_mock_audit_db(),
                        neg_path=_write_negatives([]))
        result = agent.run(_make_state(_make_entities()))

        assert result["qc_report"].approved is True
        assert result["next_stage"] == "coding"
        assert result.get("pipeline_halted") is not True

    def test_warnings_only_still_approved(self):
        """WARNING items do NOT block approval."""
        llm = _mock_llm({
            "approved": True,
            "critique_items": [
                {
                    "error_classification": "OMISSION",
                    "severity": "WARNING",
                    "affected_node_id": None,
                    "field_path": None,
                    "message": "Patient weight not extracted.",
                    "suggested_correction": None,
                }
            ],
            "qc_confidence": 0.88,
        })
        agent = QCAgent(llm=llm, audit_db=_mock_audit_db(),
                        neg_path=_write_negatives([]))
        result = agent.run(_make_state(_make_entities()))
        assert result["qc_report"].approved is True


class TestQCAgentBlocked:
    def test_blocker_causes_hitl(self):
        """BLOCKER critique item → approved=False → pipeline_halted."""
        llm = _mock_llm({
            "approved": False,
            "critique_items": [
                {
                    "error_classification": "HALLUCINATION",
                    "severity": "BLOCKER",
                    "affected_node_id": "DRG-00000001",
                    "field_path": "G.k.2",
                    "message": "Drug not mentioned in narrative.",
                    "suggested_correction": "Remove hallucinated drug.",
                }
            ],
            "qc_confidence": 0.60,
        })
        agent = QCAgent(llm=llm, audit_db=_mock_audit_db(),
                        neg_path=_write_negatives([]))
        result = agent.run(_make_state(_make_entities()))

        assert result["qc_report"].approved is False
        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "qc"

    def test_hard_negative_does_not_treat_blocks(self):
        """DOES_NOT_TREAT edge in negatives → HardNegativeViolation added → BLOCKER."""
        neg_rules = [
            {
                "edge_type": "DOES_NOT_TREAT",
                "subject":   "amoxicillin",
                "object":    ["tachycardia"],
                "source":    "pharmacology",
            }
        ]
        llm = _mock_llm({
            "approved":       True,
            "critique_items": [],
            "qc_confidence":  0.95,
        })
        entities = _make_entities(drug_name="Amoxicillin", event_term="tachycardia")
        agent = QCAgent(llm=llm, audit_db=_mock_audit_db(),
                        neg_path=_write_negatives(neg_rules))
        result = agent.run(_make_state(entities))

        assert len(result["qc_report"].hard_negative_violations) == 1
        assert result["qc_report"].approved is False
        assert result["pipeline_halted"] is True

    def test_increases_risk_of_does_not_block(self):
        """INCREASES_RISK_OF edges must NOT create violations."""
        neg_rules = [
            {
                "edge_type": "INCREASES_RISK_OF",
                "subject":   "amoxicillin",
                "object":    ["anaphylaxis"],
                "source":    "pharmacology",
            }
        ]
        llm = _mock_llm({
            "approved":       True,
            "critique_items": [],
            "qc_confidence":  0.95,
        })
        entities = _make_entities(drug_name="Amoxicillin", event_term="anaphylaxis")
        agent = QCAgent(llm=llm, audit_db=_mock_audit_db(),
                        neg_path=_write_negatives(neg_rules))
        result = agent.run(_make_state(entities))

        assert len(result["qc_report"].hard_negative_violations) == 0
        assert result["qc_report"].approved is True


class TestQCAgentErrors:
    def test_missing_entities_raises_and_halts(self):
        """Absent extracted_entities → pipeline_halted with error_log entry."""
        agent = QCAgent(
            llm=_mock_llm({"approved": True, "critique_items": [], "qc_confidence": 0.9}),
            audit_db=_mock_audit_db(),
            neg_path=_write_negatives([]),
        )
        result = agent.run(_make_state(entities=None))   # No entities!
        assert result["pipeline_halted"] is True
        assert any("QCAgent" in e for e in result["error_log"])

    def test_audit_db_called_on_success(self):
        """AuditDB.insert_entry is called after a successful QC run."""
        db = _mock_audit_db()
        agent = QCAgent(
            llm=_mock_llm({"approved": True, "critique_items": [], "qc_confidence": 0.9}),
            audit_db=db,
            neg_path=_write_negatives([]),
        )
        agent.run(_make_state(_make_entities()))
        db.insert_entry.assert_called_once()
