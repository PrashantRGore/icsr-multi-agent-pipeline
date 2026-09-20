"""
tests/integration/conftest.py
================================
Shared fixtures for pipeline integration tests.

All Ollama LLM calls are mocked end-to-end.  Each agent's LLM response
is pre-loaded from a fixture dict so that:
  - Tests run without Ollama installed
  - Tests are fully deterministic (temperature=0.0 is still verified)
  - Integration tests validate pipeline routing, state accumulation,
    audit DB writes, and HITL queue behavior

Hardware constraint:
  - Tests use an in-memory SQLite checkpointer (not file-based)
  - RxNorm and DailyMed HTTP calls are also mocked

Fixture pattern:
  - `mock_llm_responses` dict: maps (agent_id, case_id) → JSON response text
  - `pipeline_factory` builds a fully wired ICSRPipeline with mocked agents

For the integration tests to work without the real OAE/CTCAE indexes,
we mock FAISSIndex.search() to return a deterministic OAE hit above threshold.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from graph.hitl_interrupt import HITLQueue
from graph.nodes import NodeFactory
from graph.pipeline import ICSRPipeline
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic case data (mirrors data/synthetic_cases/)
# ─────────────────────────────────────────────────────────────────────────────

CASE_001_NARRATIVE = (
    "A 34-year-old female healthcare professional presented to the emergency department "
    "with acute anaphylaxis approximately 20 minutes after receiving her first dose of "
    "amoxicillin 500mg orally for a community-acquired respiratory tract infection. "
    "The patient reported no prior allergies to penicillins or other antibiotics. "
    "She developed generalized urticaria, throat tightening, severe dyspnoea, and "
    "hypotension (BP 70/40 mmHg). She was admitted to hospital on 14 March 2024. "
    "The drug was immediately discontinued. Following epinephrine, the symptoms resolved "
    "within 6 hours. Discharged on 15 March 2024. Rechallenge not performed."
)
CASE_001_ID = "ICSR-20240315-SYN"

CASE_002_NARRATIVE = (
    "An unidentified patient was taking an unknown drug when an adverse event occurred. "
    "No reporter information available. No suspect drug identified."
)
CASE_002_ID = "ICSR-20240101-INV"

CASE_003_NARRATIVE = (
    "A 58-year-old male physician in the UK was prescribed atorvastatin 40mg orally for "
    "hypercholesterolaemia starting January 2023. In March 2023 he developed drug-induced "
    "liver injury (DILI) with AST 450 IU/L. Atorvastatin was stopped. LFTs normalised "
    "within 6 weeks. Atorvastatin was restarted (rechallenge); LFTs rose again. Drug was "
    "stopped and LFTs normalised again."
)
CASE_003_ID = "ICSR-20230601-SYN"


# ─────────────────────────────────────────────────────────────────────────────
# LLM response fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _triage_valid_tier1():
    return json.dumps({
        "has_identifiable_patient":  True,
        "has_identifiable_reporter": True,
        "has_suspect_drug":          True,
        "has_adverse_event":         True,
        "status":                    "VALID",
        "seriousness_signals":       ["life-threatening", "hospitalization"],
        "confidence":                0.97,
        "failure_reasons":           [],
    })


def _triage_invalid():
    return json.dumps({
        "has_identifiable_patient":  False,
        "has_identifiable_reporter": False,
        "has_suspect_drug":          False,
        "has_adverse_event":         False,
        "status":                    "INVALID",
        "seriousness_signals":       [],
        "confidence":                0.95,
        "failure_reasons":           ["No identifiable patient", "No identifiable reporter", "No suspect drug"],
    })


def _triage_valid_tier2():
    return json.dumps({
        "has_identifiable_patient":  True,
        "has_identifiable_reporter": True,
        "has_suspect_drug":          True,
        "has_adverse_event":         True,
        "status":                    "VALID",
        "seriousness_signals":       ["hospitalization"],
        "confidence":                0.92,
        "failure_reasons":           [],
    })


def _extraction_case001():
    return json.dumps({
        "suspect_drugs": [{
            "drug_name":   "Amoxicillin",
            "dose":        "500mg",
            "route":       "Oral",
            "indication":  "community-acquired respiratory tract infection",
            "drug_role":   "SUSPECT",
            "start_date":  {"year": 2024, "month": 3, "day": 14},
            "stop_date":   {"year": 2024, "month": 3, "day": 14},
            "dechallenge": "YES",
            "rechallenge": "NOT_REPORTED",
        }],
        "verbatim_events": [{
            "verbatim_term":        "anaphylaxis",
            "reporter_term":        "acute anaphylaxis",
            "onset_date":           {"year": 2024, "month": 3, "day": 14},
            "outcome":              "recovered",
            "serious":              True,
            "seriousness_criteria": ["Life-threatening", "Hospitalization"],
            "hospitalization_details": {
                "hospitalization_flag": "Event caused hospitalization",
                "date_of_admission":    {"year": 2024, "month": 3, "day": 14},
                "date_of_discharge":    {"year": 2024, "month": 3, "day": 15},
            },
        }],
        "patient_age":           "34",
        "patient_sex":           "Female",
        "patient_ethnicity":     None,
        "patient_weight_kg":     None,
        "patient_height_cm":     None,
        "reporter_type":         "HCP",
        "country_of_occurrence": "US",
        "extraction_confidence": 0.96,
    })


def _extraction_case003():
    return json.dumps({
        "suspect_drugs": [{
            "drug_name":   "Atorvastatin",
            "dose":        "40mg",
            "route":       "Oral",
            "indication":  "hypercholesterolaemia",
            "drug_role":   "SUSPECT",
            "start_date":  {"year": 2023, "month": 1, "day": None},
            "stop_date":   {"year": 2023, "month": 3, "day": None},
            "dechallenge": "YES",
            "rechallenge": "YES",
        }],
        "verbatim_events": [{
            "verbatim_term":        "drug-induced liver injury",
            "reporter_term":        "DILI with AST 450 IU/L",
            "onset_date":           {"year": 2023, "month": 3, "day": None},
            "outcome":              "recovered",
            "serious":              True,
            "seriousness_criteria": ["Hospitalization"],
            "hospitalization_details": {
                "hospitalization_flag": "Event caused hospitalization",
                "date_of_admission":    {"year": 2023, "month": 3, "day": None},
                "date_of_discharge":    None,
            },
        }],
        "patient_age":           "58",
        "patient_sex":           "Male",
        "patient_ethnicity":     None,
        "patient_weight_kg":     None,
        "patient_height_cm":     None,
        "reporter_type":         "HCP",
        "country_of_occurrence": "UK",
        "extraction_confidence": 0.94,
    })


def _qc_approved():
    return json.dumps({
        "approved":       True,
        "critique_items": [],
        "qc_confidence":  0.95,
    })


def _coding_hit(event_node_id: str = None):
    """Return coding LLM hint — actual coding comes from mock FAISS."""
    return json.dumps({"suggested_term": "anaphylaxis"})


def _causality_related(drug_node_id: str, event_node_id: str):
    return json.dumps({"assessments": [{
        "drug_node_id":           drug_node_id,
        "event_node_id":          event_node_id,
        "drug_name":              "Amoxicillin",
        "verbatim_event":         "anaphylaxis",
        "causality_term":         "Related",
        "rationale":              "Clear temporal association; dechallenge positive.",
        "confidence":             0.95,
        "alternative_etiologies": [],
        "hard_negative_checked":  True,
    }]})


def _causality_related_case003(drug_node_id: str, event_node_id: str):
    return json.dumps({"assessments": [{
        "drug_node_id":           drug_node_id,
        "event_node_id":          event_node_id,
        "drug_name":              "Atorvastatin",
        "verbatim_event":         "drug-induced liver injury",
        "causality_term":         "Related",
        "rationale":              "Positive dechallenge and rechallenge.",
        "confidence":             0.97,
        "alternative_etiologies": [],
        "hard_negative_checked":  True,
    }]})


def _narrative_text(drug: str = "Amoxicillin", event: str = "anaphylaxis"):
    return (
        f"A patient received {drug}. They developed {event}. "
        "The drug was discontinued and the event resolved. "
        "Causality was assessed as Related (WHO-UMC). "
        "The event was listed in the RSI."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared mock builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_llm(responses: list[str]) -> OllamaClient:
    """
    Build a mock OllamaClient that returns successive responses from the list.
    The last response is repeated for any additional calls.
    """
    call_count = {"n": 0}

    def _make_resp(text: str) -> MagicMock:
        r = MagicMock()
        r.text = text
        r.processing_ms = 500
        r.content_hash = hashlib.sha256(text.encode()).hexdigest()
        return r

    def _chat(*args, **kwargs) -> MagicMock:
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return _make_resp(responses[idx])

    def _generate(*args, **kwargs) -> MagicMock:
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return _make_resp(responses[idx])

    llm = MagicMock(spec=OllamaClient)
    llm.chat.side_effect     = _chat
    llm.generate.side_effect = _generate
    return llm


def _make_mock_rxnorm(drug_name: str = "Amoxicillin", rxcui: str = "723") -> MagicMock:
    result = MagicMock()
    result.rxcui       = rxcui
    result.label       = drug_name.lower()
    result.drug_class  = "Penicillin"
    rxnorm = MagicMock()
    rxnorm.batch_lookup.return_value = {drug_name.lower(): result}
    return rxnorm


def _make_mock_faiss(score: float = 0.92, term: str = "anaphylaxis") -> MagicMock:
    idx = MagicMock()
    idx.search.return_value = [{"label": term, "score": score, "text": term}]
    return idx


def _make_mock_dailymed_session(
    is_listed: bool = True,
    event_term: str = "anaphylaxis",
) -> MagicMock:
    """Mock requests.Session for DailyMed lookups."""
    session = MagicMock()
    adverse_text = (
        f"Adverse Reactions: {event_term}, urticaria, rash."
        if is_listed else
        "Adverse Reactions: headache, nausea."
    )

    def get_side_effect(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "spls.json" in url:
            resp.json.return_value = {"data": [{"setid": "test-set-id"}]}
        else:
            resp.json.return_value = {"data": [
                {"loinc_code": "34084-4", "text": adverse_text}
            ]}
        return resp

    session.get.side_effect = get_side_effect
    return session


# ─────────────────────────────────────────────────────────────────────────────
# pytest fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_audit_db(tmp_path):
    return AuditDB(db_path=tmp_path / "audit.db")


@pytest.fixture
def tmp_neg_path():
    """Write a minimal clinical_negatives.json to a temp file."""
    import tempfile, json
    data = [
        {
            "edge_type": "DOES_NOT_TREAT",
            "subject":   "amoxicillin",
            "object":    ["bradycardia"],
            "source":    "pharmacology",
        }
    ]
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    json.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


@pytest.fixture
def in_memory_checkpointer():
    """Return an in-memory MemorySaver checkpointer (no file I/O)."""
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()


def build_pipeline_case001(
    audit_db,
    neg_path,
    checkpointer,
) -> tuple[ICSRPipeline, list[str]]:
    """
    Build a fully mocked ICSRPipeline for Case 001 (valid TIER_1 anaphylaxis).

    Returns (pipeline, llm_response_sequence) — responses are in call order:
      1. triage
      2. extraction
      3. qc
      4. coding hint (LLM not actually used for coding if FAISS hits)
      5. causality  (uses extracted node IDs — built dynamically)
      6. narrative
    """
    # LLM responses in order (causality will use extracted node IDs from state)
    responses = [
        _triage_valid_tier1(),
        _extraction_case001(),
        _qc_approved(),
        # NOTE: NO coding slot — OAE FAISS score=0.92 > threshold=0.80, LLM not called
        # NOTE: NO causality slot — _patch_causality_agent bypasses the LLM
        _narrative_text("Amoxicillin", "anaphylaxis"),
    ]

    llm      = _make_mock_llm(responses)
    rxnorm   = _make_mock_rxnorm("Amoxicillin", "723")
    oae_idx  = _make_mock_faiss(score=0.92, term="anaphylaxis")
    session  = _make_mock_dailymed_session(is_listed=True, event_term="anaphylaxis")

    factory = NodeFactory(
        llm=llm, audit_db=audit_db, rxnorm=rxnorm,
        oae_idx=oae_idx, ctcae_idx=None, neg_path=neg_path,
        session=session,
    )

    # Causality agent needs real node_ids — patch .chat to generate them dynamically
    _patch_causality_agent(factory._causality, llm)

    pipeline = ICSRPipeline(
        factory=factory,
        checkpointer=checkpointer,
        hitl_queue=HITLQueue(),
    )
    return pipeline, responses


def _patch_causality_agent(causality_agent, llm: MagicMock) -> None:
    """
    Override causality agent's _run_inner to bypass the LLM entirely.

    The causality agent requires real node_ids (auto-generated UUIDs during
    extraction) — we can't pre-load them into the mock's response list.

    IMPORTANT: We must NOT touch llm.chat.side_effect here.  The mock LLM uses
    a side_effect function with an internal call counter.  Clearing side_effect
    (the previous approach) permanently breaks the counter for all subsequent
    agents (coding → narrative), causing those agents to get wrong responses.

    Instead, we implement _run_inner directly: read node_ids from state,
    build a CausalityMatrix, and return the expected state patch.
    The mock llm.chat is never called, so the counter is preserved.
    """
    from schemas.causality import (
        CausalityAssessment, CausalityMatrix, CausalityTerm,
    )
    from schemas.extraction import DrugRole
    from schemas.triage import THRESHOLD_BY_TIER, RiskTier

    def _direct_run_inner(state: dict) -> dict:
        """Directly construct causality output from extracted entities."""
        entities = state.get("extracted_entities")
        case_id  = state.get("case_id", "UNKNOWN")
        risk_tier = state.get("risk_tier") or RiskTier.TIER_1

        if not entities:
            # No entities — return halted state so HITL queue gets it
            return {
                "pipeline_halted": True,
                "hitl_stage":      causality_agent.AGENT_ID,
                "next_stage":      "hitl_review",
                "current_stage":   "causality",
                "error_log":       [f"{causality_agent.AGENT_ID}: no entities for case={case_id}"],
            }

        suspect = [d for d in entities.suspect_drugs if d.drug_role == DrugRole.SUSPECT]
        events  = entities.verbatim_events

        assessments = []
        for drug in suspect:
            for event in events:
                assessments.append(CausalityAssessment(
                    drug_node_id           = drug.node_id,
                    event_node_id          = event.node_id,
                    drug_name              = drug.drug_name,
                    verbatim_event         = event.verbatim_term,
                    causality_term         = CausalityTerm.RELATED,
                    rationale              = "Temporal link + positive dechallenge.",
                    confidence             = 0.95,
                    alternative_etiologies = [],
                    hard_negative_checked  = True,
                    agent_id               = causality_agent.AGENT_ID,
                    prompt_version         = "causality-prompt-v1.0",
                ))

        matrix = CausalityMatrix(case_id=case_id, assessments=assessments)

        partial_e2b: dict = dict(state.get("partial_e2b") or {})
        for ca in assessments:
            k = f"G.k.9.i.2.{ca.drug_node_id}.{ca.event_node_id}"
            partial_e2b[k] = ca.causality_term.value

        threshold   = THRESHOLD_BY_TIER[risk_tier]
        min_conf    = min((a.confidence for a in assessments), default=0.0)
        needs_hitl  = min_conf < threshold

        patch: dict = {
            "causality_matrix": matrix,
            "partial_e2b":      partial_e2b,
            "current_stage":    "causality",
        }
        if needs_hitl:
            patch.update({
                "pipeline_halted": True,
                "hitl_stage":      causality_agent.AGENT_ID,
                "next_stage":      "hitl_review",
            })
        else:
            patch["next_stage"] = "listedness"

        return patch

    causality_agent._run_inner = _direct_run_inner

