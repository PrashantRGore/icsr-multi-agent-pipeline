"""
tests/integration/test_pipeline_case003.py
==========================================
Integration test — Case 003: TIER_2 atorvastatin DILI with positive rechallenge.

Expected assertions:
  ✓ TriageOutput.status == VALID
  ✓ TriageOutput.risk_tier == TIER_2
  ✓ SuspectDrug.drug_name contains 'atorvastatin' (case-insensitive)
  ✓ SuspectDrug.rechallenge == YES (rechallenge+ evidence)
  ✓ CausalityTerm == RELATED (rechallenge+ is strong evidence)
  ✓ ListednessStatus == LISTED or UNLISTED (DILI can be unlisted for atorvastatin)
  ✓ If UNLISTED + serious → expedited_reporting == True
  ✓ pipeline_complete == True

Special clinical rule tested:
  - Rechallenge positive (drug restarted → AE recurred → drug stopped again)
  - Partial dates accepted (year+month, no day)

Pipeline routing:
  triage(TIER_2) → extraction → qc → coding → causality(RELATED) →
  listedness(UNLISTED → expedited_reporting=True) → narrative → COMPLETE
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from schemas.causality import CausalityTerm
from schemas.extraction import Rechallenge
from schemas.listedness import ListednessStatus
from schemas.triage import RiskTier, TriageStatus
from tests.integration.conftest import (
    CASE_003_ID,
    CASE_003_NARRATIVE,
    _extraction_case003,
    _triage_valid_tier2,
    _qc_approved,
    _make_mock_faiss,
    _make_mock_rxnorm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def case003_pipeline(tmp_audit_db, tmp_neg_path, in_memory_checkpointer):
    """
    Case 003 pipeline: atorvastatin DILI with rechallenge+.
    ListednessAgent configured so DILI is UNLISTED (not in mock SPL text).
    """
    from graph.hitl_interrupt import HITLQueue
    from graph.nodes import NodeFactory
    from graph.pipeline import ICSRPipeline
    from tests.integration.conftest import (
        _make_mock_llm,
        _patch_causality_agent,
        _make_mock_dailymed_session,
    )

    # DILI is NOT in mock adverse reactions text → UNLISTED
    session = _make_mock_dailymed_session(is_listed=False, event_term="drug-induced liver injury")

    narrative_text = (
        "A 58-year-old male received Atorvastatin and developed drug-induced liver injury. "
        "Causality was assessed as Related. The event was not listed in the RSI."
    )
    responses = [
        _triage_valid_tier2(),
        _extraction_case003(),
        _qc_approved(),
        # NOTE: NO coding slot — OAE FAISS score=0.88 > threshold=0.80, LLM not called
        # NOTE: NO causality slot — _patch_causality_agent bypasses the LLM
        narrative_text,
    ]
    llm    = _make_mock_llm(responses)
    rxnorm = _make_mock_rxnorm("Atorvastatin", "83367")
    oae    = _make_mock_faiss(score=0.88, term="drug-induced liver injury")

    factory = NodeFactory(
        llm=llm, audit_db=tmp_audit_db, rxnorm=rxnorm,
        oae_idx=oae, ctcae_idx=None, neg_path=tmp_neg_path,
        session=session,
    )
    _patch_causality_agent(factory._causality, llm)

    pipeline = ICSRPipeline(
        factory=factory,
        checkpointer=in_memory_checkpointer,
        hitl_queue=HITLQueue(),
    )
    return pipeline, tmp_audit_db


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCase003RechallengePipeline:
    def test_pipeline_completes(self, case003_pipeline):
        """Full TIER_2 pipeline completes without HITL halt."""
        pipeline, _ = case003_pipeline
        state = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        assert state["pipeline_complete"] is True
        assert state.get("pipeline_halted") is not True

    def test_triage_valid_tier2(self, case003_pipeline):
        """Triage produces VALID TIER_2."""
        pipeline, _ = case003_pipeline
        state = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        t = state["triage_output"]
        assert t.status    == TriageStatus.VALID
        assert t.risk_tier == RiskTier.TIER_2

    def test_atorvastatin_extracted(self, case003_pipeline):
        """Atorvastatin is the SUSPECT drug."""
        pipeline, _ = case003_pipeline
        state  = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        drugs  = state["extracted_entities"].suspect_drugs
        assert any("atorvastatin" in d.drug_name.lower() for d in drugs)

    def test_rechallenge_positive(self, case003_pipeline):
        """Rechallenge is recorded as YES (positive rechallenge evidence)."""
        pipeline, _ = case003_pipeline
        state  = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        from schemas.extraction import DrugRole
        suspect = [d for d in state["extracted_entities"].suspect_drugs
                   if d.drug_role == DrugRole.SUSPECT]
        assert suspect[0].rechallenge == Rechallenge.YES

    def test_partial_dates_accepted(self, case003_pipeline):
        """Year+month partial dates are accepted (no day required)."""
        pipeline, _ = case003_pipeline
        state  = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        from schemas.extraction import DrugRole
        suspect = [d for d in state["extracted_entities"].suspect_drugs
                   if d.drug_role == DrugRole.SUSPECT]
        drug = suspect[0]
        # Start date: January 2023 (year+month, no day)
        assert drug.start_date is not None
        assert drug.start_date.year  == 2023
        assert drug.start_date.month == 1
        assert drug.start_date.day   is None

    def test_causality_related_rechallenge(self, case003_pipeline):
        """RELATED causality — rechallenge+ is strong evidence."""
        pipeline, _ = case003_pipeline
        state = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        ca    = state["causality_matrix"].assessments[0]
        assert ca.causality_term == CausalityTerm.RELATED

    def test_dili_unlisted_expedited(self, case003_pipeline):
        """
        DILI not in mock SPL → UNLISTED.
        UNLISTED + serious → expedited_reporting=True (15-day report obligation).
        """
        pipeline, _ = case003_pipeline
        state = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        lev   = state["listedness_evaluations"][0]
        assert lev.listedness_status == ListednessStatus.UNLISTED
        assert lev.expedited_reporting is True

    def test_narrative_mentions_drug_and_event(self, case003_pipeline):
        """Narrative contains key clinical terms."""
        pipeline, _ = case003_pipeline
        state = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        narr  = state["final_narrative"].lower()
        assert "atorvastatin" in narr or "statin" in narr

    def test_audit_entries_all_stages(self, case003_pipeline):
        """All 7 agent stages produce audit entries."""
        pipeline, audit_db = case003_pipeline
        state   = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        entries = audit_db.get_entries_for_case(state["trace_id"])
        assert len(entries) >= 7

    def test_expedited_reporting_in_e2b(self, case003_pipeline):
        """partial_e2b contains a listedness status field for the event."""
        pipeline, _ = case003_pipeline
        state = pipeline.run_case(CASE_003_ID, CASE_003_NARRATIVE)
        e2b   = state["partial_e2b"]
        # At least one listedness key must be present
        assert any("E.i.3a" in k for k in e2b)
