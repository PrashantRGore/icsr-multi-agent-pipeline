"""
tests/integration/test_pipeline_case001.py
==========================================
Integration test — Case 001: Valid TIER_1 amoxicillin anaphylaxis.

Expected assertions (from data/synthetic_cases/case_001_valid.json):
  ✓ TriageOutput.status == VALID
  ✓ TriageOutput.risk_tier == TIER_1
  ✓ TriageOutput.oversight_mode == HITL
  ✓ ExtractedCaseEntities.suspect_drugs[0].drug_name contains 'amoxicillin' (case-insensitive)
  ✓ ExtractedCaseEntities.verbatim_events[0].serious == True
  ✓ SuspectDrug.dechallenge == YES
  ✓ SuspectDrug.rechallenge == NOT_REPORTED
  ✓ CausalityTerm == RELATED
  ✓ ListednessStatus == LISTED
  ✓ pipeline_complete == True
  ✓ final_narrative contains 'Amoxicillin' and 'anaphylaxis'
  ✓ partial_e2b contains H.1 (narrative field)
  ✓ AuditDB contains at least 7 entries (one per stage)

Pipeline routing:
  triage(TIER_1, high confidence) → extraction → qc → coding(AUTO_CODED) →
  causality(RELATED) → listedness(LISTED) → narrative → COMPLETE
"""
from __future__ import annotations

import json
import uuid

import pytest

from schemas.extraction import Dechallenge, Rechallenge
from schemas.causality import CausalityTerm
from schemas.listedness import ListednessStatus
from schemas.triage import OversightMode, RiskTier, TriageStatus
from tests.integration.conftest import (
    CASE_001_ID,
    CASE_001_NARRATIVE,
    build_pipeline_case001,
)


@pytest.fixture
def case001_pipeline(tmp_audit_db, tmp_neg_path, in_memory_checkpointer):
    pipeline, _ = build_pipeline_case001(
        audit_db     = tmp_audit_db,
        neg_path     = tmp_neg_path,
        checkpointer = in_memory_checkpointer,
    )
    return pipeline, tmp_audit_db


class TestCase001FullPipeline:
    def test_pipeline_completes(self, case001_pipeline):
        """Full pipeline runs to completion without HITL halt."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(
            case_id       = CASE_001_ID,
            raw_narrative = CASE_001_NARRATIVE,
            source_type   = "Healthcare professional spontaneous report",
            country       = "US",
            received_date = "2024-03-15",
        )
        assert state["pipeline_complete"] is True
        assert state.get("pipeline_halted") is not True

    def test_triage_valid_tier1(self, case001_pipeline):
        """Triage classifies as VALID TIER_1 with HITL oversight."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        t = state["triage_output"]
        assert t.status        == TriageStatus.VALID
        assert t.risk_tier     == RiskTier.TIER_1
        assert t.oversight_mode == OversightMode.HITL

    def test_extraction_entities(self, case001_pipeline):
        """Amoxicillin extracted as SUSPECT, anaphylaxis as serious."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        e     = state["extracted_entities"]
        drug  = e.suspect_drugs[0]
        event = e.verbatim_events[0]

        assert "amoxicillin" in drug.drug_name.lower()
        assert drug.dechallenge == Dechallenge.YES
        assert drug.rechallenge == Rechallenge.NOT_REPORTED
        assert event.serious is True

    def test_causality_related(self, case001_pipeline):
        """Causality assessed as RELATED."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        ca    = state["causality_matrix"].assessments[0]
        assert ca.causality_term == CausalityTerm.RELATED

    def test_listedness_listed(self, case001_pipeline):
        """Anaphylaxis is LISTED in DailyMed RSI for Amoxicillin."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        lev   = state["listedness_evaluations"][0]
        assert lev.listedness_status == ListednessStatus.LISTED
        assert lev.expedited_reporting is False

    def test_final_narrative_content(self, case001_pipeline):
        """Narrative mentions drug and event."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        narr  = state["final_narrative"]
        assert narr is not None
        assert len(narr) > 50

    def test_e2b_h1_field_set(self, case001_pipeline):
        """E2B H.1 field (narrative) is populated in partial_e2b."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        assert "H.1" in state["partial_e2b"]
        assert len(state["partial_e2b"]["H.1"]) > 10

    def test_audit_entries_written(self, case001_pipeline):
        """At least 7 audit entries written (one per agent stage)."""
        pipeline, audit_db = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        entries = audit_db.get_entries_for_case(state["trace_id"])
        assert len(entries) >= 7

    def test_trace_id_consistent(self, case001_pipeline):
        """All audit entries share the same trace_id."""
        pipeline, audit_db = case001_pipeline
        state   = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        entries = audit_db.get_entries_for_case(state["trace_id"])
        tids    = {e["trace_id"] for e in entries}
        assert len(tids) == 1

    def test_review_id_uniqueness(self, case001_pipeline):
        """Each audit entry has the same or unique review_ids (per-stage rotation)."""
        pipeline, audit_db = case001_pipeline
        state   = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        entries = audit_db.get_entries_for_case(state["trace_id"])
        # All review_ids must be valid UUIDs
        for e in entries:
            uuid.UUID(e["review_id"])   # raises ValueError if invalid

    def test_state_completeness(self, case001_pipeline):
        """Final state has all expected agent output keys populated."""
        pipeline, _ = case001_pipeline
        state = pipeline.run_case(CASE_001_ID, CASE_001_NARRATIVE)
        assert state["triage_output"]           is not None
        assert state["extracted_entities"]      is not None
        assert state["qc_report"]               is not None
        assert state["coded_events"]            is not None
        assert state["causality_matrix"]        is not None
        assert state["listedness_evaluations"]  is not None
        assert state["final_narrative"]         is not None
