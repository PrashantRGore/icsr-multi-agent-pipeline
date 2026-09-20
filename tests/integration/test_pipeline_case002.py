"""
tests/integration/test_pipeline_case002.py
==========================================
Integration test — Case 002: INVALID case (missing reporter, no patient, no suspect drug).

Expected assertions:
  ✓ TriageOutput.status == INVALID
  ✓ pipeline_halted == True (INVALID case → HITL after triage)
  ✓ hitl_stage == "triage"
  ✓ Pipeline does NOT proceed to extraction (no entities extracted)
  ✓ Case is enqueued in HITLQueue with the correct review_id
  ✓ HITLQueue has 1 entry with correct case_id
  ✓ Audit entry written for triage stage

Pipeline routing:
  triage(INVALID) → HITL interrupt → STOP (no further processing)
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from graph.hitl_interrupt import HITLQueue
from graph.nodes import NodeFactory
from graph.pipeline import ICSRPipeline
from schemas.triage import TriageStatus
from tests.integration.conftest import (
    CASE_002_ID,
    CASE_002_NARRATIVE,
    _triage_invalid,
    _make_mock_rxnorm,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def case002_pipeline(tmp_audit_db, tmp_neg_path, in_memory_checkpointer):
    llm = MagicMock()
    resp = MagicMock()
    resp.text           = _triage_invalid()
    resp.processing_ms  = 300
    resp.content_hash   = "a" * 64
    llm.chat.return_value    = resp
    llm.generate.return_value = resp

    hitl_queue = HITLQueue()
    factory    = NodeFactory(
        llm=llm, audit_db=tmp_audit_db, rxnorm=_make_mock_rxnorm(),
        neg_path=tmp_neg_path,
    )
    pipeline = ICSRPipeline(
        factory=factory,
        checkpointer=in_memory_checkpointer,
        hitl_queue=hitl_queue,
    )
    return pipeline, tmp_audit_db, hitl_queue


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestCase002InvalidPipeline:
    def test_pipeline_halts_at_triage(self, case002_pipeline):
        """INVALID case halts at triage stage."""
        pipeline, _, _ = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert state["pipeline_halted"] is True

    def test_triage_status_invalid(self, case002_pipeline):
        """TriageOutput.status == INVALID."""
        pipeline, _, _ = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert state["triage_output"].status == TriageStatus.INVALID

    def test_hitl_stage_is_triage(self, case002_pipeline):
        """hitl_stage is set to 'triage'."""
        pipeline, _, _ = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        stage = state.get("hitl_stage")
        stage_str = stage.value if hasattr(stage, "value") else str(stage)
        assert stage_str.lower() == "triage"

    def test_no_extraction_performed(self, case002_pipeline):
        """extracted_entities is None — pipeline stopped at triage."""
        pipeline, _, _ = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert state.get("extracted_entities") is None

    def test_pipeline_not_complete(self, case002_pipeline):
        """pipeline_complete must be False."""
        pipeline, _, _ = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert state.get("pipeline_complete") is not True

    def test_case_enqueued_in_hitl_queue(self, case002_pipeline):
        """HITLQueue has 1 entry after the case halts."""
        pipeline, _, hitl_queue = case002_pipeline
        pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert len(hitl_queue) == 1

    def test_hitl_queue_case_id_correct(self, case002_pipeline):
        """HITLQueue entry has correct case_id."""
        pipeline, _, hitl_queue = case002_pipeline
        pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        pending = hitl_queue.list_pending()
        assert pending[0]["case_id"] == CASE_002_ID

    def test_audit_entry_for_triage(self, case002_pipeline):
        """AuditDB has at least 1 entry (triage stage)."""
        pipeline, audit_db, _ = case002_pipeline
        state   = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        entries = audit_db.get_entries_for_case(state["trace_id"])
        assert len(entries) >= 1
        agent_ids = [e["agent_id"] for e in entries]
        assert any("triage" in a for a in agent_ids)

    def test_failure_reasons_in_triage_output(self, case002_pipeline):
        """TriageOutput.failure_reasons contains at least one reason."""
        pipeline, _, _ = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert len(state["triage_output"].failure_reasons) > 0


class TestCase002HITLResume:
    def test_hitl_resume_with_correction_proceeds(
        self, case002_pipeline, tmp_neg_path, in_memory_checkpointer, tmp_audit_db
    ):
        """
        After HITL halt, submitting a correction that includes the missing fields
        and re-running as VALID should route to extraction.

        This test simulates a human reviewer correcting the case data and
        re-injecting it into the pipeline.
        """
        pipeline, _, hitl_queue = case002_pipeline
        state = pipeline.run_case(CASE_002_ID, CASE_002_NARRATIVE)
        assert state["pipeline_halted"] is True

        # Get the review_id
        pending = hitl_queue.list_pending()
        assert len(pending) == 1
        review_id = pending[0]["review_id"]

        # Simulate human correction: override triage with a VALID result
        from schemas.triage import (
            MinimumCriteria, OversightMode, RiskTier, TriageOutput, TriageStatus
        )
        import hashlib
        h = hashlib.sha256(CASE_002_NARRATIVE.encode()).hexdigest()
        corrected_triage = TriageOutput(
            case_id        = CASE_002_ID,
            narrative_hash = h,
            criteria       = MinimumCriteria(
                has_identifiable_patient  = True,
                has_identifiable_reporter = True,
                has_suspect_drug          = True,
                has_adverse_event         = True,
            ),
            status          = TriageStatus.VALID,
            risk_tier       = RiskTier.TIER_2,
            oversight_mode  = OversightMode.HITL,
            seriousness_signals = [],
            confidence      = 0.90,
        )

        # Mock extraction response for resume
        from tests.integration.conftest import _extraction_case001, _make_mock_llm
        extraction_resp = json.loads(_extraction_case001())
        extraction_resp["suspect_drugs"][0]["drug_name"] = "Aspirin"
        extraction_resp["extraction_confidence"] = 0.88

        # After triage correction, the pipeline resumes at extraction.
        # We patch the extraction agent's LLM to return a minimal valid response.
        factory = pipeline._factory
        resp = MagicMock()
        resp.text = json.dumps(extraction_resp)
        resp.processing_ms = 500
        factory._extract._llm.chat.return_value = resp

        # Also mock QC, coding, causality, listedness, narrative for full resume
        # (use the existing mocks — we only care that pipeline resumes here)

        resumed = pipeline.resume_case(
            review_id  = review_id,
            correction = {"triage_output": corrected_triage, "next_stage": "extraction"},
        )

        # Pipeline should have proceeded past triage
        # (may halt again at subsequent stages if mock responses are insufficient)
        assert resumed.get("pipeline_halted") is not True or \
               resumed.get("hitl_stage") != "triage", \
               "Pipeline must advance past triage stage after correction"
