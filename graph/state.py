"""
graph/state.py
==============
GraphState — the single immutable blackboard that flows through the entire
LangGraph pipeline without being mutated in-place.

Design decisions (v4):
  - TypedDict (not Pydantic BaseModel) because LangGraph checkpointing requires
    a plain dict-like structure serialisable by its SQLiteSaver backend.
  - All Optional fields default to None; agents write their section only.
  - review_id is a per-stage UUID: regenerated every time the case enters
    (or re-enters) a HITL interrupt, giving auditability at the stage level.
  - error_log accumulates ValidationError strings from every agent; the QCAgent
    and pipeline router read this to decide HITL routing.
  - partial_e2b: dict of E2B(R3) field-code → value, written incrementally
    by each agent. NarrativeAgent finalises the full E2B XML/JSON from this.
  - graph_payload: serialized networkx DiGraph JSON (case-level MAGMA graph).
    Populated by ExtractionAgent after entity extraction; QCAgent reads it for
    temporal and causal graph assertions.
  - hitl_correction: dict containing the human reviewer's corrected payload
    injected back by the HITL FastAPI server; triggers graph re-entry.

Compliance:
  - 21 CFR Part 11: trace_id + run_id provide e-signature-quality linkage
    to the audit_log for every state transition.
  - CIOMS WG XIV Principle 4 (Transparency): prompt_version tracked per-agent
    in audit records, not in state (avoids bloating the checkpoint).
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from typing_extensions import TypedDict

from schemas.triage import TriageOutput, RiskTier, OversightMode
from schemas.extraction import ExtractedCaseEntities
from schemas.causality import CausalityMatrix
from schemas.coding import CodedEvent
from schemas.listedness import ListednessEvaluation
from schemas.qc import QCReport
from schemas.audit import HITLStage


def _new_review_id() -> str:
    """Generate a fresh per-stage review UUID."""
    return str(uuid.uuid4())


class GraphState(TypedDict, total=False):
    """
    The single shared blackboard propagated through the LangGraph pipeline.

    Fields are intentionally Optional (total=False) because each agent writes
    only its own section. Upstream agents must NOT overwrite downstream fields.

    Routing fields (set by the pipeline router, not agents):
      current_stage: name of the agent node currently executing
      next_stage:    name of the next node to execute (set by router)
      pipeline_halted: True if the case is waiting in the HITL queue

    Agent output fields (written once, never overwritten):
      triage_output, extracted_entities, causality_matrix, coded_events,
      listedness_evaluations, qc_report

    Audit / governance fields:
      trace_id:    LangGraph thread_id (immutable from start → end)
      run_id:      LangGraph run_id (new per pipeline invocation)
      review_id:   Per-stage UUID; refreshed on each HITL interrupt
      error_log:   Accumulated ValidationError and routing error strings

    MAGMA fields:
      graph_payload: serialized networkx DiGraph (JSON) for the case graph
    """
    # ── Identity ────────────────────────────────────────────────────────────
    case_id:          str
    raw_narrative:    str
    narrative_hash:   str           # SHA-256 of raw_narrative
    source_type:      Optional[str]
    country:          Optional[str]
    received_date:    Optional[str]

    # ── Routing ─────────────────────────────────────────────────────────────
    trace_id:         str           # LangGraph thread_id — immutable
    run_id:           str           # LangGraph run_id
    review_id:        str           # Stage-level UUID; refresh on HITL
    current_stage:    Optional[str]
    next_stage:       Optional[str]
    pipeline_halted:  bool          # True = waiting in HITL queue
    risk_tier:        Optional[RiskTier]
    oversight_mode:   Optional[OversightMode]

    # ── Agent Outputs ────────────────────────────────────────────────────────
    triage_output:          Optional[TriageOutput]
    extracted_entities:     Optional[ExtractedCaseEntities]
    causality_matrix:       Optional[CausalityMatrix]
    coded_events:           Optional[list[CodedEvent]]
    listedness_evaluations: Optional[list[ListednessEvaluation]]
    qc_report:              Optional[QCReport]
    narrative_text:         Optional[str]   # Alias — set by NarrativeAgent (narrative_text)
    final_narrative:        Optional[str]   # Final E2B narrative prose (H.1)
    pipeline_complete:      bool            # True when NarrativeAgent finishes successfully

    # ── Audit ────────────────────────────────────────────────────────────────
    error_log:        list[str]     # Accumulated errors from all agents
    hitl_stage:       Optional[HITLStage]   # Stage that triggered HITL

    # ── HITL Re-injection ────────────────────────────────────────────────────
    hitl_correction:  Optional[dict[str, Any]]
    # Set by HITL FastAPI server; router uses this to re-enter the graph
    # at the stage specified in hitl_stage after human correction.

    # ── E2B Partial Blackboard ───────────────────────────────────────────────
    partial_e2b:      dict[str, Any]
    # E2B(R3) field-code → value; each agent appends its fields.
    # NarrativeAgent reads this to finalise the full E2B(R3) JSON/XML output.

    # ── MAGMA Case Graph ─────────────────────────────────────────────────────
    graph_payload:    Optional[str]
    # networkx DiGraph serialized as JSON (node_link_data format).
    # Written by ExtractionAgent; read by QCAgent for graph assertions.


def initial_state(
    case_id:       str,
    raw_narrative: str,
    narrative_hash: str,
    trace_id:      Optional[str] = None,
    run_id:        Optional[str] = None,
    source_type:   Optional[str] = None,
    country:       Optional[str] = None,
    received_date: Optional[str] = None,
) -> GraphState:
    """
    Factory function — creates a fully initialised GraphState for a new case.
    All mutable accumulator fields (error_log, partial_e2b) are initialised
    to empty containers rather than None to avoid None-check boilerplate in agents.
    """
    return GraphState(
        case_id=case_id,
        raw_narrative=raw_narrative,
        narrative_hash=narrative_hash,
        source_type=source_type,
        country=country,
        received_date=received_date,

        trace_id=trace_id or str(uuid.uuid4()),
        run_id=run_id or str(uuid.uuid4()),
        review_id=_new_review_id(),
        current_stage=None,
        next_stage=None,
        pipeline_halted=False,
        risk_tier=None,
        oversight_mode=None,

        triage_output=None,
        extracted_entities=None,
        causality_matrix=None,
        coded_events=None,
        listedness_evaluations=None,
        qc_report=None,
        narrative_text=None,
        final_narrative=None,
        pipeline_complete=False,

        error_log=[],
        hitl_stage=None,
        hitl_correction=None,
        partial_e2b={},
        graph_payload=None,
    )


def refresh_review_id(state: GraphState) -> GraphState:
    """
    Returns an updated state dict with a fresh review_id.
    Called by the pipeline router whenever a HITL interrupt fires,
    so that each HITL cycle gets a distinct, auditable review_id.
    LangGraph nodes must return a dict of updated keys; do NOT mutate in place.
    """
    return {**state, "review_id": _new_review_id()}  # type: ignore[return-value]
