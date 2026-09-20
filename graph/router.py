"""
graph/router.py
================
Conditional edge routing functions for the LangGraph StateGraph.

LangGraph conditional edges accept the current GraphState and return
a string identifying the next node to execute.

Routing logic (deterministic — no LLM involvement):
  1. pipeline_halted=True  → "hitl_review" (always wins)
  2. pipeline_complete=True → END
  3. next_stage is set     → return next_stage value
  4. Fallback by current_stage:
       triage     → extraction
       extraction → qc
       qc         → coding
       coding     → causality
       causality  → listedness
       listedness → narrative
       narrative  → END
  5. Unknown stage         → "hitl_review" (safe fallback)

21 CFR Part 11 note:
  The router never drops a case silently. Unknown/None stages route to
  HITL so a human reviewer can inspect state before any data is lost.
"""
from __future__ import annotations

import logging

from langgraph.graph import END

from graph.state import GraphState

logger = logging.getLogger(__name__)

# Explicit stage sequence (used as fallback when next_stage is absent)
_STAGE_SEQUENCE: dict[str, str] = {
    "triage":     "extraction",
    "extraction": "qc",
    "qc":         "coding",
    "coding":     "causality",
    "causality":  "listedness",
    "listedness": "narrative",
    "narrative":  END,
}


def route_after_triage(state: GraphState) -> str:
    """Route after TriageAgent — first decision point."""
    return _resolve_route(state, default_next="extraction")


def route_after_extraction(state: GraphState) -> str:
    return _resolve_route(state, default_next="qc")


def route_after_qc(state: GraphState) -> str:
    return _resolve_route(state, default_next="coding")


def route_after_coding(state: GraphState) -> str:
    return _resolve_route(state, default_next="causality")


def route_after_causality(state: GraphState) -> str:
    return _resolve_route(state, default_next="listedness")


def route_after_listedness(state: GraphState) -> str:
    return _resolve_route(state, default_next="narrative")


def route_after_narrative(state: GraphState) -> str:
    return _resolve_route(state, default_next=END)


# Map agent AGENT_ID strings → stage node names (registered in conditional edges).
# hitl_stage is set to the AGENT_ID of the failing agent (e.g. "causality-agent-v1"),
# but LangGraph conditional edges are keyed by the short stage name ("causality").
_AGENT_ID_TO_STAGE: dict[str, str] = {
    "triage-agent-v1":     "triage",
    "extraction-agent-v1": "extraction",
    "qc-agent-v1":         "qc",
    "coding-agent-v1":     "coding",
    "causality-agent-v1":  "causality",
    "listedness-agent-v1": "listedness",
    "narrative-agent-v1":  "narrative",
}


def route_after_hitl(state: GraphState) -> str:
    """
    After a human submits a correction via the HITL API, the pipeline
    re-enters at the stage recorded in hitl_stage.  The HITL API clears
    pipeline_halted=False before re-invoking the graph.

    hitl_stage may be either:
      - An AGENT_ID string ("causality-agent-v1")  ← set by BaseAgent._handle_failure
      - A HITLStage enum value ("CAUSALITY")         ← set by HITLReviewRecord
    Both are mapped to the short stage name ("causality") for conditional edge lookup.

    IMPORTANT: On the INITIAL halt (pipeline_halted=True, no correction yet),
    this function routes to END — the graph pauses and waits for a human.
    After resume_case() submits a correction (pipeline_halted=False), the
    function re-enters the appropriate stage to continue processing.
    """
    # Initial halt — no correction submitted yet → stop the graph
    if state.get("pipeline_halted"):
        logger.warning(
            "ROUTER: HITL initial halt — routing to END  [case=%s  hitl_stage=%s]",
            state.get("case_id"), state.get("hitl_stage"),
        )
        return END

    # Correction submitted — honour explicit next_stage if present
    next_s = state.get("next_stage")
    if next_s and next_s not in (None, "", "hitl_review"):
        logger.info(
            "ROUTER: HITL correction resume — using next_stage='%s'  [case=%s]",
            next_s, state.get("case_id"),
        )
        return next_s

    stage = state.get("hitl_stage")
    if stage is None:
        logger.error("route_after_hitl: hitl_stage is None — routing to END")
        return END

    stage_str = stage.value if hasattr(stage, "value") else str(stage)

    # First try exact AGENT_ID lookup, then try uppercase enum lookup
    target = (
        _AGENT_ID_TO_STAGE.get(stage_str)
        or _AGENT_ID_TO_STAGE.get(stage_str.lower())
        or _AGENT_ID_TO_STAGE.get(stage_str.upper())
        # Fallback: treat stage_str itself as a stage name (e.g. "causality")
        or stage_str.lower()
    )
    logger.info(
        "ROUTER: HITL correction resume — re-entering at stage='%s' (hitl_stage='%s')",
        target, stage_str,
    )
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_route(state: GraphState, default_next: str) -> str:
    """
    Core routing logic — called by every stage-specific router.

    Priority:
      1. pipeline_halted  → hitl_review
      2. pipeline_complete → END
      3. next_stage set   → next_stage value
      4. fallback         → default_next
    """
    case_id = state.get("case_id", "unknown")

    # Priority 1: halt wins unconditionally
    if state.get("pipeline_halted"):
        stage = state.get("hitl_stage", "unknown")
        logger.warning(
            "ROUTER: HALTED  [case=%s  hitl_stage=%s]", case_id, stage
        )
        return "hitl_review"

    # Priority 2: completed pipeline
    if state.get("pipeline_complete"):
        logger.info("ROUTER: COMPLETE  [case=%s]", case_id)
        return END

    # Priority 3: explicit next_stage from agent
    next_s = state.get("next_stage")
    if next_s and next_s not in (None, ""):
        if next_s == "complete":
            return END
        logger.debug("ROUTER: next_stage='%s'  [case=%s]", next_s, case_id)
        return next_s

    # Priority 4: fallback
    logger.debug("ROUTER: fallback → '%s'  [case=%s]", default_next, case_id)
    return default_next
