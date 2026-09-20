"""
graph/nodes.py
==============
LangGraph node wrappers for each pipeline agent.

Each node function follows the LangGraph contract:
  - Accepts GraphState (TypedDict)
  - Returns a partial dict of updated keys
  - Does NOT mutate state in place

Node registry maps stage names → callable, allowing the pipeline router
to resolve next_stage strings to the actual LangGraph node functions.

Hardware constraint (i5-13420H / 16 GB RAM):
  - Agents share a single OllamaClient with threading.Semaphore(1)
  - Nodes are called sequentially within a case; LangGraph handles
    the sequential execution within a single thread.

Dependency injection:
  - NodeFactory.build() wires all dependencies at startup and returns
    the frozen dict of node callables used by pipeline.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from agents.causality_agent import CausalityAgent
from agents.coding_agent import CodingAgent
from agents.extraction_agent import ExtractionAgent
from agents.listedness_agent import ListednessAgent
from agents.narrative_agent import NarrativeAgent
from agents.qc_agent import QCAgent
from agents.triage_agent import TriageAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.faiss_index import FAISSIndex
from infra.ollama_client import OllamaClient
from infra.rxnorm_client import RxNormClient

logger = logging.getLogger(__name__)

# Type alias
NodeFn = Callable[[GraphState], dict[str, Any]]


class NodeFactory:
    """
    Wires all agent dependencies at startup time and exposes
    the fully-initialised node callables for LangGraph registration.

    Parameters
    ----------
    llm       : shared OllamaClient (temperature=0.0 enforced)
    audit_db  : shared AuditDB (21 CFR Part 11)
    rxnorm    : RxNormClient (SQLite-cached NLM lookups)
    oae_idx   : FAISSIndex for OAE adverse-event coding
    ctcae_idx : FAISSIndex for CTCAE fallback coding (may be None)
    neg_path  : Path to clinical_negatives.json hard-negative ontology
    """

    def __init__(
        self,
        llm:       OllamaClient,
        audit_db:  AuditDB,
        rxnorm:    RxNormClient,
        oae_idx:   FAISSIndex | None       = None,
        ctcae_idx: FAISSIndex | None       = None,
        neg_path:  Path | None             = None,
        session:   Any | None              = None,  # Optional requests.Session for ListednessAgent
    ) -> None:
        self._llm      = llm
        self._audit_db = audit_db

        _neg = neg_path or Path("data/clinical_negatives.json")

        self._triage    = TriageAgent(llm=llm, audit_db=audit_db)
        self._extract   = ExtractionAgent(llm=llm, audit_db=audit_db, rxnorm=rxnorm)
        self._qc        = QCAgent(llm=llm, audit_db=audit_db, neg_path=_neg)
        self._coding    = CodingAgent(
            llm=llm, audit_db=audit_db,
            oae_index=oae_idx, ctcae_index=ctcae_idx,
        )
        self._causality  = CausalityAgent(llm=llm, audit_db=audit_db)
        self._listedness = ListednessAgent(llm=llm, audit_db=audit_db, session=session)
        self._narrative  = NarrativeAgent(llm=llm, audit_db=audit_db)

        logger.info("NodeFactory: all agents initialised")

    # ── Node callables ────────────────────────────────────────────────────────

    def triage_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → triage  [case=%s]", state["case_id"])
        return self._triage.run(state)

    def extraction_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → extraction  [case=%s]", state["case_id"])
        return self._extract.run(state)

    def qc_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → qc  [case=%s]", state["case_id"])
        return self._qc.run(state)

    def coding_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → coding  [case=%s]", state["case_id"])
        return self._coding.run(state)

    def causality_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → causality  [case=%s]", state["case_id"])
        return self._causality.run(state)

    def listedness_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → listedness  [case=%s]", state["case_id"])
        return self._listedness.run(state)

    def narrative_node(self, state: GraphState) -> dict:
        logger.info("PIPELINE: → narrative  [case=%s]", state["case_id"])
        return self._narrative.run(state)

    def hitl_node(self, state: GraphState) -> dict:
        """
        HITL interrupt node — LangGraph pauses here when pipeline_halted=True.
        The pipeline resumes after a human correction is injected via the
        HITL FastAPI server (POST /api/v1/review/{review_id}/submit).

        This node itself is a no-op; LangGraph's interrupt mechanism handles
        the pause/resume. We log the halt for audit visibility.
        """
        logger.warning(
            "PIPELINE: ⚡ HITL interrupt  [case=%s  stage=%s  review_id=%s]",
            state.get("case_id"), state.get("hitl_stage"), state.get("review_id"),
        )
        # Return empty dict — LangGraph will use the state injected by the
        # human correction (hitl_correction field) to re-enter the graph.
        return {}

    # ── Node map (used by pipeline.py) ───────────────────────────────────────

    def node_map(self) -> dict[str, NodeFn]:
        """Return all node callables keyed by their stage name."""
        return {
            "triage":     self.triage_node,
            "extraction": self.extraction_node,
            "qc":         self.qc_node,
            "coding":     self.coding_node,
            "causality":  self.causality_node,
            "listedness": self.listedness_node,
            "narrative":  self.narrative_node,
            "hitl_review":self.hitl_node,
        }
