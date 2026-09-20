"""
graph/pipeline.py
==================
LangGraph StateGraph pipeline — the main orchestration entry point.

Pipeline stages (left to right):
  triage → extraction → qc → coding → causality → listedness → narrative → END

Each stage may route to hitl_review via the conditional router if:
  - pipeline_halted=True  (agent set this due to low confidence / validation error)
  - stage is "hitl_review" explicitly

After human correction, the hitl_node's conditional edge re-enters
the graph at the stage recorded in hitl_stage (e.g., "causality").

Architecture:
  - StateGraph[GraphState]: typed state machine
  - SqliteSaver checkpointer: persists state between HITL interruptions
  - thread_id = trace_id: deterministic, one-per-case

Hardware constraint (16 GB RAM):
  - All agents share one OllamaClient with Semaphore(1)
  - FAISS indexes loaded once, held in memory (~200 MB)
  - Pipeline is synchronous (no async) to keep memory footprint minimal

Usage:
  factory  = NodeFactory(llm, audit_db, rxnorm, oae_idx, ctcae_idx, neg_path)
  pipeline = ICSRPipeline(factory, checkpointer)
  result   = pipeline.run_case(case_id, raw_narrative, ...)

Public API:
  ICSRPipeline.run_case(...)       → returns final GraphState
  ICSRPipeline.resume_case(...)    → resume from HITL correction
  ICSRPipeline.get_state(...)      → fetch checkpoint state for a thread
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from graph.hitl_interrupt import HITLQueue, build_resume_state, get_checkpointer
from graph.nodes import NodeFactory
from graph.router import (
    route_after_causality,
    route_after_coding,
    route_after_extraction,
    route_after_hitl,
    route_after_listedness,
    route_after_narrative,
    route_after_qc,
    route_after_triage,
)
from graph.state import GraphState, initial_state

logger = logging.getLogger(__name__)


class ICSRPipeline:
    """
    Compiled LangGraph StateGraph for ICSR processing.

    Parameters
    ----------
    factory      : NodeFactory — all agent node callables
    checkpointer : SqliteSaver — LangGraph checkpoint persistence
    hitl_queue   : HITLQueue   — in-memory HITL case queue (injected or default)
    """

    def __init__(
        self,
        factory:      NodeFactory,
        checkpointer: SqliteSaver | None  = None,
        hitl_queue:   HITLQueue | None    = None,
    ) -> None:
        self._factory      = factory
        self._checkpointer = checkpointer if checkpointer is not None else get_checkpointer()
        self._hitl_queue   = hitl_queue   if hitl_queue   is not None else HITLQueue()
        self._graph        = self._build_graph()
        logger.info("ICSRPipeline: compiled graph ready")

    # ── Graph Construction ────────────────────────────────────────────────────

    def _build_graph(self):
        """Assemble and compile the LangGraph StateGraph."""
        nodes = self._factory.node_map()
        sg    = StateGraph(GraphState)

        # Register all nodes
        for name, fn in nodes.items():
            sg.add_node(name, fn)

        # Entry point
        sg.set_entry_point("triage")

        # Conditional edges (agent output → next stage via router)
        sg.add_conditional_edges(
            "triage",
            route_after_triage,
            {
                "extraction":  "extraction",
                "hitl_review": "hitl_review",
                END:            END,
            },
        )
        sg.add_conditional_edges(
            "extraction",
            route_after_extraction,
            {
                "qc":          "qc",
                "hitl_review": "hitl_review",
                END:            END,
            },
        )
        sg.add_conditional_edges(
            "qc",
            route_after_qc,
            {
                "coding":      "coding",
                "hitl_review": "hitl_review",
                END:            END,
            },
        )
        sg.add_conditional_edges(
            "coding",
            route_after_coding,
            {
                "causality":   "causality",
                "hitl_review": "hitl_review",
                END:            END,
            },
        )
        sg.add_conditional_edges(
            "causality",
            route_after_causality,
            {
                "listedness":  "listedness",
                "hitl_review": "hitl_review",
                END:            END,
            },
        )
        sg.add_conditional_edges(
            "listedness",
            route_after_listedness,
            {
                "narrative":   "narrative",
                "hitl_review": "hitl_review",
                END:            END,
            },
        )
        sg.add_conditional_edges(
            "narrative",
            route_after_narrative,
            {
                "hitl_review": "hitl_review",
                END:            END,
            },
        )

        # HITL node re-enters at the corrected stage
        # All valid stage names are listed as possible resume targets
        sg.add_conditional_edges(
            "hitl_review",
            route_after_hitl,
            {
                "triage":      "triage",
                "extraction":  "extraction",
                "qc":          "qc",
                "coding":      "coding",
                "causality":   "causality",
                "listedness":  "listedness",
                "narrative":   "narrative",
                END:            END,
            },
        )

        return sg.compile(checkpointer=self._checkpointer)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_case(
        self,
        case_id:       str,
        raw_narrative: str,
        source_type:   Optional[str] = None,
        country:       Optional[str] = None,
        received_date: Optional[str] = None,
        thread_id:     Optional[str] = None,
    ) -> GraphState:
        """
        Run a new case through the full pipeline.

        Parameters
        ----------
        case_id       : Unique case identifier (e.g. "ICSR-20240101-001")
        raw_narrative : Free-text CIOMS-I narrative
        source_type   : Optional source classification string
        country       : Optional ISO country of occurrence
        received_date : Optional date received (YYYY-MM-DD)
        thread_id     : Optional LangGraph thread_id; defaults to a new UUID

        Returns
        -------
        Final GraphState (may have pipeline_halted=True if HITL triggered)
        """
        narrative_hash = hashlib.sha256(raw_narrative.encode()).hexdigest()
        _thread_id     = thread_id or str(uuid.uuid4())

        state = initial_state(
            case_id        = case_id,
            raw_narrative  = raw_narrative,
            narrative_hash = narrative_hash,
            trace_id       = _thread_id,
            run_id         = str(uuid.uuid4()),
            source_type    = source_type,
            country        = country,
            received_date  = received_date,
        )

        config = {"configurable": {"thread_id": _thread_id}}

        logger.info(
            "ICSRPipeline.run_case: case=%s thread_id=%s", case_id, _thread_id
        )

        final_state = self._invoke(state, config)

        # Enqueue to HITL queue if halted
        if final_state.get("pipeline_halted"):
            review_id = final_state.get("review_id", str(uuid.uuid4()))
            self._hitl_queue.enqueue(
                review_id  = review_id,
                thread_id  = _thread_id,
                state_snap = dict(final_state),
            )
            logger.warning(
                "ICSRPipeline: case=%s HALTED for HITL review  review_id=%s",
                case_id, review_id
            )

        return final_state

    def resume_case(
        self,
        review_id:  str,
        correction: dict[str, Any],
    ) -> GraphState:
        """
        Resume a halted case after a human correction.

        Parameters
        ----------
        review_id  : The review_id from the halted GraphState
        correction : Partial dict of corrected state fields

        Returns
        -------
        Updated GraphState after re-running from the corrected stage
        """
        entry = self._hitl_queue.dequeue(review_id)
        if entry is None:
            raise ValueError(f"No halted case found for review_id={review_id!r}")

        thread_id   = entry["thread_id"]
        state_snap  = entry["state_snap"]
        config      = {"configurable": {"thread_id": thread_id}}

        merged = build_resume_state(state_snap, correction)

        logger.info(
            "ICSRPipeline.resume_case: review_id=%s thread_id=%s correction_keys=%s",
            review_id, thread_id, list(correction.keys())
        )

        # Patch the checkpoint with the merged (corrected) state.
        # update_state() writes the correction into the MemorySaver at the
        # hitl_review node boundary so that route_after_hitl runs next with
        # the updated values (pipeline_halted=False, next_stage=<target stage>).
        self._graph.update_state(config, merged, as_node="hitl_review")

        # Resume: stream(None, config) re-enters from the updated checkpoint.
        final_state = self._invoke(None, config)

        # Re-enqueue if still halted (e.g., second HITL stage)
        if final_state.get("pipeline_halted"):
            new_review_id = final_state.get("review_id", str(uuid.uuid4()))
            self._hitl_queue.enqueue(
                review_id  = new_review_id,
                thread_id  = thread_id,
                state_snap = dict(final_state),
            )
            logger.warning(
                "ICSRPipeline.resume_case: still HALTED after correction  "
                "new_review_id=%s", new_review_id
            )

        return final_state

    def get_state(self, thread_id: str) -> Optional[GraphState]:
        """
        Retrieve the latest checkpoint state for a given thread_id.
        Returns None if no checkpoint exists.
        """
        try:
            snapshot = self._graph.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            return snapshot.values if snapshot else None
        except Exception as exc:
            logger.warning("ICSRPipeline.get_state: thread_id=%s error=%s", thread_id, exc)
            return None

    @property
    def hitl_queue(self) -> HITLQueue:
        return self._hitl_queue

    # ── Internal ──────────────────────────────────────────────────────────────

    def _invoke(self, state: Optional[GraphState], config: dict) -> GraphState:
        """
        Invoke the compiled graph and return the last emitted state.

        state=None : resume from the current checkpoint (HITL resume path).
                     last_state is seeded from the checkpoint snapshot.
        state=dict : fresh invocation (normal run_case path).
        """
        if state is None:
            # Seed from checkpoint so merge-dict logic has a valid base
            snap = self._graph.get_state(config)
            last_state: GraphState = dict(snap.values) if snap and snap.values else {}  # type: ignore
        else:
            last_state = state  # type: ignore

        for event in self._graph.stream(state, config=config):
            for node_name, node_output in event.items():
                if isinstance(node_output, dict):
                    last_state = {**last_state, **node_output}  # type: ignore
                    logger.debug(
                        "STREAM: node=%s  keys=%s",
                        node_name, list(node_output.keys())
                    )
        return last_state
