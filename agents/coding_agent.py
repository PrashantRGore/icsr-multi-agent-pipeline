"""
agents/coding_agent.py
=======================
Coding Agent — Stage 4 of the ICSR processing pipeline.

Responsibilities:
  1. Map each verbatim adverse event term to a standardised ontology code
  2. Primary: OAE (Ontology of Adverse Events) via FAISS k-NN search
  3. Fallback: NCI CTCAE v5 when OAE confidence < OAE_CONFIDENCE_THRESHOLD (0.80)
  4. NEEDS_MANUAL when both OAE and CTCAE return scores below threshold

Output: list[CodedEvent], one per VerbatimEvent
Output keys: coded_events, partial_e2b, current_stage, next_stage

Design decisions:
  - FAISS search is purely deterministic (no LLM involved at this stage).
  - An LLM fallback hint is requested only when FAISS returns no match at all
    (NEEDS_MANUAL case) to provide a suggested term for the HITL reviewer.
  - OAE ID pattern: OAE:XXXXXXX (7 digits)
  - coding_status drives listedness agent input
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.faiss_index import FAISSIndex
from infra.ollama_client import OllamaClient
from schemas.coding import CodedEvent, CodingStatus
from schemas.extraction import ExtractedCaseEntities

logger = logging.getLogger(__name__)

PROMPT_VERSION  = "coding-prompt-v1.0"

OAE_THRESHOLD   = 0.80   # Below → try CTCAE fallback
CTCAE_THRESHOLD = 0.60   # Below → NEEDS_MANUAL

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# LLM fallback prompt — used only when FAISS returns nothing useful
_FALLBACK_SYSTEM = """You are an adverse event coding specialist.
Given a verbatim adverse event term, suggest the closest MedDRA Preferred Term
or CTCAE v5 term name (text only, no code). Output ONLY valid JSON:
{"suggested_term": "<string or null>"}"""


class CodingAgent(BaseAgent):
    """
    Stage 4 Coding Agent — FAISS-first, LLM-fallback-hint.

    Parameters
    ----------
    llm          : OllamaClient — shared inference client (LLM fallback only)
    audit_db     : AuditDB     — 21 CFR Part 11 audit log
    oae_index    : FAISSIndex  — OAE FAISS index (optional — built by script)
    ctcae_index  : FAISSIndex  — CTCAE FAISS index (optional — built by script)
    """

    AGENT_ID = "coding-agent-v1"

    def __init__(
        self,
        llm:         OllamaClient,
        audit_db:    AuditDB,
        oae_index:   Optional[FAISSIndex] = None,
        ctcae_index: Optional[FAISSIndex] = None,
    ) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)
        self._oae   = oae_index
        self._ctcae = ctcae_index

        if self._oae is None:
            logger.warning("CodingAgent: OAE FAISS index not provided — coding will use CTCAE only")
        if self._ctcae is None:
            logger.warning("CodingAgent: CTCAE FAISS index not provided — NEEDS_MANUAL will be frequent")

    def _run_inner(self, state: GraphState) -> dict:
        case_id  = state["case_id"]
        entities: Optional[ExtractedCaseEntities] = state.get("extracted_entities")

        if entities is None:
            raise ValueError(
                f"CodingAgent: extracted_entities missing for case={case_id}"
            )

        logger.info("CodingAgent: coding %d events for case=%s",
                    len(entities.verbatim_events), case_id)

        coded_events: list[CodedEvent] = []
        needs_hitl = False

        for event in entities.verbatim_events:
            coded = self._code_event(event.node_id, event.verbatim_term)
            coded_events.append(coded)
            if coded.coding_status == CodingStatus.NEEDS_MANUAL:
                needs_hitl = True

        # ── Partial E2B blackboard ────────────────────────────────────────────
        partial_e2b: dict = dict(state.get("partial_e2b") or {})
        for ce in coded_events:
            k = f"E.i.2.1b.{ce.event_node_id}"   # LLT / preferred term placeholder
            partial_e2b[k] = ce.oae_term or ce.ctcae_term or ce.verbatim_term

        partial: dict = {
            "coded_events":  coded_events,
            "partial_e2b":   partial_e2b,
            "current_stage": "coding",
        }

        if needs_hitl:
            logger.warning("CodingAgent: NEEDS_MANUAL coding for case=%s → HITL", case_id)
            partial.update({
                "pipeline_halted": True,
                "hitl_stage":      "coding",
                "next_stage":      "hitl_review",
            })
        else:
            partial["next_stage"] = "causality"

        return partial

    # ── Coding helpers ────────────────────────────────────────────────────────

    def _code_event(self, node_id: str, verbatim_term: str) -> CodedEvent:
        """Code a single verbatim term. OAE → CTCAE → NEEDS_MANUAL."""

        # 1. OAE FAISS search
        if self._oae is not None:
            oae_result = self._oae.search(verbatim_term, k=1, threshold=OAE_THRESHOLD)
            if oae_result:
                top = oae_result[0]
                # Attempt to extract OAE ID from metadata
                oae_id  = top.get("oae_id") or top.get("id")
                oae_term= top.get("label")  or top.get("term") or top.get("text", "")
                score   = float(top.get("score", 0.0))

                if score >= OAE_THRESHOLD:
                    # Validate OAE ID format
                    import re
                    if oae_id and not re.match(r"^OAE:\d{7}$", str(oae_id)):
                        oae_id = None   # Non-conforming ID — omit

                    return CodedEvent(
                        event_node_id  = node_id,
                        verbatim_term  = verbatim_term,
                        oae_term       = oae_term,
                        oae_id         = oae_id,
                        oae_confidence = round(score, 4),
                        coding_status  = CodingStatus.AUTO_CODED,
                        agent_id       = self.AGENT_ID,
                        prompt_version = PROMPT_VERSION,
                    )

        # 2. CTCAE FAISS fallback
        if self._ctcae is not None:
            ctcae_result = self._ctcae.search(verbatim_term, k=1, threshold=CTCAE_THRESHOLD)
            if ctcae_result:
                top        = ctcae_result[0]
                ctcae_term = top.get("term") or top.get("label") or top.get("text", "")
                ctcae_grade= top.get("grade")
                score      = float(top.get("score", 0.0))

                if score >= CTCAE_THRESHOLD:
                    return CodedEvent(
                        event_node_id  = node_id,
                        verbatim_term  = verbatim_term,
                        ctcae_term     = ctcae_term,
                        ctcae_grade    = int(ctcae_grade) if ctcae_grade else None,
                        coding_status  = CodingStatus.FALLBACK_CTCAE,
                        agent_id       = self.AGENT_ID,
                        prompt_version = PROMPT_VERSION,
                    )

        # 3. LLM hint (suggestion for HITL reviewer — no auto-coding)
        llm_suggestion = self._get_llm_hint(verbatim_term)

        logger.warning(
            "CodingAgent: NEEDS_MANUAL for '%s' — LLM hint: %s",
            verbatim_term, llm_suggestion
        )

        return CodedEvent(
            event_node_id  = node_id,
            verbatim_term  = verbatim_term,
            meddra_pt      = llm_suggestion,   # Advisory only — not auto-applied
            coding_status  = CodingStatus.NEEDS_MANUAL,
            agent_id       = self.AGENT_ID,
            prompt_version = PROMPT_VERSION,
        )

    def _get_llm_hint(self, verbatim_term: str) -> Optional[str]:
        """Ask the LLM for a coding hint.  Soft-failure — returns None on error."""
        try:
            resp = self._llm.chat(
                system_prompt  = _FALLBACK_SYSTEM,
                user_message   = f"Verbatim term: {verbatim_term}",
                prompt_version = PROMPT_VERSION,
            )
            data = self._parse_llm_json(resp.text)
            return data.get("suggested_term")
        except Exception as exc:
            logger.debug("CodingAgent: LLM hint failed for '%s': %s", verbatim_term, exc)
            return None
