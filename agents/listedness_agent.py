"""
agents/listedness_agent.py
===========================
Listedness Agent — Stage 6 of the ICSR processing pipeline.

Responsibilities:
  1. For each coded adverse event, determine if it is LISTED or UNLISTED in the
     Reference Safety Information (RSI) from FDA DailyMed SPL XML
  2. LISTED  → event appears in the drug's approved prescribing information
  3. UNLISTED → event is unexpected → sets expedited_reporting=True if event is serious
  4. UNKNOWN  → RSI not accessible or assessment indeterminate → HITL

DailyMed API:
  - Drug lookup: GET https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json?drug_name=<name>
  - SPL content: GET https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/<setId>/sections.json
  - Target sections: Adverse Reactions (10), Warnings and Precautions (5)
  - Search sections for coded term match (case-insensitive, substring)

Regulatory context:
  - Unlisted + serious → 15-day expedited report (21 CFR 314.81(b)(1) / ICH E2D)
  - Listed + any seriousness → periodic report only

Output keys: listedness_evaluations, partial_e2b, current_stage, next_stage
"""
from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient
from schemas.coding import CodedEvent, CodingStatus
from schemas.extraction import DrugRole, ExtractedCaseEntities, SuspectDrug
from schemas.listedness import ListednessEvaluation, ListednessStatus

logger = logging.getLogger(__name__)

PROMPT_VERSION  = "listedness-prompt-v1.0"
_DAILYMED_BASE  = "https://dailymed.nlm.nih.gov/dailymed/services/v2"
_REQUEST_TIMEOUT = 15   # seconds
_RATE_LIMIT_S    = 0.5  # 500ms between DailyMed requests (polite crawl)


class ListednessAgent(BaseAgent):
    """
    Stage 6 Listedness Agent — FDA DailyMed SPL-based RSI lookup.

    Parameters
    ----------
    llm      : OllamaClient  — shared inference client
    audit_db : AuditDB       — 21 CFR Part 11 audit log
    session  : requests.Session (optional — injected for testing)
    """

    AGENT_ID = "listedness-agent-v1"

    def __init__(
        self,
        llm:      OllamaClient,
        audit_db: AuditDB,
        session:  Optional[requests.Session] = None,
    ) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)
        self._session = session or requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ICSR-PV-Engine/1.0 (pharmacovigilance research; contact: local)",
        })
        # In-process cache: drug_name -> set_id -> section_text
        self._spl_cache: dict[str, Optional[str]] = {}

    def _run_inner(self, state: GraphState) -> dict:
        case_id      = state["case_id"]
        entities: Optional[ExtractedCaseEntities] = state.get("extracted_entities")
        coded_events: Optional[list[CodedEvent]]  = state.get("coded_events")

        if entities is None:
            raise ValueError(f"ListednessAgent: extracted_entities missing for case={case_id}")
        if not coded_events:
            raise ValueError(f"ListednessAgent: coded_events missing for case={case_id}")

        logger.info("ListednessAgent: evaluating %d events for case=%s",
                    len(coded_events), case_id)

        # Build lookup map: event_node_id → VerbatimEvent
        event_map = {e.node_id: e for e in entities.verbatim_events}

        # Primary suspect drug (first SUSPECT drug)
        suspect_drugs = [d for d in entities.suspect_drugs if d.drug_role == DrugRole.SUSPECT]
        primary_drug  = suspect_drugs[0] if suspect_drugs else None

        evaluations: list[ListednessEvaluation] = []
        needs_hitl   = False

        for ce in coded_events:
            verbatim_event = event_map.get(ce.event_node_id)
            is_serious     = verbatim_event.serious if verbatim_event else False

            # Determine search term for RSI lookup
            search_term    = ce.oae_term or ce.ctcae_term or ce.verbatim_term
            drug_node_id   = primary_drug.node_id if primary_drug else "UNKNOWN"
            drug_name      = primary_drug.drug_name if primary_drug else "UNKNOWN"

            ev = self._evaluate_listedness(
                event_node_id = ce.event_node_id,
                drug_node_id  = drug_node_id,
                verbatim_term = ce.verbatim_term,
                coded_term    = search_term,
                drug_name     = drug_name,
                is_serious    = is_serious,
            )
            evaluations.append(ev)

            if ev.listedness_status == ListednessStatus.UNKNOWN:
                needs_hitl = True
            elif ev.expedited_reporting:
                logger.warning(
                    "ListednessAgent: EXPEDITED REPORTING triggered "
                    "for case=%s event='%s' drug='%s'",
                    case_id, ce.verbatim_term, drug_name
                )

        # ── Partial E2B blackboard ────────────────────────────────────────────
        partial_e2b: dict = dict(state.get("partial_e2b") or {})
        for ev in evaluations:
            k = f"E.i.3a.{ev.event_node_id}"   # Placeholder for listedness field
            partial_e2b[k] = ev.listedness_status.value

        partial: dict = {
            "listedness_evaluations": evaluations,
            "partial_e2b":            partial_e2b,
            "current_stage":          "listedness",
        }

        if needs_hitl:
            logger.warning(
                "ListednessAgent: UNKNOWN listedness for case=%s → HITL", case_id
            )
            partial.update({
                "pipeline_halted": True,
                "hitl_stage":      "listedness",
                "next_stage":      "hitl_review",
            })
        else:
            partial["next_stage"] = "narrative"

        return partial

    # ── DailyMed lookup ───────────────────────────────────────────────────────

    def _evaluate_listedness(
        self,
        event_node_id: str,
        drug_node_id:  str,
        verbatim_term: str,
        coded_term:    str,
        drug_name:     str,
        is_serious:    bool,
    ) -> ListednessEvaluation:
        """Perform RSI lookup against FDA DailyMed."""
        cache_key = drug_name.lower()

        if cache_key not in self._spl_cache:
            self._spl_cache[cache_key] = self._fetch_adverse_reactions_text(drug_name)

        adverse_reactions_text = self._spl_cache[cache_key]

        if adverse_reactions_text is None:
            return ListednessEvaluation(
                event_node_id    = event_node_id,
                drug_node_id     = drug_node_id,
                verbatim_term    = verbatim_term,
                coded_term       = coded_term,
                listedness_status= ListednessStatus.UNKNOWN,
                rsi_source       = None,
                confidence       = 0.4,
                agent_id         = self.AGENT_ID,
                prompt_version   = PROMPT_VERSION,
            )

        # Case-insensitive substring search
        is_listed = self._term_found_in_text(coded_term, adverse_reactions_text)
        if not is_listed:
            is_listed = self._term_found_in_text(verbatim_term, adverse_reactions_text)

        status     = ListednessStatus.LISTED if is_listed else ListednessStatus.UNLISTED
        expedited  = (status == ListednessStatus.UNLISTED and is_serious)
        confidence = 0.90 if is_listed else 0.85   # DailyMed text is authoritative

        return ListednessEvaluation(
            event_node_id      = event_node_id,
            drug_node_id       = drug_node_id,
            verbatim_term      = verbatim_term,
            coded_term         = coded_term,
            listedness_status  = status,
            rsi_source         = f"FDA DailyMed SPL — drug: {drug_name}",
            rsi_section_text   = adverse_reactions_text[:2000],
            expedited_reporting= expedited,
            confidence         = confidence,
            agent_id           = self.AGENT_ID,
            prompt_version     = PROMPT_VERSION,
        )

    def _fetch_adverse_reactions_text(self, drug_name: str) -> Optional[str]:
        """
        Fetch the Adverse Reactions section text from FDA DailyMed.
        Returns the combined text from section code 34084-4 (Adverse Reactions)
        and 43685-7 (Warnings and Precautions).
        Returns None on any network or parse failure.
        """
        try:
            # Step 1: Look up SPL by drug name
            resp = self._session.get(
                f"{_DAILYMED_BASE}/spls.json",
                params={"drug_name": drug_name, "limit": 1},
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            spls = resp.json().get("data", [])
            if not spls:
                logger.info("DailyMed: no SPL found for drug '%s'", drug_name)
                return None

            set_id = spls[0].get("setid") or spls[0].get("id")
            if not set_id:
                return None

            time.sleep(_RATE_LIMIT_S)

            # Step 2: Fetch sections for this SPL
            resp2 = self._session.get(
                f"{_DAILYMED_BASE}/spls/{set_id}/sections.json",
                timeout=_REQUEST_TIMEOUT,
            )
            resp2.raise_for_status()
            sections_data = resp2.json().get("data", [])

            # Collect Adverse Reactions + Warnings sections
            target_codes = {"34084-4", "43685-7", "34071-1"}
            texts: list[str] = []
            for section in sections_data:
                if section.get("loinc_code") in target_codes:
                    text = section.get("text", "")
                    if text:
                        texts.append(text)

            combined = " ".join(texts)
            if not combined.strip():
                return None

            return combined

        except requests.Timeout:
            logger.warning("DailyMed: timeout fetching SPL for '%s'", drug_name)
            return None
        except Exception as exc:
            logger.warning("DailyMed: failed to fetch SPL for '%s': %s", drug_name, exc)
            return None

    @staticmethod
    def _term_found_in_text(term: str, text: str) -> bool:
        """
        Case-insensitive substring search.
        Also tries significant word matching for multi-word terms.
        """
        if not term or not text:
            return False
        term_lower = term.lower()
        text_lower = text.lower()

        if term_lower in text_lower:
            return True

        # Multi-word: check if the longest meaningful word appears
        words = [w for w in re.split(r"\W+", term_lower) if len(w) >= 5]
        if words:
            return all(w in text_lower for w in words)

        return False
