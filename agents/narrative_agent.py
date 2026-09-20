"""
agents/narrative_agent.py
==========================
Narrative Agent — Stage 7 (final) of the ICSR processing pipeline.

Responsibilities:
  1. Synthesise a complete, CIOMS-I / ICH E2B(R3)-structured narrative
     from the pipeline outputs assembled in GraphState
  2. Write the narrative to the partial_e2b blackboard key "H.1" (narrative text)
  3. Verify that all key entities are mentioned (QC self-check)
  4. Compute a final narrative SHA-256 hash for audit trail

The NarrativeAgent uses generate() (raw text, not JSON) because the output
is free-form prose, not structured data.

Output keys: final_narrative, partial_e2b, current_stage, next_stage, pipeline_complete

CIOMS-I narrative structure:
  1. Patient demographics
  2. Suspect drug(s): name, dose, route, indication, dates
  3. Adverse event(s): onset, severity, seriousness, outcome
  4. Dechallenge / Rechallenge
  5. Causality assessment
  6. Listedness determination
  7. Expedited reporting flag (if applicable)
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient
from schemas.causality import CausalityMatrix
from schemas.coding import CodedEvent, CodingStatus
from schemas.extraction import DrugRole, ExtractedCaseEntities
from schemas.listedness import ListednessEvaluation, ListednessStatus
from schemas.triage import TriageOutput

logger = logging.getLogger(__name__)

PROMPT_VERSION = "narrative-prompt-v1.0"

_NARRATIVE_SYSTEM = """\
You are an expert pharmacovigilance (PV) medical writer.
Write a concise, factual CIOMS-I / ICH E2B(R3)-structured ICSR narrative in English.

Structure your narrative in this order (omit sections where data is absent):
1. Patient: [age/sex/weight/country] — demographics
2. Suspect drug(s): name, dose, route, indication, start/stop dates
3. Adverse event(s): verbatim term, onset date, seriousness criteria, outcome
4. Dechallenge (drug stopped → AE outcome) and Rechallenge (drug restarted → AE outcome)
5. Causality assessment: WHO-UMC term + rationale
6. Coding: OAE or CTCAE coded term
7. Listedness: LISTED or UNLISTED in Reference Safety Information
8. Expedited reporting: state if 15-day report obligation applies

Rules:
- Write 150–400 words.
- Use past tense, third-person, objective clinical language.
- Do NOT include PII beyond what is in the context provided.
- Do NOT add information not present in the source data.
- Write ONLY the narrative text — no headers, no JSON, no markdown.
"""


class NarrativeAgent(BaseAgent):
    """
    Stage 7 Narrative Agent — CIOMS-I / E2B(R3) narrative synthesis.

    Parameters
    ----------
    llm      : OllamaClient — shared inference client
    audit_db : AuditDB     — 21 CFR Part 11 audit log
    """

    AGENT_ID = "narrative-agent-v1"

    def __init__(self, llm: OllamaClient, audit_db: AuditDB) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)

    def _run_inner(self, state: GraphState) -> dict:
        case_id = state["case_id"]

        logger.info("NarrativeAgent: composing narrative for case=%s", case_id)

        # ── Assemble context for LLM ─────────────────────────────────────────
        context = self._build_context(state)

        response = self._llm.generate(
            prompt         = _NARRATIVE_SYSTEM + "\n\n" + context,
            prompt_version = PROMPT_VERSION,
        )

        narrative = response.text.strip()

        # ── Self-check: key entities present ─────────────────────────────────
        entities: Optional[ExtractedCaseEntities] = state.get("extracted_entities")
        missing_entities: list[str] = []
        if entities:
            for drug in entities.suspect_drugs:
                if drug.drug_role == DrugRole.SUSPECT:
                    if drug.drug_name.lower() not in narrative.lower():
                        missing_entities.append(f"drug:{drug.drug_name}")
            for event in entities.verbatim_events:
                core_word = event.verbatim_term.split()[0].lower()
                if core_word not in narrative.lower():
                    missing_entities.append(f"event:{event.verbatim_term}")

        if missing_entities:
            logger.warning(
                "NarrativeAgent: self-check — entities missing from narrative for "
                "case=%s: %s", case_id, missing_entities
            )

        # ── Narrative hash (21 CFR Part 11 chain) ────────────────────────────
        narrative_hash = hashlib.sha256(narrative.encode()).hexdigest()

        # ── E2B blackboard finalization ───────────────────────────────────────
        partial_e2b = dict(state.get("partial_e2b") or {})
        partial_e2b["H.1"] = narrative   # ICH E2B(R3) H.1 — case narrative

        logger.info(
            "NarrativeAgent: narrative complete for case=%s "
            "length=%d hash=%s...",
            case_id, len(narrative), narrative_hash[:8]
        )

        return {
            "final_narrative": narrative,
            "narrative_hash":  narrative_hash,    # Updated hash for final audit
            "partial_e2b":     partial_e2b,
            "current_stage":   "narrative",
            "next_stage":      "complete",
            "pipeline_complete": True,
        }

    # ── Context builder ───────────────────────────────────────────────────────

    def _build_context(self, state: GraphState) -> str:
        """
        Assemble a structured text context block from all pipeline outputs
        collected in GraphState. Passed as the user message to the LLM.
        """
        lines: list[str] = [f"CASE ID: {state['case_id']}\n"]

        entities: Optional[ExtractedCaseEntities] = state.get("extracted_entities")
        triage:   Optional[TriageOutput]           = state.get("triage_output")

        # Patient demographics
        if entities:
            demo = []
            if entities.patient_age:        demo.append(f"Age: {entities.patient_age}")
            if entities.patient_sex:        demo.append(f"Sex: {entities.patient_sex}")
            if entities.patient_ethnicity:  demo.append(f"Ethnicity: {entities.patient_ethnicity}")
            if entities.patient_weight_kg:  demo.append(f"Weight: {entities.patient_weight_kg} kg")
            if entities.country_of_occurrence: demo.append(f"Country: {entities.country_of_occurrence}")
            if demo:
                lines.append("PATIENT DEMOGRAPHICS:\n" + "; ".join(demo))

        # Suspect drugs
        if entities:
            suspect = [d for d in entities.suspect_drugs if d.drug_role == DrugRole.SUSPECT]
            concomitant = [d for d in entities.suspect_drugs if d.drug_role != DrugRole.SUSPECT]
            if suspect:
                drug_lines = []
                for d in suspect:
                    parts = [f"Name: {d.drug_name}"]
                    if d.dose:       parts.append(f"Dose: {d.dose}")
                    if d.route:      parts.append(f"Route: {d.route}")
                    if d.indication: parts.append(f"Indication: {d.indication}")
                    if d.start_date: parts.append(f"Start: {d.start_date.to_e2b_str()}")
                    if d.stop_date:  parts.append(f"Stop: {d.stop_date.to_e2b_str()}")
                    parts.append(f"Dechallenge: {d.dechallenge.value}")
                    parts.append(f"Rechallenge: {d.rechallenge.value}")
                    drug_lines.append("  • " + " | ".join(parts))
                lines.append("SUSPECT DRUG(S):\n" + "\n".join(drug_lines))
            if concomitant:
                lines.append("CONCOMITANT DRUGS: " + ", ".join(d.drug_name for d in concomitant))

        # Adverse events
        if entities:
            evt_lines = []
            for e in entities.verbatim_events:
                parts = [f"Term: {e.verbatim_term}"]
                if e.onset_date:           parts.append(f"Onset: {e.onset_date.to_e2b_str()}")
                if e.serious:
                    criteria_str = ", ".join(c.value for c in e.seriousness_criteria)
                    parts.append(f"Serious: Yes ({criteria_str})")
                else:
                    parts.append("Serious: No")
                if e.outcome:              parts.append(f"Outcome: {e.outcome}")
                evt_lines.append("  • " + " | ".join(parts))
            lines.append("ADVERSE EVENT(S):\n" + "\n".join(evt_lines))

        # Triage risk tier
        if triage:
            lines.append(f"RISK TIER: {triage.risk_tier.value}")

        # Causality
        causality_matrix: Optional[CausalityMatrix] = state.get("causality_matrix")
        if causality_matrix:
            c_lines = []
            for ca in causality_matrix.assessments:
                c_lines.append(
                    f"  • {ca.drug_name} → {ca.verbatim_event}: "
                    f"{ca.causality_term.value} (confidence={ca.confidence:.2f})"
                )
            lines.append("CAUSALITY ASSESSMENTS:\n" + "\n".join(c_lines))

        # Coding
        coded_events: Optional[list[CodedEvent]] = state.get("coded_events")
        if coded_events:
            c_lines = []
            for ce in coded_events:
                coded = ce.oae_term or ce.ctcae_term or "NEEDS_MANUAL"
                c_lines.append(f"  • {ce.verbatim_term} → {coded} ({ce.coding_status.value})")
            lines.append("CODED EVENTS:\n" + "\n".join(c_lines))

        # Listedness
        listedness: Optional[list[ListednessEvaluation]] = state.get("listedness_evaluations")
        if listedness:
            l_lines = []
            for lev in listedness:
                exp = " [EXPEDITED REPORT REQUIRED]" if lev.expedited_reporting else ""
                l_lines.append(
                    f"  • {lev.verbatim_term}: {lev.listedness_status.value}{exp}"
                )
            lines.append("LISTEDNESS:\n" + "\n".join(l_lines))

        return "\n\n".join(lines)
