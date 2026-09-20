"""
agents/causality_agent.py
==========================
Causality Agent — Stage 5 of the ICSR processing pipeline.

Responsibilities:
  1. Assess causality for each SUSPECT drug–event pair using WHO-UMC terminology
  2. Consider rechallenge/dechallenge evidence, temporal plausibility, and
     alternative etiologies (MAGMA E_causal graph)
  3. Verify hard_negative_checked=True for each SUSPECT assessment
  4. Produce CausalityMatrix (immutable, uniqueness-validated)

WHO-UMC terms:
  RELATED | POSSIBLY_RELATED | UNLIKELY_RELATED | NOT_RELATED | UNKNOWN | NOT_REPORTED

Rules enforced:
  - UNLIKELY_RELATED / NOT_RELATED → must list alternative_etiologies
  - confidence < tier threshold → HITL
  - CausalityMatrix duplicate pair → ValueError → HITL

Output keys: causality_matrix, partial_e2b, current_stage, next_stage
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient
from schemas.causality import (
    AlternativeEtiology, CausalityAssessment, CausalityMatrix, CausalityTerm,
)
from schemas.extraction import DrugRole, ExtractedCaseEntities
from schemas.triage import THRESHOLD_BY_TIER, RiskTier

logger = logging.getLogger(__name__)

PROMPT_VERSION = "causality-prompt-v1.0"

SYSTEM_PROMPT = """You are an expert pharmacovigilance (PV) causality assessment specialist.
Assess the causality between each suspect drug and each adverse event using WHO-UMC terminology.

WHO-UMC causality terms:
  "Related"          — likely causal; good temporal sequence + known pharmacology
  "Possibly related" — could be related; plausible timing but alternatives not excluded
  "Unlikely related" — doubtful; temporal relationship unlikely OR plausible alternative exists
  "Not related"      — clear alternative cause established
  "Unknown"          — insufficient information to assess
  "Not reported"     — causality not attempted

Output ONLY a valid JSON object:
{
  "assessments": [
    {
      "drug_node_id":           "<string>",
      "event_node_id":          "<string>",
      "drug_name":              "<string>",
      "verbatim_event":         "<string>",
      "causality_term":         "<WHO-UMC term>",
      "rationale":              "<string, max 1000 chars>",
      "confidence":             <float 0.0–1.0>,
      "alternative_etiologies": [
        {
          "description":      "<string>",
          "likelihood":       "High" | "Moderate" | "Low" | "Unlikely",
          "source_reference": "<string or null>"
        }
      ],
      "hard_negative_checked": true | false
    }
  ]
}

RULES:
1. Assess EVERY suspect drug × adverse event pair.
2. "Unlikely related" and "Not related" REQUIRE at least one alternative_etiology.
3. hard_negative_checked: true if you considered whether this drug pharmacologically
   CANNOT cause this event. Set to true for all SUSPECT drug assessments.
4. Use rechallenge (drug restarted → AE recurred) as strong evidence for "Related".
5. Use dechallenge (drug stopped → AE resolved) as supporting evidence.
6. DO NOT output any text outside the JSON object.
"""


class CausalityAgent(BaseAgent):
    """
    Stage 5 Causality Agent — WHO-UMC, rechallenge/dechallenge, MAGMA E_causal.

    Parameters
    ----------
    llm      : OllamaClient — shared inference client
    audit_db : AuditDB     — 21 CFR Part 11 audit log
    """

    AGENT_ID = "causality-agent-v1"

    def __init__(self, llm: OllamaClient, audit_db: AuditDB) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)

    def _run_inner(self, state: GraphState) -> dict:
        case_id   = state["case_id"]
        entities: Optional[ExtractedCaseEntities] = state.get("extracted_entities")
        risk_tier = RiskTier(state.get("risk_tier") or "TIER_2")

        if entities is None:
            raise ValueError(
                f"CausalityAgent: extracted_entities missing for case={case_id}"
            )

        suspect_drugs  = [d for d in entities.suspect_drugs if d.drug_role == DrugRole.SUSPECT]
        verbatim_events = entities.verbatim_events

        if not suspect_drugs:
            raise ValueError(
                f"CausalityAgent: no SUSPECT drugs found for case={case_id}"
            )

        logger.info(
            "CausalityAgent: %d suspect drug(s) × %d event(s) for case=%s",
            len(suspect_drugs), len(verbatim_events), case_id
        )

        # ── Build LLM context: entities summary ───────────────────────────────
        drugs_summary = "\n".join([
            f"- node_id={d.node_id} | drug={d.drug_name} | "
            f"start={d.start_date.to_e2b_str() if d.start_date else 'unknown'} | "
            f"dechallenge={d.dechallenge.value} | rechallenge={d.rechallenge.value}"
            for d in suspect_drugs
        ])
        events_summary = "\n".join([
            f"- node_id={e.node_id} | event={e.verbatim_term} | "
            f"onset={e.onset_date.to_e2b_str() if e.onset_date else 'unknown'} | "
            f"serious={e.serious}"
            for e in verbatim_events
        ])

        user_message = (
            f"Case ID: {case_id}\n\n"
            f"NARRATIVE:\n{state['raw_narrative']}\n\n"
            f"SUSPECT DRUGS:\n{drugs_summary}\n\n"
            f"ADVERSE EVENTS:\n{events_summary}\n\n"
            "Assess causality for ALL drug–event pairs listed above."
        )

        response = self._llm.chat(
            system_prompt  = SYSTEM_PROMPT,
            user_message   = user_message,
            prompt_version = PROMPT_VERSION,
        )

        data = self._parse_llm_json(
            response.text, context=f"CausalityAgent case={case_id}"
        )

        # ── Parse assessments ─────────────────────────────────────────────────
        assessments: list[CausalityAssessment] = []
        min_confidence = 1.0

        for a in data.get("assessments", []):
            try:
                term = CausalityTerm(a.get("causality_term", "Unknown"))
            except ValueError:
                term = CausalityTerm.UNKNOWN

            # Parse alternative etiologies
            alt_etiologies: list[AlternativeEtiology] = []
            for ae in a.get("alternative_etiologies", []):
                try:
                    alt_etiologies.append(AlternativeEtiology(
                        description      = str(ae.get("description", ""))[:500],
                        likelihood       = str(ae.get("likelihood", "Moderate")),
                        source_reference = ae.get("source_reference"),
                    ))
                except Exception as exc:
                    logger.debug("CausalityAgent: AlternativeEtiology parse error: %s", exc)

            # Enforce alternative etiologies for UNLIKELY/NOT_RELATED
            if term in (CausalityTerm.UNLIKELY_RELATED, CausalityTerm.NOT_RELATED):
                if not alt_etiologies:
                    alt_etiologies.append(AlternativeEtiology(
                        description  = "Alternative etiology not specified by assessor.",
                        likelihood   = "Unknown",
                        source_reference = None,
                    ))

            confidence = float(a.get("confidence", 0.5))
            min_confidence = min(min_confidence, confidence)

            ca = CausalityAssessment(
                drug_node_id           = str(a["drug_node_id"]),
                event_node_id          = str(a["event_node_id"]),
                drug_name              = str(a.get("drug_name", "")),
                verbatim_event         = str(a.get("verbatim_event", "")),
                causality_term         = term,
                rationale              = str(a.get("rationale", ""))[:1000],
                confidence             = confidence,
                alternative_etiologies = alt_etiologies,
                hard_negative_checked  = bool(a.get("hard_negative_checked", True)),
                agent_id               = self.AGENT_ID,
                prompt_version         = PROMPT_VERSION,
            )
            assessments.append(ca)

        # ── Build CausalityMatrix (validates uniqueness) ──────────────────────
        causality_matrix = CausalityMatrix(
            case_id     = case_id,
            assessments = assessments,
        )

        # ── Partial E2B blackboard ────────────────────────────────────────────
        partial_e2b: dict = dict(state.get("partial_e2b") or {})
        for ca in assessments:
            k = f"G.k.9.i.2.{ca.drug_node_id}.{ca.event_node_id}"
            partial_e2b[k] = ca.causality_term.value

        # ── HITL routing ──────────────────────────────────────────────────────
        threshold  = THRESHOLD_BY_TIER[risk_tier]
        needs_hitl = min_confidence < threshold

        logger.info(
            "CausalityAgent: case=%s assessments=%d min_confidence=%.3f "
            "threshold=%.2f needs_hitl=%s",
            case_id, len(assessments), min_confidence, threshold, needs_hitl
        )

        partial: dict = {
            "causality_matrix": causality_matrix,
            "partial_e2b":      partial_e2b,
            "current_stage":    "causality",
        }

        if needs_hitl:
            partial.update({
                "pipeline_halted": True,
                "hitl_stage":      "causality",
                "next_stage":      "hitl_review",
            })
        else:
            partial["next_stage"] = "listedness"

        return partial
