"""
agents/extraction_agent.py
===========================
Extraction Agent — Stage 2 of the ICSR processing pipeline.

Responsibilities:
  1. Extract suspect drugs, adverse events, patient demographics from the narrative
  2. Enrich each drug with RxNorm RxCUI (via RxNormClient)
  3. Build the MAGMA case graph (temporal + causal edges)
  4. Run temporal QC assertions (R1: drug start ≤ AE onset)
  5. Return ExtractedCaseEntities and CaseGraph payload

Output keys in GraphState:
  extracted_entities, graph_payload, partial_e2b, current_stage, next_stage

Prompt architecture:
  - System: role + JSON output schema (drugs array + events array + demographics)
  - User:   narrative + case_id
  - Output: JSON with "suspect_drugs" + "verbatim_events" + demographics

HITL routing:
  - extraction_confidence < tier threshold → pipeline_halted + hitl_stage=extraction
  - Pydantic validation failure (e.g. drug start > AE onset) → same
  - R1 temporal assertion failure → WARNING only (not HITL) — logged
"""
from __future__ import annotations

import json
import logging
from typing import Any

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.case_graph import CaseGraph
from infra.ollama_client import OllamaClient
from infra.rxnorm_client import RxNormClient
from schemas.extraction import (
    Dechallenge, DrugRole, ExtractedCaseEntities,
    HospitalizationCausalityFlag, HospitalizationDetails,
    PartialDate, Rechallenge, SeriousnessCriterion, SuspectDrug, VerbatimEvent,
)
from schemas.triage import THRESHOLD_BY_TIER, RiskTier

logger = logging.getLogger(__name__)

PROMPT_VERSION = "extraction-prompt-v1.0"

SYSTEM_PROMPT = """You are an expert pharmacovigilance (PV) data extraction specialist.
Extract structured information from the ICSR narrative and output ONLY a valid JSON object.

Output schema:
{
  "suspect_drugs": [
    {
      "drug_name":   "<string>",
      "dose":        "<string or null>",
      "route":       "<string or null>",
      "indication":  "<string or null>",
      "drug_role":   "SUSPECT" | "CONCOMITANT" | "INTERACTING",
      "start_date":  {"year": <int>, "month": <int or null>, "day": <int or null>} | null,
      "stop_date":   {"year": <int>, "month": <int or null>, "day": <int or null>} | null,
      "dechallenge": "YES" | "NO" | "UNKNOWN" | "N/A",
      "rechallenge": "YES" | "NO" | "NOT_REPORTED" | "UNKNOWN"
    }
  ],
  "verbatim_events": [
    {
      "verbatim_term":       "<string>",
      "reporter_term":       "<string or null>",
      "onset_date":          {"year": <int>, "month": <int or null>, "day": <int or null>} | null,
      "outcome":             "<string or null>",
      "serious":             true | false,
      "seriousness_criteria": ["Death" | "Life-threatening" | "Hospitalization" |
                               "Disability / Incapacity" | "Congenital anomaly" |
                               "Intervention required" | "Other medically important condition"],
      "hospitalization_details": {
        "hospitalization_flag": "Event caused hospitalization" | "Event did not cause hospitalization",
        "date_of_admission":    {"year": <int>, "month": <int or null>, "day": <int or null>} | null,
        "date_of_discharge":    {"year": <int>, "month": <int or null>, "day": <int or null>} | null
      } | null
    }
  ],
  "patient_age":           "<string or null>",
  "patient_sex":           "<string or null>",
  "patient_ethnicity":     "<string or null>",
  "patient_weight_kg":     <float or null>,
  "patient_height_cm":     <float or null>,
  "reporter_type":         "<string or null>",
  "country_of_occurrence": "<string or null>",
  "extraction_confidence": <float 0.0–1.0>
}

CRITICAL RULES:
1. drug_role=SUSPECT for the primary causative drug.
2. dechallenge=N/A when drug was NOT stopped.
3. rechallenge=NOT_REPORTED when rechallenge is not mentioned at all.
4. serious=true REQUIRES at least one seriousness_criterion.
5. serious=false → seriousness_criteria MUST be empty [].
6. hospitalization_details REQUIRED when "Hospitalization" is in seriousness_criteria.
7. Partial dates: provide year always; month/day only if stated.
8. DO NOT output any text outside the JSON object.
"""


def _parse_partial_date(d: Any) -> PartialDate | None:
    """Safely convert a dict to PartialDate; return None on failure."""
    if not d or not isinstance(d, dict):
        return None
    try:
        return PartialDate(
            year=int(d["year"]),
            month=int(d["month"]) if d.get("month") else None,
            day=int(d["day"])     if d.get("day")   else None,
        )
    except Exception as exc:
        logger.debug("PartialDate parse failed: %s — raw=%s", exc, d)
        return None


def _parse_hosp_details(d: Any) -> HospitalizationDetails | None:
    """Parse hospitalization_details dict."""
    if not d or not isinstance(d, dict):
        return None
    flag_raw = d.get("hospitalization_flag", "")
    try:
        flag = HospitalizationCausalityFlag(flag_raw)
    except ValueError:
        flag = HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION
    return HospitalizationDetails(
        hospitalization_flag = flag,
        date_of_admission    = _parse_partial_date(d.get("date_of_admission")),
        date_of_discharge    = _parse_partial_date(d.get("date_of_discharge")),
    )


class ExtractionAgent(BaseAgent):
    """
    Stage 2 Extraction Agent.

    Parameters
    ----------
    llm       : OllamaClient  — shared inference client
    audit_db  : AuditDB       — 21 CFR Part 11 audit log
    rxnorm    : RxNormClient  — drug normalisation service
    """

    AGENT_ID = "extraction-agent-v1"

    def __init__(
        self,
        llm:      OllamaClient,
        audit_db: AuditDB,
        rxnorm:   RxNormClient,
    ) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)
        self._rxnorm = rxnorm

    def _run_inner(self, state: GraphState) -> dict:
        case_id        = state["case_id"]
        narrative      = state["raw_narrative"]
        narrative_hash = state["narrative_hash"]
        risk_tier_str  = state.get("risk_tier") or "TIER_2"
        risk_tier      = RiskTier(risk_tier_str)

        logger.info("ExtractionAgent: processing case=%s", case_id)

        # ── LLM call ─────────────────────────────────────────────────────────
        user_message = (
            f"Case ID: {case_id}\n\n"
            f"Narrative:\n{narrative}\n\n"
            "Extract all drug and adverse event entities."
        )

        response = self._llm.chat(
            system_prompt  = SYSTEM_PROMPT,
            user_message   = user_message,
            prompt_version = PROMPT_VERSION,
        )

        data = self._parse_llm_json(response.text, context=f"ExtractionAgent case={case_id}")

        # ── Build SuspectDrug list ────────────────────────────────────────────
        suspect_drugs: list[SuspectDrug] = []
        drug_names    : list[str]         = []

        for d in data.get("suspect_drugs", []):
            try:
                role = DrugRole(d.get("drug_role", "SUSPECT"))
            except ValueError:
                role = DrugRole.SUSPECT

            try:
                dechallenge = Dechallenge(d.get("dechallenge", "UNKNOWN"))
            except ValueError:
                dechallenge = Dechallenge.UNKNOWN

            try:
                rechallenge = Rechallenge(d.get("rechallenge", "NOT_REPORTED"))
            except ValueError:
                rechallenge = Rechallenge.NOT_REPORTED

            sd = SuspectDrug(
                drug_name   = str(d["drug_name"]),
                dose        = d.get("dose"),
                route       = d.get("route"),
                indication  = d.get("indication"),
                drug_role   = role,
                start_date  = _parse_partial_date(d.get("start_date")),
                stop_date   = _parse_partial_date(d.get("stop_date")),
                dechallenge = dechallenge,
                rechallenge = rechallenge,
            )
            suspect_drugs.append(sd)
            drug_names.append(sd.drug_name)

        # ── Build VerbatimEvent list ──────────────────────────────────────────
        verbatim_events: list[VerbatimEvent] = []

        for e in data.get("verbatim_events", []):
            serious = bool(e.get("serious", False))

            criteria: list[SeriousnessCriterion] = []
            for c in e.get("seriousness_criteria", []):
                try:
                    criteria.append(SeriousnessCriterion(c))
                except ValueError:
                    pass   # Unknown criterion — skip

            hosp = None
            if SeriousnessCriterion.HOSPITALIZATION in criteria:
                hosp = _parse_hosp_details(e.get("hospitalization_details"))
                if hosp is None:
                    # Provide a minimal default to satisfy schema
                    hosp = HospitalizationDetails(
                        hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION
                    )

            ve = VerbatimEvent(
                verbatim_term          = str(e["verbatim_term"]),
                reporter_term          = e.get("reporter_term"),
                onset_date             = _parse_partial_date(e.get("onset_date")),
                outcome                = e.get("outcome"),
                serious                = serious,
                seriousness_criteria   = criteria,
                hospitalization_details= hosp,
            )
            verbatim_events.append(ve)

        # ── RxNorm enrichment ────────────────────────────────────────────────
        rx_results = self._rxnorm.batch_lookup(drug_names)
        enriched_drugs: list[SuspectDrug] = []
        for sd in suspect_drugs:
            rx = rx_results.get(sd.drug_name.lower())
            if rx and rx.rxcui:
                # SuspectDrug is frozen — rebuild with RxNorm fields
                enriched_drugs.append(SuspectDrug(
                    **{
                        **sd.model_dump(),
                        "rxcui":        rx.rxcui,
                        "rxnorm_label": rx.label,
                        "drug_class":   rx.drug_class,
                    }
                ))
            else:
                enriched_drugs.append(sd)

        # ── Build ExtractedCaseEntities ───────────────────────────────────────
        confidence = float(data.get("extraction_confidence", 0.5))

        entities = ExtractedCaseEntities(
            case_id               = case_id,
            narrative_hash        = narrative_hash,
            suspect_drugs         = enriched_drugs,
            verbatim_events       = verbatim_events,
            patient_age           = data.get("patient_age"),
            patient_sex           = data.get("patient_sex"),
            patient_ethnicity     = data.get("patient_ethnicity"),
            patient_weight_kg     = data.get("patient_weight_kg"),
            patient_height_cm     = data.get("patient_height_cm"),
            reporter_type         = data.get("reporter_type"),
            country_of_occurrence = data.get("country_of_occurrence"),
            extraction_confidence = confidence,
            agent_id              = self.AGENT_ID,
            prompt_version        = PROMPT_VERSION,
        )

        # ── MAGMA Case Graph ──────────────────────────────────────────────────
        g = CaseGraph.from_entities(entities)

        # Log R1 temporal assertion results (WARNING only, not HITL)
        assertions = g.assert_temporal_order()
        r1_failures = [a for a in assertions if not a.passed]
        if r1_failures:
            logger.warning(
                "ExtractionAgent: temporal assertion failures for case=%s: %s",
                case_id, [a.description for a in r1_failures]
            )

        graph_json = g.serialize_to_json()

        # ── Partial E2B blackboard ────────────────────────────────────────────
        partial_e2b: dict = dict(state.get("partial_e2b") or {})
        partial_e2b.update({
            "D.2.2":   entities.patient_age,
            "D.5":     entities.patient_sex,
            "C.2.1":   entities.reporter_type,
            "C.1.9":   entities.country_of_occurrence,
        })
        for sd in entities.suspect_drugs:
            k = f"G.k.2.1.1.{sd.node_id}"
            partial_e2b[k] = sd.rxcui or sd.drug_name

        # ── HITL routing ──────────────────────────────────────────────────────
        threshold   = THRESHOLD_BY_TIER[risk_tier]
        needs_hitl  = confidence < threshold

        partial: dict = {
            "extracted_entities": entities,
            "graph_payload":      graph_json,
            "partial_e2b":        partial_e2b,
            "current_stage":      "extraction",
        }

        if needs_hitl:
            logger.warning(
                "ExtractionAgent: confidence=%.3f < threshold=%.2f for case=%s → HITL",
                confidence, threshold, case_id
            )
            partial.update({
                "pipeline_halted": True,
                "hitl_stage":      "extraction",
                "next_stage":      "hitl_review",
            })
        else:
            partial["next_stage"] = "qc"

        return partial
