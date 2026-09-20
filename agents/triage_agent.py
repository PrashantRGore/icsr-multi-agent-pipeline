"""
agents/triage_agent.py
=======================
Triage Agent — Stage 1 of the ICSR processing pipeline.

Responsibilities:
  1. Verify the four ICH E2A minimum criteria (patient / reporter / drug / AE)
  2. Assign a RiskTier (TIER_1 / TIER_2 / TIER_3) based on seriousness signals
  3. Set confidence and oversight mode
  4. Return TriageOutput and a partial GraphState dict

LangGraph node contract:
    TriageAgent.run(state) -> dict
    Returns: {"triage_output": TriageOutput, "risk_tier": ..., "oversight_mode": ...}

Prompt architecture:
  - System prompt: role + structured JSON output schema
  - User message:  case_id + raw_narrative
  - Response:      JSON conforming to TriageOutput fields

HITL routing:
  - status=INVALID or confidence < threshold → pipeline_halted=True, hitl_stage=triage
  - Pydantic validation error → same HITL route via BaseAgent._handle_failure
"""
from __future__ import annotations

import logging

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient
from schemas.triage import (
    MinimumCriteria, OversightMode, RiskTier, TriageOutput, TriageStatus,
    THRESHOLD_BY_TIER, _derive_oversight_mode, _derive_risk_tier,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt version (bump on every structural change to the prompt)
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_VERSION = "triage-prompt-v1.0"

SYSTEM_PROMPT = """You are an expert pharmacovigilance (PV) triage specialist.
Your task is to assess an Individual Case Safety Report (ICSR) narrative and
determine whether it constitutes a valid adverse drug reaction (ADR) report.

You MUST output a single, valid JSON object with EXACTLY these fields:

{
  "has_identifiable_patient":  true | false,
  "has_identifiable_reporter": true | false,
  "has_suspect_drug":          true | false,
  "has_adverse_event":         true | false,
  "status":                    "VALID" | "INVALID" | "PENDING_HITL",
  "seriousness_signals":       ["<verbatim keyword from narrative>", ...],
  "confidence":                <float 0.0–1.0>,
  "failure_reasons":           ["<reason>", ...]
}

RULES:
1. has_identifiable_patient:  True if ANY patient identifier exists (age, sex, initials,
   code, anonymised ID). Does NOT require full name.
2. has_identifiable_reporter: True if a reporter is mentioned (physician, nurse, patient,
   pharmacist, company). Does NOT require name.
3. has_suspect_drug:          True if at least one drug name is mentioned in a suspected
   causal role.
4. has_adverse_event:         True if at least one adverse event or symptom is described.
5. status=VALID requires ALL four criteria to be True.
6. status=INVALID requires at least one failure_reason (which criterion is missing).
7. seriousness_signals: extract VERBATIM words indicating seriousness (death, fatal,
   life-threatening, hospitalised, hospitalized, disabled, congenital, intervention required).
   Leave empty [] for non-serious cases.
8. confidence: your probability [0.0, 1.0] that your assessment is correct.
9. failure_reasons: list only when status=INVALID; explain which criterion is missing.
10. DO NOT include any text outside the JSON object.
"""


class TriageAgent(BaseAgent):
    """
    Stage 1 Triage Agent.

    Parameters
    ----------
    llm      : OllamaClient — shared serialized inference client
    audit_db : AuditDB      — 21 CFR Part 11 immutable audit log
    """

    AGENT_ID = "triage-agent-v1"

    def __init__(self, llm: OllamaClient, audit_db: AuditDB) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)

    def _run_inner(self, state: GraphState) -> dict:
        case_id       = state["case_id"]
        narrative     = state["raw_narrative"]
        narrative_hash = state["narrative_hash"]

        logger.info("TriageAgent: processing case=%s", case_id)

        # ── Call LLM ─────────────────────────────────────────────────────────
        user_message = (
            f"Case ID: {case_id}\n\n"
            f"Narrative:\n{narrative}\n\n"
            "Assess the four ICH E2A minimum criteria and classify this case."
        )

        response = self._llm.chat(
            system_prompt  = SYSTEM_PROMPT,
            user_message   = user_message,
            prompt_version = PROMPT_VERSION,
        )

        # ── Parse + validate ─────────────────────────────────────────────────
        data = self._parse_llm_json(response.text, context=f"TriageAgent case={case_id}")

        criteria = MinimumCriteria(
            has_identifiable_patient  = bool(data.get("has_identifiable_patient",  False)),
            has_identifiable_reporter = bool(data.get("has_identifiable_reporter", False)),
            has_suspect_drug          = bool(data.get("has_suspect_drug",          False)),
            has_adverse_event         = bool(data.get("has_adverse_event",         False)),
        )

        seriousness_signals: list[str] = data.get("seriousness_signals", [])
        confidence: float              = float(data.get("confidence", 0.5))
        failure_reasons: list[str]     = data.get("failure_reasons", [])
        status_str: str                = data.get("status", "PENDING_HITL")

        # Derive tier + oversight deterministically (not from LLM)
        risk_tier      = _derive_risk_tier(criteria, seriousness_signals)
        oversight_mode = _derive_oversight_mode(risk_tier)

        # Coerce status: if criteria all met but LLM said INVALID → override
        if criteria.all_met and status_str == "INVALID":
            logger.warning(
                "TriageAgent: criteria all met but LLM returned INVALID — overriding to VALID"
            )
            status_str      = "VALID"
            failure_reasons = []

        # Coerce status: if criteria NOT all met but LLM said VALID → override
        if not criteria.all_met and status_str == "VALID":
            logger.warning(
                "TriageAgent: criteria not met but LLM returned VALID — overriding to INVALID"
            )
            status_str = "INVALID"
            if not failure_reasons:
                missing = [k for k, v in criteria.model_dump().items() if not v]
                failure_reasons = [f"Missing ICH E2A criterion: {m}" for m in missing]

        triage_out = TriageOutput(
            case_id             = case_id,
            narrative_hash      = narrative_hash,
            criteria            = criteria,
            status              = TriageStatus(status_str),
            risk_tier           = risk_tier,
            oversight_mode      = oversight_mode,
            seriousness_signals = seriousness_signals,
            confidence          = confidence,
            failure_reasons     = failure_reasons,
            agent_id            = self.AGENT_ID,
            prompt_version      = PROMPT_VERSION,
            processing_ms       = response.processing_ms,
        )

        # ── HITL routing ──────────────────────────────────────────────────────
        needs_hitl = triage_out.needs_hitl
        threshold  = THRESHOLD_BY_TIER[risk_tier]

        logger.info(
            "TriageAgent: case=%s status=%s tier=%s confidence=%.3f "
            "threshold=%.2f needs_hitl=%s",
            case_id, triage_out.status, risk_tier, confidence, threshold, needs_hitl
        )

        partial: dict = {
            "triage_output":  triage_out,
            "risk_tier":      risk_tier.value,
            "oversight_mode": oversight_mode.value,
            "current_stage":  "triage",
        }

        if needs_hitl:
            partial.update({
                "pipeline_halted": True,
                "hitl_stage":      "triage",
                "next_stage":      "hitl_review",
            })
        else:
            partial["next_stage"] = "extraction"

        return partial
