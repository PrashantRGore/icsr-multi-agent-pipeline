"""
agents/qc_agent.py
==================
QC Agent — Stage 3 of the ICSR processing pipeline.

The QC Agent is an adversarial critic: it re-reads the narrative and the
extracted entities, identifies errors, and produces a QCReport.

Responsibilities:
  1. Validate extraction completeness against the source narrative
  2. Check hard-negative ontology (clinical_negatives.json) for each
     suspect drug–event pair to block pharmacologically implausible causality
  3. Classify errors using ErrorClassification taxonomy (10 types)
  4. Set approved=True/False; BLOCKER → pipeline_halted + HITL

Hard-negative ontology:
  - DOES_NOT_TREAT edges: drug class pharmacologically cannot cause the AE
    (e.g. beta-blockers do not cause tachycardia as a direct effect)
  - Loaded from data/clinical_negatives.json at agent construction time

Output keys: qc_report, current_stage, next_stage
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from graph.state import GraphState
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient
from schemas.extraction import DrugRole, ExtractedCaseEntities
from schemas.qc import (
    CritiqueItem, CritiqueSeverity, ErrorClassification,
    HardNegativeViolation, QCReport,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "qc-prompt-v1.0"

_CLINICAL_NEGATIVES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "clinical_negatives.json"
)

SYSTEM_PROMPT = """You are an expert pharmacovigilance (PV) quality control specialist
and adversarial critic. Your job is to find errors in extracted ICSR data by comparing
it against the source narrative.

You will receive:
  - The original narrative text
  - The extracted entities (JSON)

Output ONLY a valid JSON object:
{
  "approved": true | false,
  "critique_items": [
    {
      "error_classification": "<one of: HALLUCINATION | POLARITY_FLIP | OMISSION | DATE_ORDER_VIOLATION | CAUSALITY_UNSUPPORTED | SCHEMA_VIOLATION | CONFIDENCE_INSUFFICIENT | CONSISTENCY_ERROR | CODING_MISMATCH | LISTEDNESS_UNCERTAIN>",
      "severity": "BLOCKER" | "WARNING" | "INFO",
      "affected_node_id": "<drug or event node_id or null>",
      "field_path": "<E2B field code or schema path or null>",
      "message": "<max 500 chars>",
      "suggested_correction": "<max 500 chars or null>"
    }
  ],
  "qc_confidence": <float 0.0–1.0>
}

RULES:
1. approved=true only when there are NO BLOCKER items.
2. approved=false requires at least one BLOCKER item.
3. BLOCKER = factual error, invented entity, impossible date, or critical omission.
4. WARNING = possible issue or ambiguity that does not prevent submission.
5. INFO = observation that adds context without blocking.
6. Search for: hallucinated drugs/AEs not in narrative, wrong dates, missing entities,
   impossible sequences (drug started after AE), wrong seriousness flag, misidentified
   drug role (SUSPECT vs CONCOMITANT).
7. If extraction is perfect, output approved=true and empty critique_items=[].
8. DO NOT output any text outside the JSON object.
"""


class QCAgent(BaseAgent):
    """
    Stage 3 QC Agent (adversarial critic + hard-negative ontology checker).

    Parameters
    ----------
    llm          : OllamaClient — shared inference client
    audit_db     : AuditDB     — 21 CFR Part 11 audit log
    neg_path     : Path        — path to clinical_negatives.json
    """

    AGENT_ID = "qc-agent-v1"

    def __init__(
        self,
        llm:      OllamaClient,
        audit_db: AuditDB,
        neg_path: Path = _CLINICAL_NEGATIVES_PATH,
    ) -> None:
        super().__init__(llm=llm, audit_db=audit_db, agent_id=self.AGENT_ID)
        self._neg_path     = neg_path
        self._negatives    = self._load_negatives()

    def _load_negatives(self) -> list[dict]:
        """Load clinical_negatives.json. Return empty list if missing."""
        try:
            with open(self._neg_path, encoding="utf-8") as f:
                data = json.load(f)
            # Support both {"rules": [...]} and [...]
            if isinstance(data, dict):
                return data.get("rules", data.get("negatives", []))
            return data
        except FileNotFoundError:
            logger.warning("QCAgent: clinical_negatives.json not found at %s", self._neg_path)
            return []
        except Exception as exc:
            logger.error("QCAgent: failed to load clinical_negatives.json: %s", exc)
            return []

    def _run_inner(self, state: GraphState) -> dict:
        case_id   = state["case_id"]
        narrative = state["raw_narrative"]
        entities  = state.get("extracted_entities")

        if entities is None:
            raise ValueError(
                f"QCAgent: extracted_entities is None for case={case_id}. "
                "ExtractionAgent must run before QCAgent."
            )

        logger.info("QCAgent: processing case=%s", case_id)

        entities_json = entities.model_dump_json(indent=2)

        # ── LLM adversarial critique ──────────────────────────────────────────
        user_message = (
            f"Case ID: {case_id}\n\n"
            f"ORIGINAL NARRATIVE:\n{narrative}\n\n"
            f"EXTRACTED ENTITIES (JSON):\n{entities_json}\n\n"
            "Review the extraction for errors. Be strict — flag ALL discrepancies."
        )

        response = self._llm.chat(
            system_prompt  = SYSTEM_PROMPT,
            user_message   = user_message,
            prompt_version = PROMPT_VERSION,
        )

        data = self._parse_llm_json(response.text, context=f"QCAgent case={case_id}")

        # ── Parse critique items ──────────────────────────────────────────────
        critique_items: list[CritiqueItem] = []
        for item in data.get("critique_items", []):
            try:
                ec  = ErrorClassification(item.get("error_classification", "CONSISTENCY_ERROR"))
                sev = CritiqueSeverity(item.get("severity", "WARNING"))
            except ValueError:
                ec  = ErrorClassification.CONSISTENCY_ERROR
                sev = CritiqueSeverity.WARNING

            critique_items.append(CritiqueItem(
                error_classification = ec,
                severity             = sev,
                affected_node_id     = item.get("affected_node_id"),
                field_path           = item.get("field_path"),
                message              = str(item.get("message", ""))[:500],
                suggested_correction = item.get("suggested_correction"),
            ))

        # ── Hard-negative ontology check ─────────────────────────────────────
        hard_neg_violations = self._check_hard_negatives(entities)

        for hnv in hard_neg_violations:
            critique_items.append(CritiqueItem(
                error_classification = ErrorClassification.CAUSALITY_UNSUPPORTED,
                severity             = CritiqueSeverity.BLOCKER,
                affected_node_id     = hnv.drug_node_id,
                field_path           = "G.k.9.i (causality)",
                message              = (
                    f"Hard-negative ontology block: '{hnv.drug_name}' "
                    f"DOES_NOT_TREAT/DOES_NOT_CAUSE '{hnv.event_term}'. "
                    f"Source: {hnv.ontology_source}"
                )[:500],
                suggested_correction = "Review causality assessment; mark as NOT_RELATED.",
            ))

        # ── Determine approval ────────────────────────────────────────────────
        has_blocker = (
            any(c.severity == CritiqueSeverity.BLOCKER for c in critique_items)
            or bool(hard_neg_violations)
        )
        qc_confidence = float(data.get("qc_confidence", 0.9))
        approved      = not has_blocker

        qc_report = QCReport(
            case_id                  = case_id,
            approved                 = approved,
            critique_items           = critique_items,
            hard_negative_violations = hard_neg_violations,
            qc_confidence            = qc_confidence,
            agent_id                 = self.AGENT_ID,
            prompt_version           = PROMPT_VERSION,
        )

        logger.info(
            "QCAgent: case=%s approved=%s blockers=%d warnings=%d hard_neg=%d",
            case_id, approved,
            sum(1 for c in critique_items if c.severity == CritiqueSeverity.BLOCKER),
            sum(1 for c in critique_items if c.severity == CritiqueSeverity.WARNING),
            len(hard_neg_violations),
        )

        partial: dict = {
            "qc_report":     qc_report,
            "current_stage": "qc",
        }

        if not approved:
            partial.update({
                "pipeline_halted": True,
                "hitl_stage":      "qc",
                "next_stage":      "hitl_review",
            })
        else:
            partial["next_stage"] = "coding"

        return partial

    # ── Hard-negative ontology helpers ────────────────────────────────────────

    def _check_hard_negatives(
        self, entities: ExtractedCaseEntities
    ) -> list[HardNegativeViolation]:
        """
        Check all SUSPECT drug–event pairs against clinical_negatives.json.
        Returns a list of violations (DOES_NOT_TREAT edges).
        INCREASES_RISK_OF edges are NOT violations (valid causal direction).
        """
        violations: list[HardNegativeViolation] = []

        suspect_drugs = [d for d in entities.suspect_drugs if d.drug_role == DrugRole.SUSPECT]

        for drug in suspect_drugs:
            drug_name_lower = drug.drug_name.lower()
            drug_class      = (drug.drug_class or "").lower()

            for event in entities.verbatim_events:
                event_term_lower = event.verbatim_term.lower()

                for rule in self._negatives:
                    if rule.get("edge_type") != "DOES_NOT_TREAT":
                        continue  # Only block DOES_NOT_TREAT

                    subject = str(rule.get("subject", "")).lower()
                    objects = [str(o).lower() for o in rule.get("object", [])]

                    # Match drug by name or class
                    drug_match = (
                        subject in drug_name_lower
                        or drug_name_lower in subject
                        or (drug_class and subject in drug_class)
                    )
                    if not drug_match:
                        continue

                    # Match event term
                    for obj in objects:
                        if obj in event_term_lower or event_term_lower in obj:
                            violations.append(HardNegativeViolation(
                                drug_node_id   = drug.node_id,
                                drug_name      = drug.drug_name,
                                event_node_id  = event.node_id,
                                event_term     = event.verbatim_term,
                                edge_type      = "DOES_NOT_TREAT",
                                ontology_source= rule.get("source", rule.get("subject", "")),
                            ))
                            break

        return violations
