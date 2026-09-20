"""
schemas/qc.py
=============
QCReport — output of the QC Agent (adversarial critic).

Design decisions (v4):
  - ErrorClassification enum provides a structured taxonomy for GAMP 5 validation
    test records and CIOMS WG XIV Principle 3 (Validity and Robustness) compliance.
  - CritiqueItem carries:
      * error_classification (ErrorClassification enum)
      * affected_node_id: the MAGMA provenance ID of the entity in question
        (enables exact source citation: <ref:DRG-XXXX> or <ref:AE-YYYY>)
      * field_path: the E2B(R3) field code or schema path being critiqued
      * severity: BLOCKER | WARNING | INFO
  - QCReport.approved means ALL critiqueItems are severity=INFO or WARNING;
    any BLOCKER → approved=False → routes to HITL.
  - hard_negative_violations: drugs blocked by clinical_negatives.json ontology.
    Populated by QCAgent; causes BLOCKER CritiqueItem automatically.

Compliance:
  - 21 CFR Part 11: frozen schema, prompt_version tracked
  - GAMP 5: ErrorClassification enables defect classification log
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ErrorClassification(str, Enum):
    """
    GAMP 5 / CIOMS WG XIV structured error taxonomy.
    Every CritiqueItem must be assigned exactly one classification.

    HALLUCINATION:           Entity invented by LLM with no basis in the narrative
    POLARITY_FLIP:           Boolean or categorical inversion (YES→NO, RELATED→NOT_RELATED)
    OMISSION:                Entity present in narrative but absent from extraction
    DATE_ORDER_VIOLATION:    Temporal ordering violated (drug start > event onset)
    CAUSALITY_UNSUPPORTED:   Causality claim contradicted by hard_negative ontology
    SCHEMA_VIOLATION:        Pydantic validation failure (type error, range error, etc.)
    CONFIDENCE_INSUFFICIENT: Output confidence below the RiskTier effective threshold
    CONSISTENCY_ERROR:       Internal inconsistency between two fields (not date-related)
    CODING_MISMATCH:         Coded term does not semantically match verbatim term
    LISTEDNESS_UNCERTAIN:    RSI lookup was inconclusive; manual review required
    """
    HALLUCINATION            = "HALLUCINATION"
    POLARITY_FLIP            = "POLARITY_FLIP"
    OMISSION                 = "OMISSION"
    DATE_ORDER_VIOLATION     = "DATE_ORDER_VIOLATION"
    CAUSALITY_UNSUPPORTED    = "CAUSALITY_UNSUPPORTED"
    SCHEMA_VIOLATION         = "SCHEMA_VIOLATION"
    CONFIDENCE_INSUFFICIENT  = "CONFIDENCE_INSUFFICIENT"
    CONSISTENCY_ERROR        = "CONSISTENCY_ERROR"
    CODING_MISMATCH          = "CODING_MISMATCH"
    LISTEDNESS_UNCERTAIN     = "LISTEDNESS_UNCERTAIN"


class CritiqueSeverity(str, Enum):
    BLOCKER = "BLOCKER"   # Causes approved=False; routes to HITL
    WARNING = "WARNING"   # Logged; does not block auto-approval
    INFO    = "INFO"      # Informational only


class CritiqueItem(BaseModel):
    """Single QC finding."""
    error_classification: ErrorClassification
    severity:             CritiqueSeverity
    affected_node_id:     Optional[str]  = Field(
        None,
        description=(
            "MAGMA provenance ID of the affected entity "
            "(SuspectDrug.node_id or VerbatimEvent.node_id). "
            "Enables <ref:node_id> citation in E2B narrative QC reports."
        )
    )
    field_path:           Optional[str]  = Field(
        None,
        description="E2B(R3) field code or Python schema path (e.g. 'G.k.8' or 'suspect_drugs[0].dechallenge')"
    )
    message:              str            = Field(..., max_length=500)
    suggested_correction: Optional[str] = Field(None, max_length=500)


class HardNegativeViolation(BaseModel):
    """
    Record of a drug-AE pair blocked by the clinical_negatives.json ontology.
    Automatically creates a BLOCKER CritiqueItem in QCReport.
    """
    drug_node_id:   str
    drug_name:      str
    event_node_id:  str
    event_term:     str
    edge_type:      str   = Field(
        ...,
        description=(
            "Edge type from clinical_negatives.json "
            "(e.g. DOES_NOT_TREAT, CONTRAINDICATED_WITH)"
        )
    )
    ontology_source: Optional[str] = Field(
        None,
        description="Source entry in clinical_negatives.json (drug class or drug name)"
    )


class QCReport(BaseModel):
    """
    Full QC report for a case.
    Immutable after creation.

    approved=True  → all critiqueItems are WARNING or INFO → continue pipeline
    approved=False → at least one BLOCKER found → route to HITL
    """
    model_config = {"frozen": True}

    case_id:                   str
    approved:                  bool
    critique_items:            list[CritiqueItem]         = Field(default_factory=list)
    hard_negative_violations:  list[HardNegativeViolation] = Field(default_factory=list)
    qc_confidence:             float                      = Field(..., ge=0.0, le=1.0)
    agent_id:                  str                        = "qc-agent-v1"
    prompt_version:            str                        = "qc-prompt-v1.0"

    @model_validator(mode="after")
    def approval_consistent_with_blockers(self) -> "QCReport":
        has_blocker = any(
            c.severity == CritiqueSeverity.BLOCKER for c in self.critique_items
        )
        # Hard negative violations always count as blockers
        if self.hard_negative_violations:
            has_blocker = True

        if self.approved and has_blocker:
            raise ValueError(
                "approved=True but BLOCKER critique items or hard_negative_violations exist. "
                "A BLOCKER must result in approved=False."
            )
        if not self.approved and not has_blocker:
            # approved=False requires a BLOCKER to be documented.
            # WARNING / INFO items are advisory only — they do not block approval.
            raise ValueError(
                "approved=False requires at least one BLOCKER CritiqueItem or a "
                "hard_negative_violation. WARNING/INFO items are advisory and do not "
                "block approval on their own. A rejection must document a BLOCKER."
            )
        return self
