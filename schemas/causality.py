"""
schemas/causality.py
====================
CausalityAssessment and CausalityMatrix — output of the Causality Agent.

Design decisions (v4):
  - CausalityTerm uses WHO-UMC vocabulary (RELATED / POSSIBLY_RELATED /
    UNLIKELY_RELATED / NOT_RELATED / UNKNOWN / NOT_REPORTED) — approved for
    spontaneous ICSR reporting. Naranjo algorithm removed per user decision.
  - CausalityAssessment now carries:
      * alternative_etiologies: list of alternative causes considered
        (required for "Not related" and "Unlikely" assessments — MAGMA E_causal)
      * confidence: float
      * rationale: structured free-text (up to 1000 chars)
      * hard_negative_checked: bool — True if clinical_negatives.json was consulted
  - CausalityMatrix wraps all assessments for a case and validates uniqueness
    of (drug_node_id, event_node_id) pairs.
  - Both schemas are frozen (21 CFR Part 11 immutability).

Compliance:
  - WHO-UMC Causality Terminology
  - ICH E2B(R3) G.k.9.i (causality assessment field)
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class CausalityTerm(str, Enum):
    """
    WHO-UMC Causality Assessment Terminology.

    RELATED          — likely causal; temporal sequence + known pharmacology
    POSSIBLY_RELATED — could be related; temporal plausible + alternative explanations not excluded
    UNLIKELY_RELATED — doubtful; temporal relationship unlikely / plausible alternative exists
    NOT_RELATED      — unrelated; clear alternative cause established
    UNKNOWN          — insufficient information to assess
    NOT_REPORTED     — causality assessment not attempted by reporter
    Maps to ICH E2B(R3) G.k.9.i.
    """
    RELATED          = "Related"
    POSSIBLY_RELATED = "Possibly related"
    UNLIKELY_RELATED = "Unlikely related"
    NOT_RELATED      = "Not related"
    UNKNOWN          = "Unknown"
    NOT_REPORTED     = "Not reported"


class AlternativeEtiology(BaseModel):
    """
    A single alternative cause considered during causality assessment.
    Maps to the MAGMA E_causal graph: Pre-existing_Condition → AE node edges.
    """
    description:      str    = Field(..., min_length=1, max_length=500)
    likelihood:       str    = Field(..., description="e.g. 'High', 'Moderate', 'Low', 'Unlikely'")
    source_reference: Optional[str] = Field(
        None,
        description="Relevant narrative sentence or drug label section supporting this etiology"
    )


class CausalityAssessment(BaseModel):
    """
    Single (drug, event) causality assessment pair.
    Immutable after creation.

    alternative_etiologies MUST be non-empty when term is UNLIKELY_RELATED or NOT_RELATED.
    hard_negative_checked   MUST be True for all SUSPECT drugs.
    """
    model_config = {"frozen": True}

    drug_node_id:          str              = Field(..., description="SuspectDrug.node_id")
    event_node_id:         str              = Field(..., description="VerbatimEvent.node_id")
    drug_name:             str
    verbatim_event:        str
    causality_term:        CausalityTerm
    rationale:             str              = Field(..., max_length=1000)
    confidence:            float            = Field(..., ge=0.0, le=1.0)
    alternative_etiologies: list[AlternativeEtiology] = Field(
        default_factory=list,
        description=(
            "Alternative causes considered. "
            "MUST be non-empty when causality_term is UNLIKELY_RELATED or NOT_RELATED."
        )
    )
    hard_negative_checked: bool             = Field(
        default=False,
        description=(
            "True if clinical_negatives.json was consulted for this drug-event pair. "
            "Must be True for all SUSPECT drug assessments."
        )
    )
    agent_id:              str              = "causality-agent-v1"
    prompt_version:        str              = "causality-prompt-v1.0"

    @model_validator(mode="after")
    def etiologies_required_for_non_related(self) -> "CausalityAssessment":
        if self.causality_term in (CausalityTerm.UNLIKELY_RELATED, CausalityTerm.NOT_RELATED):
            if not self.alternative_etiologies:
                raise ValueError(
                    f"causality_term={self.causality_term} requires at least one "
                    "alternative_etiology to document the competing explanation."
                )
        return self


class CausalityMatrix(BaseModel):
    """
    All causality assessments for a case.
    Validates uniqueness of (drug_node_id, event_node_id) pairs.
    Immutable after creation.
    """
    model_config = {"frozen": True}

    case_id:     str
    assessments: list[CausalityAssessment] = Field(..., min_length=1)

    @model_validator(mode="after")
    def unique_drug_event_pairs(self) -> "CausalityMatrix":
        seen: set[tuple[str, str]] = set()
        for a in self.assessments:
            key = (a.drug_node_id, a.event_node_id)
            if key in seen:
                raise ValueError(
                    f"Duplicate (drug_node_id, event_node_id) pair: "
                    f"({a.drug_node_id}, {a.event_node_id}). "
                    "Each drug-event combination must have exactly one assessment."
                )
            seen.add(key)
        return self
