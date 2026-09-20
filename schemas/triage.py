"""
schemas/triage.py
=================
TriageOutput — output of the Triage Agent.

Design decisions:
  - RiskTier drives confidence thresholds and oversight mode throughout the pipeline.
  - OversightMode is determined at triage and propagates unchanged through GraphState
    unless a downstream agent escalates (e.g., QC detects hallucination on TIER_3 case).
  - status_consistency validator ensures TriageStatus and MinimumCriteria are coherent.
  - TIER_1 cases (Death / Life-threatening) always force HITL regardless of confidence.

Compliance:
  - ICH E2A: four minimum criteria for valid ICSR (patient, reporter, drug, AE)
  - CIOMS WG XIV Principle 1 (Risk-Based Approach): tier-differentiated thresholds
  - CIOMS WG XIV Principle 2 (Human Oversight): OversightMode maps to HITL/HOTL/HIC
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TriageStatus(str, Enum):
    VALID        = "VALID"
    INVALID      = "INVALID"
    PENDING_HITL = "PENDING_HITL"


class RiskTier(str, Enum):
    """
    CIOMS WG XIV risk-based stratification.
    Drives confidence thresholds and OversightMode downstream.

    TIER_1: Death or Life-threatening — threshold 0.95, always HITL.
    TIER_2: Other serious SAE (hospitalization, disability, intervention required,
            congenital anomaly, other medically important condition) — threshold 0.85.
    TIER_3: Non-serious AE — threshold 0.75, HOTL permitted.
    """
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class OversightMode(str, Enum):
    """
    CIOMS WG XIV Table 3 — modalities of human oversight.

    HITL: Human intervenes in every decision cycle (mandatory for TIER_1).
    HOTL: Human monitors and can intervene; auto-submit if all thresholds met (TIER_3).
    HIC:  Human has full override authority; used for regulatory submission gate.
    """
    HITL = "HITL"   # Human-in-the-loop
    HOTL = "HOTL"   # Human-on-the-loop
    HIC  = "HIC"    # Human-in-command


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class MinimumCriteria(BaseModel):
    """
    ICH E2A minimum criteria for a valid ICSR.
    All four must be True for a case to be VALID.
    """
    has_identifiable_patient:  bool
    has_identifiable_reporter: bool
    has_suspect_drug:          bool
    has_adverse_event:         bool

    @property
    def all_met(self) -> bool:
        return all([
            self.has_identifiable_patient,
            self.has_identifiable_reporter,
            self.has_suspect_drug,
            self.has_adverse_event,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# Confidence threshold constants (mirrored from .env defaults)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_BY_TIER: dict[RiskTier, float] = {
    RiskTier.TIER_1: 0.95,
    RiskTier.TIER_2: 0.85,
    RiskTier.TIER_3: 0.75,
}


def _derive_oversight_mode(risk_tier: RiskTier) -> OversightMode:
    """Determine default OversightMode from RiskTier."""
    if risk_tier == RiskTier.TIER_1:
        return OversightMode.HITL
    if risk_tier == RiskTier.TIER_2:
        return OversightMode.HITL
    return OversightMode.HOTL


def _derive_risk_tier(criteria: MinimumCriteria, seriousness_signals: list[str]) -> RiskTier:
    """
    Determine RiskTier from seriousness signals extracted by the triage agent.
    The LLM passes seriousness_signals as a list of strings from the narrative.
    Known TIER_1 signals (case-insensitive): death, fatal, life-threatening, lethal.
    """
    tier1_keywords = {"death", "fatal", "fatality", "life-threatening", "lethal", "died", "mortal"}
    lowered = {s.lower() for s in seriousness_signals}
    if lowered & tier1_keywords:
        return RiskTier.TIER_1
    if seriousness_signals:
        return RiskTier.TIER_2
    return RiskTier.TIER_3


# ─────────────────────────────────────────────────────────────────────────────
# Main schema
# ─────────────────────────────────────────────────────────────────────────────

class TriageOutput(BaseModel):
    """
    Output of the Triage Agent.
    Immutable after creation (frozen=True).
    """
    model_config = {"frozen": True}

    case_id:             str      = Field(..., pattern=r"^ICSR-\d{8}-[A-Z]{3}$",
                                          description="e.g. ICSR-20240315-SYN")
    received_date:       datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    narrative_hash:      str      = Field(..., min_length=64, max_length=64,
                                          description="SHA-256 of the raw narrative text")
    criteria:            MinimumCriteria
    status:              TriageStatus
    risk_tier:           RiskTier
    oversight_mode:      OversightMode
    seriousness_signals: list[str] = Field(
        default_factory=list,
        description="Verbatim seriousness keyword(s) extracted from the narrative by the LLM"
    )
    confidence:          Annotated[float, Field(ge=0.0, le=1.0)]
    failure_reasons:     list[str] = Field(default_factory=list)
    agent_id:            str       = "triage-agent-v1"
    prompt_version:      str       = "triage-prompt-v1.0"
    processing_ms:       Optional[int] = None

    # ── Validators ──────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def status_consistency(self) -> "TriageOutput":
        if self.criteria.all_met and self.status == TriageStatus.INVALID:
            raise ValueError(
                "status=INVALID but all four ICH E2A minimum criteria are satisfied"
            )
        if not self.criteria.all_met and self.status == TriageStatus.VALID:
            raise ValueError(
                "status=VALID but not all four ICH E2A minimum criteria are met. "
                f"Missing: {[k for k, v in self.criteria.model_dump().items() if not v]}"
            )
        return self

    @model_validator(mode="after")
    def tier1_requires_hitl(self) -> "TriageOutput":
        """TIER_1 cases (death/life-threatening) must always use HITL oversight."""
        if self.risk_tier == RiskTier.TIER_1 and self.oversight_mode != OversightMode.HITL:
            raise ValueError(
                "TIER_1 (Death / Life-threatening) cases require OversightMode=HITL. "
                f"Got {self.oversight_mode}"
            )
        return self

    @model_validator(mode="after")
    def failure_reasons_populated_for_invalid(self) -> "TriageOutput":
        if self.status == TriageStatus.INVALID and not self.failure_reasons:
            raise ValueError(
                "status=INVALID requires at least one entry in failure_reasons"
            )
        return self

    # ── Helpers ─────────────────────────────────────────────────────────────

    @property
    def effective_threshold(self) -> float:
        """Confidence threshold that applies to this case given its RiskTier."""
        return THRESHOLD_BY_TIER[self.risk_tier]

    @property
    def needs_hitl(self) -> bool:
        """True if this triage output should route to HITL."""
        return (
            self.status == TriageStatus.PENDING_HITL
            or self.status == TriageStatus.INVALID
            or self.confidence < self.effective_threshold
        )
