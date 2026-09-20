"""
schemas/coding.py
=================
CodedEvent — output of the Coding Agent.

Each VerbatimEvent receives one CodedEvent mapping it to a standardized term.
The Coding Agent attempts OAE (Ontology of Adverse Events) first, then
falls back to NCI CTCAE v5 if OAE confidence < 0.80.

CodingStatus.NEEDS_MANUAL indicates HITL routing is required.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CodingStatus(str, Enum):
    AUTO_CODED     = "AUTO_CODED"      # High-confidence automated mapping
    LOW_CONFIDENCE = "LOW_CONFIDENCE"  # Mapped but below 0.80 OAE confidence
    NEEDS_MANUAL   = "NEEDS_MANUAL"    # No acceptable mapping found → HITL
    FALLBACK_CTCAE = "FALLBACK_CTCAE"  # OAE failed; CTCAE used


class CodedEvent(BaseModel):
    """
    Coded mapping for a single VerbatimEvent.

    oae_term / oae_id: Ontology of Adverse Events preferred term and OBO ID.
    ctcae_term: NCI CTCAE v5 fallback term.
    meddra_pt: placeholder for future MedDRA mapping (not implemented — zero-cost stack).
    """
    model_config = {"frozen": True}

    event_node_id:      str             = Field(..., description="VerbatimEvent.node_id")
    verbatim_term:      str
    oae_term:           Optional[str]   = None
    oae_id:             Optional[str]   = Field(None, pattern=r"^OAE:\d{7}$")
    oae_confidence:     Optional[float] = Field(None, ge=0.0, le=1.0)
    ctcae_term:         Optional[str]   = None
    ctcae_grade:        Optional[int]   = Field(None, ge=1, le=5)
    meddra_pt:          Optional[str]   = Field(
        None,
        description=(
            "MedDRA Preferred Term — placeholder only. "
            "MedDRA requires a subscription; this field populated only when provided externally."
        )
    )
    coding_status:      CodingStatus
    agent_id:           str             = "coding-agent-v1"
    prompt_version:     str             = "coding-prompt-v1.0"
