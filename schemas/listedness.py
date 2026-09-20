"""
schemas/listedness.py
=====================
ListednessEvaluation — output of the Listedness Agent.

Determines whether each coded AE is "listed" (expected) or "unlisted" (unexpected)
according to the Reference Safety Information (RSI) in the FDA DailyMed SPL XML.

An unlisted event on a serious case triggers expedited regulatory reporting
(15-day rule under 21 CFR 314.81(b)(1) / ICH E2D).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ListednessStatus(str, Enum):
    """
    LISTED:   AE appears in the RSI / core data sheet (expected event)
    UNLISTED: AE does NOT appear in RSI → expedited reporting may apply
    UNKNOWN:  RSI not available or assessment indeterminate
    """
    LISTED   = "LISTED"
    UNLISTED = "UNLISTED"
    UNKNOWN  = "UNKNOWN"


class ListednessEvaluation(BaseModel):
    """
    Listedness result for a single coded event.
    Immutable after creation.
    """
    model_config = {"frozen": True}

    event_node_id:         str              = Field(..., description="VerbatimEvent.node_id")
    drug_node_id:          str              = Field(..., description="SuspectDrug.node_id")
    verbatim_term:         str
    coded_term:            Optional[str]    = None    # OAE or CTCAE term used for RSI lookup
    listedness_status:     ListednessStatus
    rsi_source:            Optional[str]    = Field(
        None,
        description="FDA DailyMed SetID or SPL section reference"
    )
    rsi_section_text:      Optional[str]    = Field(
        None,
        max_length=2000,
        description="Verbatim RSI text from SPL that supports the listing decision"
    )
    expedited_reporting:   bool             = Field(
        False,
        description=(
            "True when listedness_status=UNLISTED and the parent event is serious. "
            "Triggers 15-day reporting obligation."
        )
    )
    confidence:            float            = Field(..., ge=0.0, le=1.0)
    agent_id:              str              = "listedness-agent-v1"
    prompt_version:        str              = "listedness-prompt-v1.0"
