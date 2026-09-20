"""
schemas/audit.py
================
AuditLogEntry, HITLReviewRecord — audit trail schemas supporting selected 21 CFR Part 11 controls.

Design decisions (v4):
  - AuditLogEntry is the atomic immutable audit unit; one per agent invocation.
  - prompt_version is mandatory (CIOMS WG XIV Principle 4 — Transparency).
  - content_hash: SHA-256 of the agent's serialized output — provides tamper evidence.
  - HITLReviewRecord captures a human reviewer's correction; includes:
      * stage: which pipeline stage triggered the HITL queue
      * reviewer_id: identifier of the human reviewer (role-based, not personal)
      * original_value / corrected_value: for Learning Pipeline extraction
      * correction_rationale: free text
      * review_id: stage-specific UUID (tracked across the audit trail)
  - HITLStage enum identifies which agent/stage triggered the HITL interrupt.
  - AuditStatus tracks outcome per entry.

Compliance:
  - 21 CFR Part 11: immutable records, content_hash, reviewer_id, timestamps
  - CIOMS WG XIV Principle 7 (Governance and Accountability): PSMF-ready fields
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class AuditStatus(str, Enum):
    SUCCESS     = "SUCCESS"
    FAILED      = "FAILED"
    HITL_QUEUED = "HITL_QUEUED"
    SKIPPED     = "SKIPPED"


class HITLStage(str, Enum):
    """Identifies which pipeline stage triggered the HITL interrupt."""
    TRIAGE               = "TRIAGE"
    EXTRACTION           = "EXTRACTION"
    QC                   = "QC"
    CODING               = "CODING"
    CAUSALITY            = "CAUSALITY"
    LISTEDNESS           = "LISTEDNESS"
    NARRATIVE            = "NARRATIVE"
    SECONDARY_QC_AUDIT   = "SECONDARY_QC_AUDIT"   # HITL Learning Pipeline secondary review


class AuditLogEntry(BaseModel):
    """
    Single immutable audit record for one agent invocation.

    Fields cross-reference:
      - trace_id:      LangGraph thread_id (links all entries for a case)
      - run_id:        LangGraph run_id (unique per graph invocation)
      - review_id:     Stage-level UUID, regenerated at each HITL interrupt
      - agent_id:      Agent identifier string (e.g. "extraction-agent-v1")
      - prompt_version: Version of the prompt template used (21 CFR Part 11 reproducibility)
      - model_name:    Ollama model identifier (e.g. "llama3.1:8b-instruct-q4_K_M")
      - content_hash:  SHA-256 of the serialized agent output (tamper evidence)
      - processing_ms: Wall-clock latency of the agent invocation
    """
    model_config = {"frozen": True}

    entry_id:       str      = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID for this audit entry"
    )
    trace_id:       str      = Field(..., description="LangGraph thread_id — links all case entries")
    run_id:         str      = Field(..., description="LangGraph run_id — unique per pipeline run")
    review_id:      str      = Field(..., description="Stage-level review UUID; refreshed at each HITL")
    timestamp:      datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    agent_id:       str      = Field(..., description="e.g. 'triage-agent-v1'")
    prompt_version: str      = Field(
        ...,
        description="Prompt template version (CIOMS WG XIV Transparency requirement)"
    )
    model_name:     str      = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        description="Ollama model identifier"
    )
    model_temp:     float    = Field(default=0.0, ge=0.0, le=1.0)

    status:         AuditStatus
    content_hash:   str      = Field(
        ..., min_length=64, max_length=64,
        description="SHA-256 of the serialized agent output JSON"
    )
    processing_ms:  Optional[int]  = Field(None, ge=0, description="Agent wall-clock latency")
    error_message:  Optional[str]  = Field(None, max_length=2000)
    extra_metadata: dict[str, Any] = Field(default_factory=dict)


class HITLReviewRecord(BaseModel):
    """
    Record of a human reviewer's correction submitted via the HITL API.
    Written by the HITL FastAPI server into the LearningDB.
    Immutable after submission.
    """
    model_config = {"frozen": True}

    review_id:          str          = Field(
        ...,
        description="Matches the review_id of the AuditLogEntry that triggered HITL"
    )
    trace_id:           str          = Field(..., description="LangGraph thread_id")
    stage:              HITLStage
    reviewer_id:        str          = Field(
        ...,
        description=(
            "Role-based reviewer identifier (e.g. 'QPPV', 'MED-REVIEWER-01'). "
            "NOT a personal name (data minimisation)."
        )
    )
    reviewed_at:        datetime     = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_value:     Any          = Field(
        ...,
        description="The agent's original output that was disputed"
    )
    corrected_value:    Any          = Field(
        ...,
        description="The human reviewer's corrected value"
    )
    correction_rationale: str        = Field(..., min_length=5, max_length=1000)
    approved:           bool         = Field(
        ...,
        description="True = reviewer accepts the (corrected) case; False = rejected for rework"
    )
    learning_signal:    bool         = Field(
        default=True,
        description="If True, this correction is ingested into the LearningDB for pattern analysis"
    )

    @model_validator(mode="after")
    def rejection_requires_rationale_detail(self) -> "HITLReviewRecord":
        if not self.approved and len(self.correction_rationale) < 20:
            raise ValueError(
                "Rejected cases require a correction_rationale of at least 20 characters "
                "to support audit trail quality."
            )
        return self
