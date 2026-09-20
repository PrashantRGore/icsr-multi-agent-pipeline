"""
hitl/schemas.py
================
Request/response Pydantic models for the HITL FastAPI server.

All models are validated at the API boundary to prevent malformed
corrections from corrupting the audit trail.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class HITLCorrectionRequest(BaseModel):
    """
    Payload submitted by a human reviewer to correct a halted case.

    Fields:
      reviewer_id         : Role-based identifier (e.g. 'QPPV', 'MED-REVIEWER-01')
                            NOT a personal name — data minimisation.
      correction          : Dict of corrected state field-name → value
                            e.g. {"causality_matrix": {...}} or
                                 {"extracted_entities": {...}}
      correction_rationale: Free-text explanation of the correction
      approved            : True = accept the (corrected) case; False = reject for rework
      learning_signal     : If True (default), correction is ingested into LearningDB
    """
    reviewer_id:          str         = Field(..., min_length=2, max_length=50)
    correction:           dict[str, Any] = Field(
        default_factory=dict,
        description="Partial state dict of corrected fields"
    )
    correction_rationale: str         = Field(..., min_length=5, max_length=1000)
    approved:             bool        = Field(
        ...,
        description="True = reviewer accepts and continues; False = reject for rework"
    )
    learning_signal:      bool        = Field(
        default=True,
        description="If True, this correction feeds the Learning Pipeline"
    )


class HITLCorrectionResponse(BaseModel):
    """Response returned after a HITL correction is accepted."""
    review_id:        str
    case_id:          str
    thread_id:        str
    status:           str             # "resumed" | "rejected" | "still_halted"
    pipeline_complete:bool
    next_stage:       Optional[str]
    message:          str
    timestamp:        str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HITLQueueEntry(BaseModel):
    """Summary of one entry in the HITL queue (for GET /api/v1/review/queue)."""
    review_id:    str
    thread_id:    str
    case_id:      Optional[str]
    hitl_stage:   str
    enqueued_at:  str
    error_log:    list[str]


class HITLQueueResponse(BaseModel):
    """Response for GET /api/v1/review/queue."""
    total_pending: int
    entries:       list[HITLQueueEntry]


class CaseRunRequest(BaseModel):
    """
    Request body for POST /api/v1/cases/run — submit a new case.
    """
    case_id:       str              = Field(..., min_length=3, max_length=100)
    raw_narrative: str              = Field(..., min_length=50, max_length=50_000)
    source_type:   Optional[str]   = None
    country:       Optional[str]   = Field(None, max_length=50)
    received_date: Optional[str]   = Field(None, description="YYYY-MM-DD")


class CaseRunResponse(BaseModel):
    """Response for POST /api/v1/cases/run."""
    case_id:           str
    thread_id:         str
    status:            str   # "complete" | "hitl_pending" | "error"
    pipeline_complete: bool
    pipeline_halted:   bool
    hitl_stage:        Optional[str]
    review_id:         Optional[str]
    final_narrative:   Optional[str]
    message:           str
    timestamp:         str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
