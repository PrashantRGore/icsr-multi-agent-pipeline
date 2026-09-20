"""
hitl/routes/review.py
======================
FastAPI router for HITL review endpoints.

Endpoints:
  GET  /api/v1/review/queue                — List all pending HITL cases
  GET  /api/v1/review/{review_id}          — Get details for one HITL case
  POST /api/v1/review/{review_id}/submit   — Submit human correction + resume pipeline
  POST /api/v1/cases/run                   — Submit a new case for processing

21 CFR Part 11 audit trail controls:
  - Every correction is written to AuditDB (hitl_review_log table)
  - reviewer_id is recorded on every correction (no anonymous edits)
  - Rejected cases remain in audit trail with approved=False

Error handling:
  - 404 if review_id not in HITL queue
  - 422 from Pydantic validation on correction payload
  - 500 if pipeline fails to resume (logged + returned with detail)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from hitl.auth import require_reviewer
from hitl.schemas import (
    CaseRunRequest,
    CaseRunResponse,
    HITLCorrectionRequest,
    HITLCorrectionResponse,
    HITLQueueEntry,
    HITLQueueResponse,
)
from infra.auth_db import ReviewerRecord
from agents.secondary_qc_auditor import SecondaryQCAuditor
from infra.metrics import metrics
from schemas.audit import HITLReviewRecord, HITLStage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["HITL Review"])


# ── Dependency helpers ────────────────────────────────────────────────────────

def get_pipeline(request: Request):
    """Extract the ICSRPipeline from FastAPI application state."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialised. Start the server with lifespan enabled.",
        )
    return pipeline


def get_audit_db(request: Request):
    """Extract the AuditDB from FastAPI application state."""
    db = getattr(request.app.state, "audit_db", None)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AuditDB not initialised.",
        )
    return db


def get_learning_db(request: Request):
    """Extract the LearningDB from FastAPI application state."""
    db = getattr(request.app.state, "learning_db", None)
    if db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LearningDB not initialised.",
        )
    return db


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/review/queue",
    response_model=HITLQueueResponse,
    summary="List all pending HITL cases",
)
def list_hitl_queue(
    pipeline=Depends(get_pipeline),
    reviewer: ReviewerRecord = Depends(require_reviewer),
) -> HITLQueueResponse:
    """
    Return a summary of all cases currently waiting in the HITL queue.
    Each entry includes the review_id, case_id, stage, and recent errors.
    """
    pending = pipeline.hitl_queue.list_pending()
    entries = [HITLQueueEntry(**e) for e in pending]
    return HITLQueueResponse(
        total_pending=len(entries),
        entries=entries,
    )


@router.get(
    "/review/{review_id}",
    summary="Get HITL case details",
)
def get_hitl_case(
    review_id: str,
    pipeline=Depends(get_pipeline),
    reviewer: ReviewerRecord = Depends(require_reviewer),
) -> dict:
    """
    Return full state snapshot for the given review_id.
    Includes all pipeline outputs accumulated so far (for reviewer context).
    """
    entry = pipeline.hitl_queue.get(review_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending HITL case found for review_id={review_id!r}",
        )
    snap = entry["state_snap"]
    return {
        "review_id":   review_id,
        "thread_id":   entry["thread_id"],
        "enqueued_at": entry["enqueued_at"],
        "case_id":     snap.get("case_id"),
        "hitl_stage":  str(snap.get("hitl_stage", "")),
        "risk_tier":   str(snap.get("risk_tier", "")),
        "error_log":   snap.get("error_log", []),
        "raw_narrative": snap.get("raw_narrative", ""),
        "partial_e2b": snap.get("partial_e2b", {}),
        # Omit large objects (extracted_entities, causality_matrix) for brevity
        # Reviewer can look up thread state in audit DB if needed
    }


@router.post(
    "/review/{review_id}/submit",
    response_model=HITLCorrectionResponse,
    summary="Submit a HITL correction and resume the pipeline",
)
def submit_hitl_correction(
    review_id:   str,
    body:        HITLCorrectionRequest,
    pipeline=Depends(get_pipeline),
    audit_db=Depends(get_audit_db),
    learning_db=Depends(get_learning_db),
    reviewer: ReviewerRecord = Depends(require_reviewer),
) -> HITLCorrectionResponse:
    """
    Accept a human reviewer's correction and resume the pipeline.

    The correction dict is merged into the checkpoint state, pipeline_halted
    is cleared, and the pipeline re-runs from the failed stage.

    Returns the final state summary after resumption.
    """
    # Validate review_id exists before writing to audit trail
    entry = pipeline.hitl_queue.get(review_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No pending HITL case found for review_id={review_id!r}",
        )

    state_snap = entry["state_snap"]
    thread_id  = entry["thread_id"]
    case_id    = state_snap.get("case_id", "unknown")
    stage_str  = str(state_snap.get("hitl_stage", "UNKNOWN")).upper()

    # ── Write to 21 CFR Part 11 audit trail ──────────────────────────────────
    try:
        hitl_stage = HITLStage(stage_str)
    except ValueError:
        hitl_stage = HITLStage.NARRATIVE   # Safe fallback

    audit_record = HITLReviewRecord(
        review_id             = review_id,
        trace_id              = thread_id,
        stage                 = hitl_stage,
        reviewer_id           = reviewer.reviewer_id,  # ← from auth key, NOT body
        original_value        = {
            k: str(state_snap.get(k))[:500]
            for k in ("triage_output", "causality_matrix", "coded_events",
                      "listedness_evaluations", "extracted_entities")
            if state_snap.get(k) is not None
        },
        corrected_value       = body.correction,
        correction_rationale  = body.correction_rationale,
        approved              = body.approved,
        learning_signal       = body.learning_signal,
    )
    try:
        audit_db.insert_hitl_review(audit_record)
    except Exception as audit_exc:
        logger.error("HITL audit write failed: %s", audit_exc)
        # Do not block the reviewer -- log the failure and continue

    # -- Learning Pipeline: classify correction and write signals -------------
    if body.approved and body.learning_signal:
        try:
            auditor = SecondaryQCAuditor(audit_db=audit_db, learning_db=learning_db)
            sig_result = auditor.audit_correction(
                review_record = audit_record,
                state_snap    = state_snap,
            )
            logger.info(
                "HITL learning signals: review_id=%s signals=%d fields=%s",
                review_id, sig_result.signals_written, sig_result.field_paths,
            )
        except Exception as learn_exc:
            # Learning Pipeline errors must NEVER block the reviewer
            logger.error(
                "SecondaryQCAuditor failed for review_id=%s: %s",
                review_id, learn_exc,
            )

    # ── Handle rejection ──────────────────────────────────────────────────────
    if not body.approved:
        pipeline.hitl_queue.dequeue(review_id)
        metrics.inc_hitl_review(approved=False)
        logger.warning(
            "HITL: case=%s REJECTED by reviewer=%s  rationale=%s",
            case_id, reviewer.reviewer_id, body.correction_rationale[:80]
        )
        return HITLCorrectionResponse(
            review_id        = review_id,
            case_id          = case_id,
            thread_id        = thread_id,
            status           = "rejected",
            pipeline_complete= False,
            next_stage       = None,
            message          = (
                f"Case {case_id} rejected by {reviewer.reviewer_id}. "
                "Returned to originator for rework."
            ),
        )

    # ── Resume pipeline ───────────────────────────────────────────────────────
    try:
        final_state = pipeline.resume_case(
            review_id  = review_id,
            correction = body.correction,
        )
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(ve))
    except Exception as exc:
        logger.exception("HITL resume failed for review_id=%s: %s", review_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline resume failed: {exc}",
        )

    is_complete = bool(final_state.get("pipeline_complete"))
    is_halted   = bool(final_state.get("pipeline_halted"))
    new_stage   = str(final_state.get("next_stage", "")) or None

    return HITLCorrectionResponse(
        review_id        = review_id,
        case_id          = case_id,
        thread_id        = thread_id,
        status           = "complete" if is_complete else ("still_halted" if is_halted else "resumed"),
        pipeline_complete= is_complete,
        next_stage       = new_stage,
        message          = (
            f"Case {case_id} pipeline {'completed' if is_complete else 'resumed'} "
            f"successfully by {reviewer.reviewer_id}."
        ),
    )


@router.post(
    "/cases/run",
    response_model=CaseRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a new ICSR case for processing",
)
def run_new_case(
    body:     CaseRunRequest,
    request:  Request,
    pipeline=Depends(get_pipeline),
    reviewer: ReviewerRecord = Depends(require_reviewer),
) -> CaseRunResponse:
    """
    Submit a new ICSR narrative for end-to-end pipeline processing.

    The pipeline runs synchronously and returns the final state.
    If HITL is triggered, the response includes the review_id for
    the reviewer to use with /review/{review_id}/submit.

    PII de-identification is applied to the raw narrative before processing
    if PIIDeidentifier is available (presidio installed + configured).
    """
    logger.info("API: run_new_case  case_id=%s", body.case_id)
    metrics.inc_cases_total()

    # ── PII De-identification ─────────────────────────────────────────────────────
    deidentifier = getattr(request.app.state, "deidentifier", None)
    narrative_to_process = body.raw_narrative
    if deidentifier is not None:
        try:
            deid_result = deidentifier.deidentify(
                text     = body.raw_narrative,
                trace_id = body.case_id,
            )
            narrative_to_process = deid_result.anonymized_text
            if deid_result.entities_found:
                logger.info(
                    "PIIDeidentifier: case_id=%s redacted %d entities, map saved to %s",
                    body.case_id, len(deid_result.entities_found), deid_result.map_path,
                )
        except Exception as deid_exc:
            # ── Privacy-first: de-identification failure → REJECT case ────────
            # Sending unredacted PHI to the LLM is worse than losing one case.
            # Route to authorized human review; do not process the original text.
            logger.error(
                "PIIDeidentifier: SECURITY EVENT — de-identification FAILED for "
                "case_id=%s: %s. Rejecting case to prevent PHI exposure (privacy-first policy).",
                body.case_id, deid_exc,
            )
            raise HTTPException(
                status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
                detail = (
                    f"PII de-identification failed for case_id={body.case_id!r}. "
                    "Case rejected — patient privacy protected. "
                    "Please inspect the narrative manually and re-submit, "
                    "or contact the system administrator to diagnose the Presidio engine."
                ),
            )

    import time as _time
    _t0 = _time.monotonic()
    try:
        final_state = pipeline.run_case(
            case_id       = body.case_id,
            raw_narrative = narrative_to_process,
            source_type   = body.source_type,
            country       = body.country,
            received_date = body.received_date,
        )
    except Exception as exc:
        metrics.inc_cases_failed()
        logger.exception("Pipeline failed for case_id=%s: %s", body.case_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    metrics.observe_pipeline_duration(_time.monotonic() - _t0)
    is_complete = bool(final_state.get("pipeline_complete"))
    is_halted   = bool(final_state.get("pipeline_halted"))
    review_id   = final_state.get("review_id") if is_halted else None
    hitl_stage  = str(final_state.get("hitl_stage", "")) or None

    if is_halted:
        metrics.inc_cases_hitl()
    elif is_complete:
        metrics.inc_cases_complete()

    return CaseRunResponse(
        case_id           = body.case_id,
        thread_id         = final_state.get("trace_id", ""),
        status            = "complete" if is_complete else ("hitl_pending" if is_halted else "error"),
        pipeline_complete = is_complete,
        pipeline_halted   = is_halted,
        hitl_stage        = hitl_stage,
        review_id         = review_id,
        final_narrative   = final_state.get("final_narrative"),
        message           = (
            f"Case {body.case_id} completed successfully."
            if is_complete else
            f"Case {body.case_id} requires HITL review at stage {hitl_stage}. "
            f"Use review_id={review_id} to submit correction."
        ),
    )
