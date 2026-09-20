"""
hitl/routes/export.py
======================
ICSR case export endpoints.

Endpoints:
  GET /api/v1/cases/{case_id}/export/e2b    — ICH E2B(R3) XML download
  GET /api/v1/cases/{case_id}/export/json   — Structured JSON (ICH field names)

Only completed cases (pipeline_complete=True, not HITL-pending) can be exported.
Both endpoints require X-API-Key authentication.

MedDRA / E2B(R3) note:
  See governance/decisions.md ADR-001 for the rationale behind using CTCAE terms
  as free-text <reactionmeddrapt> values. The exported XML is suitable for internal
  review and sandbox submissions; MedDRA coding must be completed by the QPPV before
  direct regulatory transmission.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response

from hitl.auth import require_reviewer
from infra.auth_db import ReviewerRecord
from infra.audit_db import AuditDB
from infra.e2b_exporter import E2BExporter

router = APIRouter(prefix="/api/v1/cases", tags=["Export"])

# Module-level exporter (no meddra_mapper — ADR-001 zero-cost mode)
_exporter = E2BExporter()


def _get_audit_db(request: Request) -> AuditDB:
    return request.app.state.audit_db


def _get_case_state(audit_db: AuditDB, case_id: str) -> dict:
    """
    Retrieve the most recent finalized state snapshot for a case from AuditDB.
    Raises 404 if no completed entries exist for case_id.
    """
    try:
        with audit_db._connect() as conn:
            row = conn.execute(
                """SELECT extra_metadata FROM audit_log
                   WHERE trace_id = ? AND status = 'SUCCESS'
                   ORDER BY timestamp DESC LIMIT 1""",
                (case_id,),
            ).fetchone()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AuditDB error: {exc}",
        )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No completed case found for case_id={case_id!r}. "
                "Only cases with status=SUCCESS can be exported. "
                "If the case is HITL-pending, submit a correction first."
            ),
        )

    import json
    raw = row["extra_metadata"]
    if isinstance(raw, bytes) and raw.startswith(b"\xfe\xfe"):
        # Encrypted — cannot decrypt here without the key; return minimal state
        return {"case_id": case_id, "_encrypted": True}
    try:
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {"case_id": case_id}


@router.get(
    "/{case_id}/export/e2b",
    summary="Export case as ICH E2B(R3) XML",
    description=(
        "Returns a well-formed ICH E2B(R3) XML document for a completed case. "
        "CTCAE terms are used in <reactionmeddrapt> per ADR-001 (zero-cost constraint). "
        "MedDRA coding must be completed by the QPPV before regulatory submission. "
        "Requires X-API-Key."
    ),
    responses={
        200: {
            "content": {"application/xml": {}},
            "description": "E2B(R3) XML document",
        },
    },
)
def export_e2b(
    case_id:   str,
    request:   Request,
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> Response:
    audit_db = _get_audit_db(request)
    state    = _get_case_state(audit_db, case_id)

    if state.get("_encrypted"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Case data is stored in encrypted form. "
                "Run export from a server instance with DB_ENCRYPTION_KEY set."
            ),
        )

    xml_str = _exporter.export(state=state, case_id=case_id)

    filename = f"{case_id}.xml"
    return Response(
        content     = xml_str,
        media_type  = "application/xml",
        headers     = {"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get(
    "/{case_id}/export/json",
    summary="Export case as structured JSON (ICH field names)",
    description=(
        "Returns a structured JSON representation of the case using ICH E2B(R3)-aligned "
        "field names. Useful for dashboard integration or manual MedDRA coding workflows. "
        "Requires X-API-Key."
    ),
)
def export_json(
    case_id:   str,
    request:   Request,
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> dict:
    import json as _json

    audit_db = _get_audit_db(request)
    state    = _get_case_state(audit_db, case_id)

    if state.get("_encrypted"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Case data is encrypted. DB_ENCRYPTION_KEY must be set.",
        )

    entities        = state.get("extracted_entities") or {}
    causality_matrix = state.get("causality_matrix") or {}
    coded_events    = state.get("coded_events") or []
    triage          = state.get("triage_output") or {}

    return {
        "case_id":             case_id,
        "e2b_format_version":  "R3",
        "meddra_note":         (
            "CTCAE v5 preferred terms used. MedDRA LLT/PT coding required before "
            "regulatory submission. See governance/decisions.md ADR-001."
        ),
        "safetyreportid":      case_id,
        "seriousness":         triage.get("risk_tier"),
        "patient": {
            "sex":       entities.get("patient_sex") if isinstance(entities, dict) else None,
            "age":       entities.get("patient_age") if isinstance(entities, dict) else None,
            "drugs":     entities.get("suspect_drugs", []) if isinstance(entities, dict) else [],
            "reactions": coded_events,
        },
        "causality":           causality_matrix,
        "narrative":           state.get("final_narrative"),
    }
