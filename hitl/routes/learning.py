"""
hitl/routes/learning.py
========================
Learning pipeline report API endpoints.

Exposes LearningDB data as authenticated JSON endpoints for dashboards
and governance tooling.

Endpoints:
  GET /api/v1/learning/report   — Full report (agent accuracy + bias)
  GET /api/v1/learning/agents   — Per-agent accuracy summary only
  GET /api/v1/learning/bias     — Demographic stratification table only

All endpoints require X-API-Key authentication.

Note on E2B(R3):
  These endpoints return system performance metrics (internal governance).
  ICH E2B(R3) applies to ICSR case data transmission — see /api/v1/cases/export/e2b.
  See governance/decisions.md ADR-001 for the full rationale.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from hitl.auth import require_reviewer
from infra.auth_db import ReviewerRecord
from infra.learning_db import LearningDB

router = APIRouter(prefix="/api/v1/learning", tags=["Learning & Governance"])

_ALL_AGENTS = [
    "triage-agent-v1",
    "extraction-agent-v1",
    "qc-agent-v1",
    "coding-agent-v1",
    "causality-agent-v1",
    "listedness-agent-v1",
    "narrative-agent-v1",
    "secondary-qc-auditor-v1",
]


def _get_learning_db(request: Request) -> LearningDB:
    return request.app.state.learning_db


@router.get(
    "/report",
    summary="Full HITL learning governance report",
    description=(
        "Returns per-agent correction rates, field-level error patterns, and "
        "demographic bias stratification (CIOMS WG XIV Principle 6). "
        "This is a system performance report — not an E2B(R3) ICSR document. "
        "See governance/decisions.md ADR-001."
    ),
)
def learning_report(
    request:  Request,
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> dict:
    db = _get_learning_db(request)
    total = db.count_signals()

    agent_stats = []
    for agent_id in _ALL_AGENTS:
        acc = db.agent_accuracy(agent_id)
        corrections = acc["total_corrections"]
        top_class = (
            max(acc["by_classification"], key=acc["by_classification"].get)
            if acc["by_classification"] else None
        )
        agent_stats.append({
            "agent_id":           agent_id,
            "corrections":        corrections,
            "correction_rate_pct": round(corrections / total * 100, 1) if total else 0.0,
            "top_error_class":    top_class,
            "by_classification":  acc["by_classification"],
        })

    return {
        "total_signals":  total,
        "agent_accuracy": agent_stats,
        "top_patterns":   db.top_patterns(limit=20),
        "bias_report":    db.demographic_bias_report(),
    }


@router.get(
    "/agents",
    summary="Per-agent accuracy summary",
    description="Correction rate per pipeline agent. Requires X-API-Key.",
)
def agents_summary(
    request:  Request,
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> dict:
    db    = _get_learning_db(request)
    total = db.count_signals()
    stats = [
        {
            "agent_id":           a,
            "corrections":        (acc := db.agent_accuracy(a))["total_corrections"],
            "correction_rate_pct": (
                round(acc["total_corrections"] / total * 100, 1) if total else 0.0
            ),
            "top_error_class": (
                max(acc["by_classification"], key=acc["by_classification"].get)
                if acc["by_classification"] else None
            ),
        }
        for a in _ALL_AGENTS
    ]
    return {"total_signals": total, "agents": stats}


@router.get(
    "/bias",
    summary="Demographic bias stratification (CIOMS WG XIV Principle 6)",
    description=(
        "Returns correction counts stratified by patient_sex, patient_age_group, "
        "and patient_ethnicity. Requires X-API-Key."
    ),
)
def bias_report(
    request:  Request,
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> dict:
    db = _get_learning_db(request)
    return {
        "total_signals": db.count_signals(),
        "bias_table":    db.demographic_bias_report(),
    }
