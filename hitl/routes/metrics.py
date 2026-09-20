"""
hitl/routes/metrics.py
=======================
Prometheus metrics + JSON stats endpoints for the HITL server.

Endpoints:
  GET /metrics          — Prometheus text exposition format (scraper-compatible)
  GET /api/v1/stats     — Same data as JSON (for dashboards / quick inspection)

Both endpoints require X-API-Key authentication (same as HITL review endpoints).

Content-Type:
  /metrics  → text/plain; version=0.0.4; charset=utf-8
  /stats    → application/json (FastAPI default)
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from hitl.auth import require_reviewer
from infra.auth_db import ReviewerRecord
from infra.metrics import metrics

router = APIRouter(tags=["Observability"])


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus metrics scrape endpoint",
    description=(
        "Returns all ICSR pipeline metrics in Prometheus text exposition format 0.0.4. "
        "Suitable for direct scraping by Prometheus, Grafana Agent, or VictoriaMetrics. "
        "Requires X-API-Key authentication."
    ),
)
async def prometheus_metrics(
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> PlainTextResponse:
    return PlainTextResponse(
        content=metrics.to_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get(
    "/api/v1/stats",
    summary="JSON metrics snapshot",
    description=(
        "Returns all ICSR pipeline metrics as a JSON object. "
        "Suitable for dashboards or manual inspection. "
        "Requires X-API-Key authentication."
    ),
)
async def json_stats(
    _reviewer: ReviewerRecord = Depends(require_reviewer),
) -> dict:
    return metrics.to_dict()
