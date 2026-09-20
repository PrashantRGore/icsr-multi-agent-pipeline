"""
hitl/main.py
=============
FastAPI application factory for the HITL review server.

Startup lifecycle:
  1. Load environment configuration (.env / environment variables)
  2. Configure JSON structured logging + PIIRedactionFilter
  3. Initialise OllamaClient (temperature=0.0, Semaphore(1))
  4. Initialise AuditDB (21 CFR Part 11 SQLite, column-level encryption)
  5. Initialise AuthDB (reviewer API-key store)
  6. Initialise RxNormClient (SQLite-cached NLM API)
  7. Load FAISSIndex for OAE and CTCAE (if available)
  8. Build NodeFactory → ICSRPipeline
  9. Attach shared objects to app.state for dependency injection
 10. Reload surviving HITL queue entries from previous server run

Running:
  uvicorn hitl.main:app --host 0.0.0.0 --port 8000 --reload

Or via the CLI entry point:
  python -m hitl.main

Environment variables (see .env.example):
  OLLAMA_HOST          : Ollama server URL (default: http://localhost:11434)
  OLLAMA_MODEL         : Model name (default: llama3.1:8b-instruct-q4_K_M)
  AUDIT_DB_PATH        : SQLite audit DB path (default: audit/audit.db)
  AUTH_DB_PATH         : SQLite reviewer key DB (default: audit/auth.db)
  LEARNING_DB_PATH     : SQLite learning DB path (default: audit/learning.db)
  RXNORM_CACHE_PATH    : RxNorm SQLite cache (default: data/rxnorm_cache.db)
  OAE_INDEX_PATH       : OAE FAISS index directory (default: data/oae.faiss)
  CTCAE_INDEX_PATH     : CTCAE FAISS index directory (default: data/ctcae.faiss)
  NEG_PATH             : Clinical negatives JSON (default: data/clinical_negatives.json)
  CHECKPOINT_DB_PATH   : LangGraph checkpoint DB (default: checkpoints/pipeline.db)
  PII_MAPS_DIR         : Directory for entity map files (default: data/pii_maps)
  PII_ENCRYPTION_KEY   : Fernet key for encrypting entity maps (REQUIRED in production)
  DB_ENCRYPTION_KEY    : Fernet key for AuditDB column encryption (REQUIRED in production)
  APP_ENV              : Set to 'production' to enforce encryption key requirements at startup
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from graph.hitl_interrupt import HITLQueue, get_checkpointer
from graph.nodes import NodeFactory
from graph.pipeline import ICSRPipeline
from hitl.routes.export import router as export_router
from hitl.routes.learning import router as learning_router
from hitl.routes.metrics import router as metrics_router
from hitl.routes.review import router as review_router
from infra.audit_db import AuditDB
from infra.auth_db import AuthDB
from infra.json_log_formatter import JSONFormatter
from infra.learning_db import LearningDB
from infra.log_filter import PIIRedactionFilter
from infra.metrics import metrics
from infra.ollama_client import OllamaClient
from infra.rxnorm_client import RxNormClient

# ── Logging setup ─────────────────────────────────────────────────────────────
# Use JSON formatter in production; PIIRedactionFilter applied in lifespan
# so it runs AFTER the formatter is installed.
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
if not _root_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(JSONFormatter(service_name="icsr-hitl"))
    _root_logger.addHandler(_handler)

logger = logging.getLogger(__name__)

_APP_VERSION = "0.6.0"


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan: initialize all resources at startup, clean up at shutdown."""
    # Apply PII redaction to ALL log output before anything else
    logging.getLogger().addFilter(PIIRedactionFilter())
    logger.info("HITL Server: starting up …")

    # ── Configuration from environment ───────────────────────────────────────
    app_env          = os.getenv("APP_ENV", "development").lower()
    ollama_host      = os.getenv("OLLAMA_HOST",        "http://localhost:11434")
    ollama_model     = os.getenv("OLLAMA_MODEL",       "llama3.1:8b-instruct-q4_K_M")
    audit_db_path    = os.getenv("AUDIT_DB_PATH",      "audit/audit.db")
    auth_db_path     = os.getenv("AUTH_DB_PATH",       "audit/auth.db")
    learning_db_path = os.getenv("LEARNING_DB_PATH",   "audit/learning.db")
    rxnorm_cache     = os.getenv("RXNORM_CACHE_PATH",  "data/rxnorm_cache.db")
    oae_path         = os.getenv("OAE_INDEX_PATH",     "data/oae.faiss")
    ctcae_path       = os.getenv("CTCAE_INDEX_PATH",   "data/ctcae.faiss")
    neg_path         = os.getenv("NEG_PATH",           "data/clinical_negatives.json")
    ckpt_path        = os.getenv("CHECKPOINT_DB_PATH", "checkpoints/pipeline.db")
    pii_maps_dir     = os.getenv("PII_MAPS_DIR",       "data/pii_maps")
    pii_enc_key      = os.getenv("PII_ENCRYPTION_KEY", "")
    db_enc_key       = os.getenv("DB_ENCRYPTION_KEY",  "")

    # ── Production encryption guard ──────────────────────────────────────────
    # In production, both encryption keys are REQUIRED. Unencrypted audit logs
    # and PII entity maps in a production environment is a privacy violation.
    if app_env == "production":
        missing_keys: list[str] = []
        if not db_enc_key:
            missing_keys.append("DB_ENCRYPTION_KEY")
        if not pii_enc_key:
            missing_keys.append("PII_ENCRYPTION_KEY")
        if missing_keys:
            logger.critical(
                "STARTUP ABORTED: APP_ENV=production but required encryption key(s) "
                "are not set: %s. "
                "Generate keys with: python -c \"from cryptography.fernet import Fernet; "
                "print(Fernet.generate_key().decode())\" "
                "and set them in your .env file.",
                ", ".join(missing_keys),
            )
            sys.exit(1)
        logger.info("HITL Server: production mode — encryption keys verified.")
    else:
        if not db_enc_key:
            logger.warning(
                "DB_ENCRYPTION_KEY not set — AuditDB stored without column encryption. "
                "Set APP_ENV=production to enforce encryption at startup."
            )
        if not pii_enc_key:
            logger.warning(
                "PII_ENCRYPTION_KEY not set — PII entity maps stored in plaintext. "
                "Set APP_ENV=production to enforce encryption at startup."
            )

    # ── Infrastructure ───────────────────────────────────────────────────────
    llm         = OllamaClient(host=ollama_host, model=ollama_model)
    audit_db    = AuditDB(db_path=audit_db_path, encryption_key=db_enc_key or None)
    auth_db     = AuthDB(db_path=auth_db_path)
    learning_db = LearningDB(db_path=learning_db_path)
    rxnorm      = RxNormClient(db_path=rxnorm_cache)

    # Store references for health check
    app.state._ollama_host  = ollama_host
    app.state._ollama_model = ollama_model

    # ── PII De-identification (optional — graceful skip if presidio not available) ──
    try:
        from infra.pii_deidentifier import PIIDeidentifier
        deidentifier: PIIDeidentifier | None = PIIDeidentifier(
            maps_dir       = pii_maps_dir,
            encryption_key = pii_enc_key or None,
        )
        logger.info("PIIDeidentifier: active")
    except ImportError:
        deidentifier = None
        logger.warning("PIIDeidentifier: presidio not installed — PII de-identification disabled")
    except Exception as pii_exc:
        deidentifier = None
        logger.warning("PIIDeidentifier: failed to initialise: %s — PII de-identification disabled", pii_exc)

    # ── FAISS indexes (optional — agent degrades gracefully if missing) ───────
    oae_idx   = _load_faiss(oae_path,   label="OAE")
    ctcae_idx = _load_faiss(ctcae_path, label="CTCAE")

    # ── LangGraph pipeline ────────────────────────────────────────────────────
    factory      = NodeFactory(
        llm=llm, audit_db=audit_db, rxnorm=rxnorm,
        oae_idx=oae_idx, ctcae_idx=ctcae_idx,
        neg_path=Path(neg_path),
    )
    checkpointer = get_checkpointer(Path(ckpt_path))
    hitl_queue   = HITLQueue(audit_db=audit_db)
    pipeline     = ICSRPipeline(
        factory=factory,
        checkpointer=checkpointer,
        hitl_queue=hitl_queue,
    )

    # ── Reload surviving HITL queue entries from previous server run ────────────
    pending_cases = audit_db.get_pending_hitl()
    for row in pending_cases:
        hitl_queue._queue[row["review_id"]] = {
            "thread_id":   row["thread_id"],
            "state_snap":  row["state_snap"],
            "enqueued_at": row["enqueued_at"],
        }
    if pending_cases:
        logger.info(
            "HITL Queue: reloaded %d pending case(s) from AuditDB after restart",
            len(pending_cases),
        )

    # ── Attach to app.state for dependency injection ──────────────────────────
    app.state.llm           = llm
    app.state.audit_db      = audit_db
    app.state.auth_db       = auth_db
    app.state.learning_db   = learning_db
    app.state.rxnorm        = rxnorm
    app.state.pipeline      = pipeline
    app.state.deidentifier  = deidentifier

    logger.info("HITL Server: all components initialised — ready to accept requests")

    yield  # ← Application runs here

    # ── Cleanup ───────────────────────────────────────────────────────────────
    logger.info(
        "HITL Server: shutting down. HITL queue had %d pending cases.",
        len(hitl_queue)
    )


def _load_faiss(path: str, label: str):
    """Attempt to load a FAISSIndex; return None if file doesn't exist."""
    from infra.faiss_index import FAISSIndex
    p = Path(path)
    if p.exists():
        try:
            idx = FAISSIndex(index_dir=p)
            logger.info("FAISS[%s]: loaded from %s", label, p)
            return idx
        except Exception as exc:
            logger.warning("FAISS[%s]: failed to load from %s: %s", label, p, exc)
    else:
        logger.warning("FAISS[%s]: index not found at %s — coding agent will degrade", label, p)
    return None


# ── Application factory ────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title       = "ICSR Multi-Agent HITL Server",
        description = (
            "Zero-cost, privacy-first, local ICSR Processing Engine HITL API.\n\n"
            "Provides endpoints for submitting new cases, reviewing halted cases, "
            "and submitting human corrections to resume the LangGraph pipeline."
        ),
        version     = _APP_VERSION,
        lifespan    = lifespan,
    )

    # Allow local dashboard / CLI to call the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(review_router)
    app.include_router(metrics_router)
    app.include_router(learning_router)
    app.include_router(export_router)

    # ── Health check — liveness + readiness ──────────────────────────────────
    @app.get("/health", tags=["Health"], summary="Liveness and readiness probe")
    async def health(request: Request) -> JSONResponse:
        """
        Returns 200 if all critical subsystems are operational.
        Returns 503 if any critical check fails.
        Ollama degradation is non-fatal (returns 200 with status=degraded).
        """
        checks: dict[str, Any] = {}
        overall_ok = True

        # ── AuditDB check ─────────────────────────────────────────────────────
        try:
            audit_db: AuditDB = request.app.state.audit_db
            entry_count = audit_db.count_entries()   # lightweight COUNT(*)
            checks["audit_db"] = {"status": "ok", "entry_count": entry_count}
        except Exception as exc:
            checks["audit_db"] = {"status": "error", "detail": str(exc)}
            overall_ok = False

        # ── AuthDB check ──────────────────────────────────────────────────────
        try:
            auth_db: AuthDB = request.app.state.auth_db
            reviewer_count = auth_db.count_reviewers()
            checks["auth_db"] = {"status": "ok", "reviewer_count": reviewer_count}
        except Exception as exc:
            checks["auth_db"] = {"status": "error", "detail": str(exc)}
            overall_ok = False

        # ── HITL queue check ──────────────────────────────────────────────────
        try:
            pipeline = request.app.state.pipeline
            pending  = len(pipeline.hitl_queue)
            metrics.set_queue_pending(pending)
            checks["hitl_queue"] = {"status": "ok", "pending": pending}
        except Exception as exc:
            checks["hitl_queue"] = {"status": "error", "detail": str(exc)}

        # ── Ollama check (non-fatal — degrades gracefully) ────────────────────
        try:
            import httpx
            host = getattr(request.app.state, "_ollama_host", "http://localhost:11434")
            model = getattr(request.app.state, "_ollama_model", "unknown")
            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{host}/api/tags")
            latency_ms = round((time.monotonic() - t0) * 1000, 1)
            if resp.status_code == 200:
                checks["ollama"] = {"status": "ok", "model": model, "latency_ms": latency_ms}
            else:
                checks["ollama"] = {"status": "degraded", "http_status": resp.status_code}
        except Exception as exc:
            checks["ollama"] = {"status": "degraded", "detail": str(exc)}

        # ── Aggregate ─────────────────────────────────────────────────────────
        any_degraded = any(
            c.get("status") == "degraded" for c in checks.values()
        )
        status_str = "healthy" if overall_ok else "unhealthy"
        if overall_ok and any_degraded:
            status_str = "degraded"

        body = {
            "status":                    status_str,
            "version":                   _APP_VERSION,
            "checks":                    checks,
            "encryption_active":         getattr(
                request.app.state.audit_db, "encryption_active", False
            ),
            "pii_deidentifier_active":   request.app.state.deidentifier is not None,
        }
        http_status = 200 if overall_ok else 503
        return JSONResponse(content=body, status_code=http_status)

    return app


app = create_app()


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "hitl.main:app",
        host    = os.getenv("HOST", "0.0.0.0"),
        port    = int(os.getenv("PORT", "8000")),
        reload  = os.getenv("RELOAD", "false").lower() == "true",
        workers = 1,   # Single worker — OllamaClient Semaphore(1) requires single process
        log_config = None,  # Disable uvicorn's default log config; our JSONFormatter is active
    )
