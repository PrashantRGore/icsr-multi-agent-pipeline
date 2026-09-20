"""
agents/base_agent.py
=====================
Abstract base class shared by all PV agents.

Provides:
  - Standardized audit log entry writing (21 CFR Part 11)
  - Structured error handling → pipeline_halted + error_log update
  - review_id refresh on every agent call (per-stage traceability)
  - LangGraph node contract: run(state) -> dict

Design rules:
  1. All subclasses implement _run_inner(state) -> dict
  2. _run_inner must never mutate state directly — return a new partial dict
  3. All Ollama calls go through self._llm (OllamaClient instance)
  4. All schema validation errors are caught and converted to HITL routing

Audit contract:
  _write_audit() constructs a full AuditLogEntry Pydantic object and passes
  it to AuditDB.insert_entry(entry).  AuditLogEntry requires:
    trace_id, run_id, review_id, agent_id, prompt_version, status, content_hash.
  run_id defaults to trace_id when not present in state (graceful fallback).
"""
from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from graph.state import GraphState, refresh_review_id
from infra.audit_db import AuditDB
from infra.ollama_client import OllamaClient, OllamaError
from schemas.audit import AuditLogEntry, AuditStatus

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base for all ICSR processing agents.

    Parameters
    ----------
    llm       : OllamaClient — shared (serialized) inference client
    audit_db  : AuditDB      — 21 CFR Part 11 immutable audit log
    agent_id  : str          — unique agent identifier (e.g. "triage-agent-v1")
    """

    def __init__(
        self,
        llm:       OllamaClient,
        audit_db:  AuditDB,
        agent_id:  str,
    ) -> None:
        self._llm      = llm
        self._audit_db = audit_db
        self._agent_id = agent_id

    # ── LangGraph node entry point ────────────────────────────────────────────

    def run(self, state: GraphState) -> dict:
        """
        LangGraph node callable.  Called by LangGraph with the current state.
        Returns a PARTIAL state dict — LangGraph merges it with existing state.

        Behaviour:
          1. Refresh review_id (per-stage audit trail)
          2. Call _run_inner()
          3. On any exception: log to audit_db, set pipeline_halted=True, HITL route
        """
        state = refresh_review_id(state)

        try:
            partial = self._run_inner(state)
        except Exception as exc:
            return self._handle_failure(state, exc)

        # Always write an audit entry on success
        self._write_audit(
            state      = state,
            stage      = self._agent_id,
            status     = "SUCCESS",
            content    = str(partial),
            error_msg  = None,
        )
        return partial

    # ── Abstract ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _run_inner(self, state: GraphState) -> dict:
        """
        Subclass implements actual agent logic here.
        MUST return a partial GraphState dict.
        MUST NOT mutate state directly.
        """

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_system_prompt(self, template: str) -> str:
        """Return the full system prompt.  Subclasses provide template."""
        return template

    def _write_audit(
        self,
        state:     GraphState,
        stage:     str,
        status:    str,
        content:   str,
        error_msg: str | None,
    ) -> None:
        """Write one immutable audit entry to AuditDB."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        # Map status string to AuditStatus enum
        try:
            audit_status = AuditStatus(status.upper())
        except ValueError:
            audit_status = AuditStatus.FAILED

        # Construct the Pydantic AuditLogEntry object
        entry = AuditLogEntry(
            trace_id      = state.get("trace_id", "unknown"),
            run_id        = state.get("run_id")  or state.get("trace_id", "unknown"),
            review_id     = state.get("review_id", "unknown"),
            agent_id      = self._agent_id,
            prompt_version= stage,          # reuse stage name as prompt_version fallback
            status        = audit_status,
            content_hash  = content_hash,
            error_message = error_msg,
            extra_metadata= {"case_id": state.get("case_id", "unknown")},
        )

        try:
            self._audit_db.insert_entry(entry)
        except Exception as audit_exc:
            # Audit failure must NOT silently swallow the original error
            logger.error(
                "AuditDB write failed for case=%s stage=%s: %s",
                state.get("case_id", "unknown"), stage, audit_exc
            )

    def _handle_failure(self, state: GraphState, exc: Exception) -> dict:
        """
        Convert any unhandled exception into a pipeline halt + HITL route.
        Writes a FAILURE audit entry.
        """
        error_msg = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Agent %s FAILED for case=%s: %s",
            self._agent_id, state["case_id"], error_msg
        )

        self._write_audit(
            state     = state,
            stage     = self._agent_id,
            status    = "FAILURE",
            content   = error_msg,
            error_msg = error_msg,
        )

        new_errors = list(state.get("error_log", [])) + [
            f"[{self._agent_id}] {error_msg}"
        ]

        return {
            "pipeline_halted": True,
            "hitl_stage":      self._agent_id,
            "error_log":       new_errors,
        }

    def _parse_llm_json(self, raw_text: str, context: str = "") -> dict:
        """
        Parse LLM output as JSON.  On failure raises ValueError with context.
        The ValueError will be caught by run() → HITL route.
        """
        import json
        try:
            return json.loads(raw_text)
        except Exception as exc:
            raise ValueError(
                f"LLM output is not valid JSON [{context}]: {exc}\n"
                f"Raw output (first 500 chars): {raw_text[:500]}"
            ) from exc
