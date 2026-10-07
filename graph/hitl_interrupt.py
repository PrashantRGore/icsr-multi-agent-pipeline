"""
graph/hitl_interrupt.py
========================
HITL interrupt / resume utilities for the LangGraph pipeline.

LangGraph v0.2+ supports graph interruption via langgraph.graph.interrupt().
When pipeline_halted=True, the pipeline is suspended at the hitl_node and
persisted to the SQLiteSaver checkpoint database.

Resume protocol:
  1. Human reviewer submits correction via POST /api/v1/review/{review_id}/submit
  2. HITLServer calls `resume_pipeline(thread_id, correction)` here
  3. This function merges the correction into the checkpoint state,
     clears pipeline_halted=False, and re-invokes the compiled graph
  4. LangGraph replays from the checkpoint with the corrected state

Checkpoint database:
  - langgraph-checkpoint-sqlite: writes to checkpoints/pipeline.db
  - One DB is shared across all cases (different thread_ids)
  - 21 CFR Part 11: checkpoints are append-only by design (SQLite WAL)

HITL correction payload (hitl_correction dict):
  The correction is keyed by the field that was wrong. The re-entry node
  reads `state["hitl_correction"]` and applies the correction before
  re-running its logic.

  Example for extraction stage:
    {"extracted_entities": {...pydantic model dict...}}

  Example for causality stage:
    {"causality_matrix": {...}}

  The correction payload is validated by the re-entry agent before use.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger(__name__)

# Path for the LangGraph checkpoint SQLite DB
CHECKPOINT_DB_PATH = Path("checkpoints/pipeline.db")


def get_checkpointer(db_path: Path = CHECKPOINT_DB_PATH) -> SqliteSaver:
    """
    Return a LangGraph SqliteSaver checkpointer.
    SqliteSaver persists every GraphState snapshot to disk so HITL queue
    entries survive server restarts.

    Database: checkpoints/pipeline.db (SQLite WAL mode, shared across all cases
    via different thread_ids).

    Implementation note: langgraph-checkpoint-sqlite ≥ 3.1 changed
    SqliteSaver.from_conn_string() to return a context manager, which
    newer LangGraph rejects. We pass an open sqlite3.Connection directly.
    check_same_thread=False is safe here because OllamaClient uses
    Semaphore(1), ensuring only one pipeline thread runs at a time.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    logger.info("Checkpointer (SqliteSaver) initialised at: %s", db_path)
    return checkpointer


def build_resume_state(
    current_state: dict[str, Any],
    correction:    dict[str, Any],
) -> dict[str, Any]:
    """
    Merge a HITL correction into the current checkpoint state.

    Rules:
    1. pipeline_halted is always reset to False (allows re-entry)
    2. Correction keys overwrite the corresponding state fields
    3. hitl_correction stores the raw correction dict for audit trail
    4. review_id is refreshed (new stage-level UUID) to separate audit entries

    Parameters
    ----------
    current_state : The full GraphState dict from the checkpoint
    correction    : Partial dict of corrected field-name → value pairs

    Returns
    -------
    Updated GraphState dict ready for graph re-invocation
    """
    import uuid
    from datetime import datetime, timezone

    merged = {
        **current_state,
        **correction,
        "pipeline_halted":  False,       # Clear the halt flag
        "hitl_correction":  correction,  # Preserve for audit
        "review_id":        str(uuid.uuid4()),  # Fresh per-stage UUID
    }

    logger.info(
        "HITL resume: case=%s  stage=%s  new_review_id=%s  correction_keys=%s",
        current_state.get("case_id", "unknown"),
        current_state.get("hitl_stage", "unknown"),
        merged["review_id"],
        list(correction.keys()),
    )

    return merged


class HITLQueue:
    """
    HITL queue: maps review_id → (thread_id, halted_state_snapshot).

    Backed by an optional AuditDB for persistence across server restarts.
    When audit_db is provided, every enqueue/dequeue is mirrored to the
    hitl_queue SQLite table so pending cases survive process restarts.

    Thread safety: not required — OllamaClient serializes inference
    via Semaphore(1), so only one pipeline thread runs at a time.
    """

    def __init__(self, audit_db=None) -> None:
        self._queue: dict[str, dict[str, Any]] = {}
        self._db = audit_db  # Optional AuditDB write-through

    def enqueue(
        self,
        review_id:  str,
        thread_id:  str,
        state_snap: dict[str, Any],
    ) -> None:
        """Add a halted case to the queue, persisting to AuditDB if available."""
        enqueued_at = _utcnow()
        self._queue[review_id] = {
            "thread_id":   thread_id,
            "state_snap":  state_snap,
            "enqueued_at": enqueued_at,
        }
        # Write-through to persistent store
        if self._db is not None:
            try:
                self._db.enqueue_hitl(
                    review_id  = review_id,
                    thread_id  = thread_id,
                    case_id    = state_snap.get("case_id", "unknown"),
                    hitl_stage = str(state_snap.get("hitl_stage", "UNKNOWN")),
                    state_snap = state_snap,
                )
            except Exception as exc:
                logger.warning("HITLQueue: AuditDB write-through failed: %s", exc)
        logger.info("HITLQueue: enqueued review_id=%s thread_id=%s", review_id, thread_id)

    def get(self, review_id: str) -> dict[str, Any] | None:
        """Return the queue entry for review_id, or None if not found."""
        return self._queue.get(review_id)

    def dequeue(self, review_id: str, final_status: str = "DEQUEUED") -> dict[str, Any] | None:
        """
        Pop and return the queue entry. Returns None if not found.

        Parameters
        ----------
        final_status : DEQUEUED (default), RESUMED, or REJECTED
        """
        entry = self._queue.pop(review_id, None)
        if entry:
            logger.info("HITLQueue: dequeued review_id=%s status=%s", review_id, final_status)
            if self._db is not None:
                try:
                    self._db.update_hitl_status(review_id, final_status)
                except Exception as exc:
                    logger.warning("HITLQueue: AuditDB status update failed: %s", exc)
        return entry

    def list_pending(self) -> list[dict[str, Any]]:
        """Return all pending HITL entries (summary view, no full state)."""
        return [
            {
                "review_id":    rid,
                "thread_id":    v["thread_id"],
                "case_id":      v["state_snap"].get("case_id"),
                "hitl_stage":   str(v["state_snap"].get("hitl_stage", "")),
                "enqueued_at":  v["enqueued_at"],
                "error_log":    v["state_snap"].get("error_log", [])[-3:],
            }
            for rid, v in self._queue.items()
        ]

    def __len__(self) -> int:
        return len(self._queue)


def _utcnow() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
