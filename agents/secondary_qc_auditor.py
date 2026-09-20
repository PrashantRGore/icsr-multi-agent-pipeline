"""
agents/secondary_qc_auditor.py
================================
Secondary QC Auditor — Learning Pipeline signal extractor.

Called synchronously by the HITL FastAPI route (hitl/routes/review.py) after
every accepted HITL correction. Analyses the diff between the original and
corrected values, maps each changed field to an ErrorClassification, and
writes one LearningSignal row to LearningDB per changed field.

Design rationale:
  - NO LLM calls (hardware constraint — 16 GB RAM, Semaphore(1)).
    Classification is entirely deterministic/rule-based.
  - NOT a LangGraph node. Does not inherit from BaseAgent.
    It is a standalone service class called imperatively.
  - Append-only: writes to LearningDB (immutable by trigger).
  - Writes one AuditLogEntry (status=SUCCESS) to AuditDB for traceability.
  - learning_signal=False on the HITLReviewRecord -> skips DB writes entirely.

Compliance:
  - CIOMS WG XIV Principle 6 — Fairness and Equity:
      demographic fields (patient_ethnicity, patient_sex, patient_age_group)
      are extracted from state_snap and stored with every signal for
      stratified bias analysis.
  - 21 CFR Part 11: every call produces an AuditLogEntry.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from infra.audit_db import AuditDB
from infra.learning_db import LearningDB
from schemas.audit import AuditLogEntry, AuditStatus, HITLReviewRecord
from schemas.qc import ErrorClassification

logger = logging.getLogger(__name__)

AGENT_ID       = "secondary-qc-auditor-v1"
PROMPT_VERSION = "secondary-qc-auditor-v1.0"


# -- Result dataclass ----------------------------------------------------------

@dataclass
class LearningSignalResult:
    """
    Summary of one SecondaryQCAuditor.audit_correction() call.

    Fields
    ------
    signals_written : int        -- number of LearningSignal rows inserted
    field_paths     : list[str]  -- e.g. ["extracted_entities.suspect_drugs"]
    error_classes   : list[str]  -- e.g. ["OMISSION"]
    skipped         : bool       -- True when learning_signal=False on record
    """
    signals_written: int       = 0
    field_paths:     list[str] = field(default_factory=list)
    error_classes:   list[str] = field(default_factory=list)
    skipped:         bool      = False


# -- Field to ErrorClassification mapping -------------------------------------

# Top-level corrected field keys -> primary ErrorClassification.
# Sub-field refinement is applied inside _classify_field().
_TOP_LEVEL_CLASSIFICATION: dict[str, ErrorClassification] = {
    "triage_output":             ErrorClassification.CONFIDENCE_INSUFFICIENT,
    "extracted_entities":        ErrorClassification.OMISSION,
    "causality_matrix":          ErrorClassification.CAUSALITY_UNSUPPORTED,
    "coded_events":              ErrorClassification.CODING_MISMATCH,
    "listedness_evaluations":    ErrorClassification.LISTEDNESS_UNCERTAIN,
    "qc_report":                 ErrorClassification.CONSISTENCY_ERROR,
    "final_narrative":           ErrorClassification.CONSISTENCY_ERROR,
    "next_stage":                ErrorClassification.SCHEMA_VIOLATION,
}

# Sub-field refinements for extracted_entities corrections
_EXTRACTION_SUB_FIELD_CLASSIFICATION: dict[str, ErrorClassification] = {
    "suspect_drugs":          ErrorClassification.OMISSION,
    "concomitant_drugs":      ErrorClassification.OMISSION,
    "adverse_events":         ErrorClassification.OMISSION,
    "onset_date":             ErrorClassification.DATE_ORDER_VIOLATION,
    "resolution_date":        ErrorClassification.DATE_ORDER_VIOLATION,
    "start_date":             ErrorClassification.DATE_ORDER_VIOLATION,
    "end_date":               ErrorClassification.DATE_ORDER_VIOLATION,
    "dechallenge":            ErrorClassification.POLARITY_FLIP,
    "rechallenge":            ErrorClassification.POLARITY_FLIP,
    "serious":                ErrorClassification.POLARITY_FLIP,
}


def _classify_field(
    field_key: str,
    original_value: Any,
    corrected_value: Any,
) -> tuple[str, ErrorClassification]:
    """
    Map one corrected field to an (annotated_field_path, ErrorClassification).

    Applies sub-field refinements for extracted_entities.  All other
    top-level keys use the static mapping; unknown keys fall back to
    SCHEMA_VIOLATION.

    Returns
    -------
    (field_path, ErrorClassification)
    """
    if field_key == "extracted_entities" and isinstance(corrected_value, dict):
        orig_dict = original_value if isinstance(original_value, dict) else {}
        changed_sub: list[str] = [
            k for k in corrected_value
            if corrected_value.get(k) != orig_dict.get(k)
        ]
        if changed_sub:
            sub = changed_sub[0]
            classification = _EXTRACTION_SUB_FIELD_CLASSIFICATION.get(
                sub, ErrorClassification.OMISSION
            )
            return f"{field_key}.{sub}", classification

    classification = _TOP_LEVEL_CLASSIFICATION.get(
        field_key, ErrorClassification.SCHEMA_VIOLATION
    )
    return field_key, classification


# -- SecondaryQCAuditor --------------------------------------------------------

class SecondaryQCAuditor:
    """
    Deterministic, LLM-free correction classifier for the Learning Pipeline.

    Parameters
    ----------
    audit_db    : AuditDB     -- for writing the regulatory traceability entry
    learning_db : LearningDB  -- for writing learning signals (append-only)
    """

    def __init__(self, audit_db: AuditDB, learning_db: LearningDB) -> None:
        self._audit_db    = audit_db
        self._learning_db = learning_db

    def audit_correction(
        self,
        review_record: HITLReviewRecord,
        state_snap:    dict[str, Any],
    ) -> LearningSignalResult:
        """
        Classify a HITL correction and write learning signals to LearningDB.

        Parameters
        ----------
        review_record : HITLReviewRecord -- the validated, immutable correction record
        state_snap    : dict             -- full pipeline state at the time of the halt
                        (used to extract demographic fields for bias monitoring)

        Returns
        -------
        LearningSignalResult -- summary of signals written
        """
        if not review_record.learning_signal:
            logger.info(
                "%s: learning_signal=False -- skipping DB writes for review_id=%s",
                AGENT_ID, review_record.review_id,
            )
            self._write_audit(review_record, 0, skipped=True)
            return LearningSignalResult(skipped=True)

        # -- Determine which fields were corrected ---------------------------
        corrected: dict[str, Any] = (
            review_record.corrected_value
            if isinstance(review_record.corrected_value, dict)
            else {}
        )
        original: dict[str, Any] = (
            review_record.original_value
            if isinstance(review_record.original_value, dict)
            else {}
        )

        # If corrected is empty -- treat the stage output as corrected wholesale
        if not corrected:
            corrected = {review_record.stage.value.lower(): review_record.corrected_value}
            original  = {review_record.stage.value.lower(): review_record.original_value}

        # -- Extract demographic context from state snapshot -----------------
        extracted = state_snap.get("extracted_entities") or {}
        if isinstance(extracted, dict):
            patient_ethnicity = extracted.get("patient_ethnicity")
            patient_sex       = (extracted.get("patient") or {}).get("sex")
            patient_age_group = _derive_age_group(extracted)
        else:
            patient_ethnicity = patient_sex = patient_age_group = None

        # -- Build and write one signal per changed field --------------------
        result = LearningSignalResult()
        reviewed_at = review_record.reviewed_at.isoformat()

        for field_key, corrected_val in corrected.items():
            original_val = original.get(field_key)

            # Skip if nothing actually changed
            if _values_equal(original_val, corrected_val):
                continue

            field_path, error_class = _classify_field(
                field_key, original_val, corrected_val
            )

            try:
                self._learning_db.insert_signal(
                    trace_id             = review_record.trace_id,
                    review_id            = review_record.review_id,
                    stage                = review_record.stage.value,
                    agent_id             = AGENT_ID,
                    prompt_version       = PROMPT_VERSION,
                    error_classification = error_class.value,
                    field_path           = field_path,
                    original_value       = original_val,
                    corrected_value      = corrected_val,
                    correction_rationale = review_record.correction_rationale,
                    reviewer_id          = review_record.reviewer_id,
                    reviewed_at          = reviewed_at,
                    patient_ethnicity    = patient_ethnicity,
                    patient_sex          = patient_sex,
                    patient_age_group    = patient_age_group,
                )
                result.signals_written += 1
                result.field_paths.append(field_path)
                result.error_classes.append(error_class.value)
                logger.debug(
                    "%s: signal written  field=%s  class=%s  trace=%s",
                    AGENT_ID, field_path, error_class.value, review_record.trace_id,
                )
            except Exception as exc:
                logger.error(
                    "%s: failed to insert signal for field=%s: %s",
                    AGENT_ID, field_path, exc,
                )

        self._write_audit(review_record, result.signals_written, skipped=False)

        logger.info(
            "%s: audit_correction complete  review_id=%s  signals=%d  fields=%s",
            AGENT_ID, review_record.review_id,
            result.signals_written, result.field_paths,
        )
        return result

    # -- Private helpers -------------------------------------------------------

    def _write_audit(
        self,
        review_record: HITLReviewRecord,
        signals_written: int,
        *,
        skipped: bool,
    ) -> None:
        """Write a regulatory traceability AuditLogEntry for this auditor call."""
        content = json.dumps({
            "review_id":       review_record.review_id,
            "signals_written": signals_written,
            "skipped":         skipped,
        }, sort_keys=True)
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        entry = AuditLogEntry(
            trace_id       = review_record.trace_id,
            run_id         = review_record.review_id,
            review_id      = review_record.review_id,
            agent_id       = AGENT_ID,
            prompt_version = PROMPT_VERSION,
            status         = AuditStatus.SUCCESS,
            content_hash   = content_hash,
            extra_metadata = {
                "signals_written": signals_written,
                "skipped":         skipped,
                "reviewer_id":     review_record.reviewer_id,
                "stage":           review_record.stage.value,
            },
        )
        try:
            self._audit_db.insert_entry(entry)
        except Exception as exc:
            logger.error(
                "%s: AuditDB write failed for review_id=%s: %s",
                AGENT_ID, review_record.review_id, exc,
            )


# -- Utility helpers -----------------------------------------------------------

def _values_equal(a: Any, b: Any) -> bool:
    """
    Compare two values for equality, handling JSON-serialised strings.
    The HITLReviewRecord stores original_value/corrected_value as Any,
    but they may arrive as JSON strings in some code paths.
    """
    if a == b:
        return True
    try:
        a_obj = json.loads(a) if isinstance(a, str) else a
        b_obj = json.loads(b) if isinstance(b, str) else b
        return a_obj == b_obj
    except (TypeError, ValueError):
        return False


def _derive_age_group(extracted: dict) -> Optional[str]:
    """
    Derive PEDIATRIC / ADULT / ELDERLY from patient age in extracted entities.
    Returns None if age is not available.
    """
    patient = extracted.get("patient") or {}
    age = patient.get("age_at_onset")
    if age is None:
        return None
    try:
        age_val = float(age)
    except (TypeError, ValueError):
        return None
    if age_val < 18:
        return "PEDIATRIC"
    if age_val >= 65:
        return "ELDERLY"
    return "ADULT"
