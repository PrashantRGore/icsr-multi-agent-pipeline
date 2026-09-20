"""
tests/unit/test_qc_schema.py
=============================
Unit tests for schemas/qc.py

Tests:
  1.  QCReport approved=True, no blockers (valid)
  2.  QCReport approved=False with BLOCKER item (valid)
  3.  QCReport approved=True with BLOCKER item raises
  4.  QCReport approved=False with no items raises
  5.  QCReport approved=False with WARNING only raises (no blocker documented)
  6.  ErrorClassification: all 10 values are valid enum members
  7.  CritiqueItem: node_id and field_path optional
  8.  CritiqueItem: suggested_correction optional
  9.  HardNegativeViolation: creates correctly
  10. QCReport: hard_negative_violations forces approved=False
"""
import pytest
from pydantic import ValidationError

from schemas.qc import (
    CritiqueItem,
    CritiqueSeverity,
    ErrorClassification,
    HardNegativeViolation,
    QCReport,
)


def _blocker(node_id="DRG-00000001") -> CritiqueItem:
    return CritiqueItem(
        error_classification=ErrorClassification.CAUSALITY_UNSUPPORTED,
        severity=CritiqueSeverity.BLOCKER,
        affected_node_id=node_id,
        field_path="G.k.9.i",
        message="Drug-AE pair blocked by hard negative ontology.",
        suggested_correction="Set causality_term=NOT_RELATED; document alternative etiology.",
    )


def _warning() -> CritiqueItem:
    return CritiqueItem(
        error_classification=ErrorClassification.CONFIDENCE_INSUFFICIENT,
        severity=CritiqueSeverity.WARNING,
        message="Confidence 0.83 is below TIER_2 threshold of 0.85.",
    )


def _info() -> CritiqueItem:
    return CritiqueItem(
        error_classification=ErrorClassification.CONSISTENCY_ERROR,
        severity=CritiqueSeverity.INFO,
        message="Reporter type not specified; defaulting to Unknown.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. approved=True, no blockers
# ─────────────────────────────────────────────────────────────────────────────

def test_approved_with_warnings_only():
    report = QCReport(
        case_id="ICSR-TEST-001",
        approved=True,
        critique_items=[_warning(), _info()],
        qc_confidence=0.89,
    )
    assert report.approved is True
    assert len(report.critique_items) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. approved=False with BLOCKER
# ─────────────────────────────────────────────────────────────────────────────

def test_rejected_with_blocker():
    report = QCReport(
        case_id="ICSR-TEST-001",
        approved=False,
        critique_items=[_blocker()],
        qc_confidence=0.72,
    )
    assert report.approved is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. approved=True with BLOCKER raises
# ─────────────────────────────────────────────────────────────────────────────

def test_approved_true_with_blocker_raises():
    with pytest.raises(ValidationError, match="BLOCKER"):
        QCReport(
            case_id="ICSR-TEST-001",
            approved=True,
            critique_items=[_blocker()],
            qc_confidence=0.90,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. approved=False with no items raises
# ─────────────────────────────────────────────────────────────────────────────

def test_rejected_without_any_items_raises():
    with pytest.raises(ValidationError, match="BLOCKER"):
        QCReport(
            case_id="ICSR-TEST-001",
            approved=False,
            critique_items=[],
            qc_confidence=0.60,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. approved=False with WARNING only (no BLOCKER) raises
# ─────────────────────────────────────────────────────────────────────────────

def test_rejected_with_warning_only_raises():
    """approved=False requires a BLOCKER. WARNING alone is insufficient."""
    with pytest.raises(ValidationError, match="BLOCKER"):
        QCReport(
            case_id="ICSR-TEST-001",
            approved=False,
            critique_items=[_warning()],  # Only WARNING — no BLOCKER
            qc_confidence=0.80,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 6. All ErrorClassification values are valid
# ─────────────────────────────────────────────────────────────────────────────

def test_all_error_classifications_valid():
    expected = {
        "HALLUCINATION", "POLARITY_FLIP", "OMISSION", "DATE_ORDER_VIOLATION",
        "CAUSALITY_UNSUPPORTED", "SCHEMA_VIOLATION", "CONFIDENCE_INSUFFICIENT",
        "CONSISTENCY_ERROR", "CODING_MISMATCH", "LISTEDNESS_UNCERTAIN",
    }
    actual = {e.value for e in ErrorClassification}
    assert actual == expected


# ─────────────────────────────────────────────────────────────────────────────
# 7. CritiqueItem: node_id and field_path are optional
# ─────────────────────────────────────────────────────────────────────────────

def test_critique_item_optional_fields():
    item = CritiqueItem(
        error_classification=ErrorClassification.HALLUCINATION,
        severity=CritiqueSeverity.BLOCKER,
        message="Hallucinated drug name 'Placebix' not present in narrative.",
    )
    assert item.affected_node_id is None
    assert item.field_path is None


# ─────────────────────────────────────────────────────────────────────────────
# 8. CritiqueItem: suggested_correction optional
# ─────────────────────────────────────────────────────────────────────────────

def test_critique_item_no_suggested_correction():
    item = CritiqueItem(
        error_classification=ErrorClassification.OMISSION,
        severity=CritiqueSeverity.WARNING,
        message="Concomitant drug amlodipine not extracted.",
    )
    assert item.suggested_correction is None


# ─────────────────────────────────────────────────────────────────────────────
# 9. HardNegativeViolation valid construction
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_negative_violation_valid():
    v = HardNegativeViolation(
        drug_node_id="DRG-00000001",
        drug_name="Propranolol",
        event_node_id="AE-00000001",
        event_term="Tachycardia",
        edge_type="DOES_NOT_TREAT",
        ontology_source="beta-blocker / tachycardia rule",
    )
    assert v.edge_type == "DOES_NOT_TREAT"


# ─────────────────────────────────────────────────────────────────────────────
# 10. hard_negative_violations forces approved=False
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_negative_violation_forces_rejection():
    violation = HardNegativeViolation(
        drug_node_id="DRG-00000001",
        drug_name="Propranolol",
        event_node_id="AE-00000001",
        event_term="Tachycardia",
        edge_type="DOES_NOT_TREAT",
    )
    # approved=True with violations must raise
    with pytest.raises(ValidationError, match="BLOCKER"):
        QCReport(
            case_id="ICSR-TEST-001",
            approved=True,              # Should fail — violation forces BLOCKER
            critique_items=[],
            hard_negative_violations=[violation],
            qc_confidence=0.88,
        )
