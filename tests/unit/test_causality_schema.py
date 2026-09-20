"""
tests/unit/test_causality_schema.py
=====================================
Unit tests for schemas/causality.py

Tests:
  1.  CausalityAssessment: RELATED term, no alternative etiologies (valid)
  2.  CausalityAssessment: UNLIKELY_RELATED without alternatives raises
  3.  CausalityAssessment: NOT_RELATED without alternatives raises
  4.  CausalityAssessment: POSSIBLY_RELATED without alternatives = valid
  5.  AlternativeEtiology: valid construction
  6.  CausalityMatrix: single pair, valid
  7.  CausalityMatrix: duplicate (drug_node_id, event_node_id) pair raises
  8.  CausalityMatrix: empty assessments raises (min_length=1)
  9.  CausalityMatrix: multiple distinct pairs valid
  10. hard_negative_checked field defaults to False
"""
import pytest
from pydantic import ValidationError

from schemas.causality import (
    AlternativeEtiology,
    CausalityAssessment,
    CausalityMatrix,
    CausalityTerm,
)


def _assessment(
    drug_node_id="DRG-00000001",
    event_node_id="AE-00000001",
    term=CausalityTerm.RELATED,
    alternatives=None,
    hard_negative_checked=False,
) -> CausalityAssessment:
    return CausalityAssessment(
        drug_node_id=drug_node_id,
        event_node_id=event_node_id,
        drug_name="Amoxicillin",
        verbatim_event="Anaphylaxis",
        causality_term=term,
        rationale="Temporal sequence plausible; known penicillin allergy mechanism.",
        confidence=0.92,
        alternative_etiologies=alternatives or [],
        hard_negative_checked=hard_negative_checked,
    )


def _alternative() -> AlternativeEtiology:
    return AlternativeEtiology(
        description="Pre-existing latex allergy with cross-reactivity",
        likelihood="Low",
        source_reference="Narrative paragraph 2",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. RELATED — no alternatives required
# ─────────────────────────────────────────────────────────────────────────────

def test_related_no_alternatives_valid():
    a = _assessment(term=CausalityTerm.RELATED)
    assert a.causality_term == CausalityTerm.RELATED
    assert a.alternative_etiologies == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. UNLIKELY_RELATED without alternatives raises
# ─────────────────────────────────────────────────────────────────────────────

def test_unlikely_related_without_alternatives_raises():
    with pytest.raises(ValidationError, match="alternative_etiology"):
        _assessment(term=CausalityTerm.UNLIKELY_RELATED, alternatives=[])


# ─────────────────────────────────────────────────────────────────────────────
# 3. NOT_RELATED without alternatives raises
# ─────────────────────────────────────────────────────────────────────────────

def test_not_related_without_alternatives_raises():
    with pytest.raises(ValidationError, match="alternative_etiology"):
        _assessment(term=CausalityTerm.NOT_RELATED, alternatives=[])


# ─────────────────────────────────────────────────────────────────────────────
# 4. POSSIBLY_RELATED without alternatives = valid
# ─────────────────────────────────────────────────────────────────────────────

def test_possibly_related_without_alternatives_valid():
    a = _assessment(term=CausalityTerm.POSSIBLY_RELATED, alternatives=[])
    assert a.causality_term == CausalityTerm.POSSIBLY_RELATED


# ─────────────────────────────────────────────────────────────────────────────
# 5. AlternativeEtiology valid
# ─────────────────────────────────────────────────────────────────────────────

def test_alternative_etiology_valid():
    alt = _alternative()
    assert alt.likelihood == "Low"


# ─────────────────────────────────────────────────────────────────────────────
# 6. CausalityMatrix single pair valid
# ─────────────────────────────────────────────────────────────────────────────

def test_causality_matrix_single_pair():
    m = CausalityMatrix(
        case_id="ICSR-TEST-001",
        assessments=[_assessment()],
    )
    assert len(m.assessments) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. CausalityMatrix duplicate pair raises
# ─────────────────────────────────────────────────────────────────────────────

def test_causality_matrix_duplicate_pair_raises():
    with pytest.raises(ValidationError, match="Duplicate"):
        CausalityMatrix(
            case_id="ICSR-TEST-001",
            assessments=[
                _assessment(drug_node_id="DRG-0001", event_node_id="AE-0001"),
                _assessment(drug_node_id="DRG-0001", event_node_id="AE-0001"),  # Duplicate
            ],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. CausalityMatrix empty raises
# ─────────────────────────────────────────────────────────────────────────────

def test_causality_matrix_empty_raises():
    with pytest.raises(ValidationError):
        CausalityMatrix(case_id="ICSR-TEST-001", assessments=[])


# ─────────────────────────────────────────────────────────────────────────────
# 9. CausalityMatrix multiple distinct pairs valid
# ─────────────────────────────────────────────────────────────────────────────

def test_causality_matrix_multiple_pairs():
    m = CausalityMatrix(
        case_id="ICSR-TEST-001",
        assessments=[
            _assessment(drug_node_id="DRG-0001", event_node_id="AE-0001"),
            _assessment(drug_node_id="DRG-0001", event_node_id="AE-0002"),
            _assessment(
                drug_node_id="DRG-0002", event_node_id="AE-0001",
                term=CausalityTerm.UNLIKELY_RELATED,
                alternatives=[_alternative()],
            ),
        ],
    )
    assert len(m.assessments) == 3


# ─────────────────────────────────────────────────────────────────────────────
# 10. hard_negative_checked defaults to False
# ─────────────────────────────────────────────────────────────────────────────

def test_hard_negative_checked_defaults_false():
    a = _assessment()
    assert a.hard_negative_checked is False


def test_hard_negative_checked_can_be_set_true():
    a = _assessment(hard_negative_checked=True)
    assert a.hard_negative_checked is True
