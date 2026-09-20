"""
tests/unit/test_triage_schema.py
=================================
Unit tests for schemas/triage.py

Tests:
  1. Valid TIER_1 case (life-threatening, HITL mandatory)
  2. Valid TIER_2 case (hospitalization, HITL conditional)
  3. Valid TIER_3 case (non-serious, HOTL)
  4. INVALID status with failure_reasons
  5. Validator: VALID status requires all 4 criteria met
  6. Validator: INVALID status requires at least 1 failure_reason
  7. Validator: TIER_1 + HOTL raises ValueError
  8. Validator: VALID + not all criteria = ValueError
  9. effective_threshold returns correct value per tier
  10. needs_hitl logic
"""
import pytest
from pydantic import ValidationError
from schemas.triage import (
    MinimumCriteria,
    OversightMode,
    RiskTier,
    TriageOutput,
    TriageStatus,
    THRESHOLD_BY_TIER,
)


def _all_criteria() -> MinimumCriteria:
    return MinimumCriteria(
        has_identifiable_patient=True,
        has_identifiable_reporter=True,
        has_suspect_drug=True,
        has_adverse_event=True,
    )


def _missing_reporter() -> MinimumCriteria:
    return MinimumCriteria(
        has_identifiable_patient=True,
        has_identifiable_reporter=False,
        has_suspect_drug=True,
        has_adverse_event=True,
    )


def _make_valid_triage(
    risk_tier: RiskTier = RiskTier.TIER_2,
    oversight_mode: OversightMode = OversightMode.HITL,
    seriousness_signals: list[str] | None = None,
    confidence: float = 0.92,
) -> TriageOutput:
    return TriageOutput(
        case_id="ICSR-20240315-SYN",
        narrative_hash="a" * 64,
        criteria=_all_criteria(),
        status=TriageStatus.VALID,
        risk_tier=risk_tier,
        oversight_mode=oversight_mode,
        seriousness_signals=seriousness_signals or ["hospitalization"],
        confidence=confidence,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Valid TIER_1
# ─────────────────────────────────────────────────────────────────────────────

def test_tier1_life_threatening_valid():
    t = _make_valid_triage(
        risk_tier=RiskTier.TIER_1,
        oversight_mode=OversightMode.HITL,
        seriousness_signals=["life-threatening"],
        confidence=0.97,
    )
    assert t.risk_tier == RiskTier.TIER_1
    assert t.oversight_mode == OversightMode.HITL
    assert t.effective_threshold == 0.95


# ─────────────────────────────────────────────────────────────────────────────
# 2. Valid TIER_2
# ─────────────────────────────────────────────────────────────────────────────

def test_tier2_valid():
    t = _make_valid_triage(risk_tier=RiskTier.TIER_2, confidence=0.88)
    assert t.risk_tier == RiskTier.TIER_2
    assert t.effective_threshold == 0.85


# ─────────────────────────────────────────────────────────────────────────────
# 3. Valid TIER_3 with HOTL
# ─────────────────────────────────────────────────────────────────────────────

def test_tier3_hotl_valid():
    t = _make_valid_triage(
        risk_tier=RiskTier.TIER_3,
        oversight_mode=OversightMode.HOTL,
        seriousness_signals=[],
        confidence=0.80,
    )
    assert t.risk_tier == RiskTier.TIER_3
    assert t.oversight_mode == OversightMode.HOTL
    assert t.effective_threshold == 0.75
    assert not t.needs_hitl


# ─────────────────────────────────────────────────────────────────────────────
# 4. INVALID status with failure_reasons
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_case_with_failure_reasons():
    t = TriageOutput(
        case_id="ICSR-20240402-SYN",
        narrative_hash="b" * 64,
        criteria=_missing_reporter(),
        status=TriageStatus.INVALID,
        risk_tier=RiskTier.TIER_3,
        oversight_mode=OversightMode.HITL,
        seriousness_signals=[],
        confidence=0.60,
        failure_reasons=["has_identifiable_reporter is False — no reporter name or qualification found"],
    )
    assert t.status == TriageStatus.INVALID
    assert len(t.failure_reasons) == 1
    assert t.needs_hitl  # INVALID always needs HITL


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validator: VALID with all criteria = OK
# ─────────────────────────────────────────────────────────────────────────────

def test_all_criteria_met_allows_valid_status():
    t = _make_valid_triage()
    assert t.status == TriageStatus.VALID
    assert t.criteria.all_met is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validator: INVALID requires failure_reasons
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_without_failure_reasons_raises():
    with pytest.raises(ValidationError, match="failure_reasons"):
        TriageOutput(
            case_id="ICSR-20240402-SYN",
            narrative_hash="c" * 64,
            criteria=_missing_reporter(),
            status=TriageStatus.INVALID,
            risk_tier=RiskTier.TIER_3,
            oversight_mode=OversightMode.HITL,
            seriousness_signals=[],
            confidence=0.50,
            failure_reasons=[],   # Empty — should fail
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. Validator: TIER_1 + HOTL must raise
# ─────────────────────────────────────────────────────────────────────────────

def test_tier1_with_hotl_raises():
    with pytest.raises(ValidationError, match="TIER_1"):
        _make_valid_triage(
            risk_tier=RiskTier.TIER_1,
            oversight_mode=OversightMode.HOTL,  # Invalid: TIER_1 must be HITL
            seriousness_signals=["death"],
            confidence=0.97,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. Validator: VALID status but criteria not all met must raise
# ─────────────────────────────────────────────────────────────────────────────

def test_valid_status_with_missing_criteria_raises():
    with pytest.raises(ValidationError, match="minimum criteria"):
        TriageOutput(
            case_id="ICSR-20240402-SYN",
            narrative_hash="d" * 64,
            criteria=_missing_reporter(),
            status=TriageStatus.VALID,   # Invalid: reporter is missing
            risk_tier=RiskTier.TIER_2,
            oversight_mode=OversightMode.HITL,
            seriousness_signals=["hospitalization"],
            confidence=0.87,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 9. effective_threshold matches THRESHOLD_BY_TIER constant
# ─────────────────────────────────────────────────────────────────────────────

def test_effective_threshold_matches_constant():
    for tier, expected in THRESHOLD_BY_TIER.items():
        oversight = OversightMode.HITL
        if tier == RiskTier.TIER_3:
            oversight = OversightMode.HOTL
        t = _make_valid_triage(risk_tier=tier, oversight_mode=oversight, confidence=1.0)
        assert t.effective_threshold == expected, f"Mismatch for {tier}"


# ─────────────────────────────────────────────────────────────────────────────
# 10. needs_hitl logic
# ─────────────────────────────────────────────────────────────────────────────

def test_needs_hitl_when_confidence_below_threshold():
    # TIER_2 threshold is 0.85; confidence=0.80 → needs HITL
    t = _make_valid_triage(risk_tier=RiskTier.TIER_2, confidence=0.80)
    assert t.needs_hitl is True


def test_needs_hitl_false_when_above_threshold():
    # TIER_3 threshold is 0.75; confidence=0.90 → does NOT need HITL
    t = _make_valid_triage(
        risk_tier=RiskTier.TIER_3,
        oversight_mode=OversightMode.HOTL,
        seriousness_signals=[],
        confidence=0.90,
    )
    assert t.needs_hitl is False
