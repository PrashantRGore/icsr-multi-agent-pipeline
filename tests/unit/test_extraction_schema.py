"""
tests/unit/test_extraction_schema.py
=====================================
Unit tests for schemas/extraction.py

Tests:
  1.  PartialDate: year-only, year+month, full date (valid)
  2.  PartialDate: day without month raises
  3.  PartialDate: future year raises
  4.  PartialDate.__le__: year-level comparison
  5.  PartialDate.__le__: partial-date leniency (same year-month, one side missing day)
  6.  PartialDate.__le__: full date comparison
  7.  SuspectDrug: stop_date before start_date raises
  8.  VerbatimEvent: serious=True, no criteria raises
  9.  VerbatimEvent: serious=False, no criteria = valid (non-serious)
  10. VerbatimEvent: HOSPITALIZATION without HospitalizationDetails raises
  11. VerbatimEvent: HospitalizationDetails without criterion raises
  12. HospitalizationDetails: discharge before admission raises
  13. ExtractedCaseEntities: drug start after event onset raises
  14. ExtractedCaseEntities: partial-date leniency in cross-model check
  15. ExtractedCaseEntities: no SUSPECT drug raises
  16. node_id: auto-generated, unique per entity
  17. SuspectDrug: CONCOMITANT role is allowed (but SUSPECT must also exist)
  18. Rechallenge enum: four states all valid
"""
import pytest
from pydantic import ValidationError

from schemas.extraction import (
    DrugRole,
    Dechallenge,
    ExtractedCaseEntities,
    HospitalizationCausalityFlag,
    HospitalizationDetails,
    PartialDate,
    Rechallenge,
    SeriousnessCriterion,
    SuspectDrug,
    VerbatimEvent,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pd(year, month=None, day=None):
    return PartialDate(year=year, month=month, day=day)


def suspect_drug(name="Amoxicillin", start=None, stop=None, role=DrugRole.SUSPECT):
    return SuspectDrug(drug_name=name, start_date=start, stop_date=stop, drug_role=role)


def event(verbatim="Anaphylaxis", serious=True, onset=None,
          criteria=None, hosp_details=None):
    criteria = criteria or ([SeriousnessCriterion.LIFE_THREATENING] if serious else [])
    return VerbatimEvent(
        verbatim_term=verbatim,
        serious=serious,
        onset_date=onset,
        seriousness_criteria=criteria,
        hospitalization_details=hosp_details,
    )


def minimal_entities(**overrides):
    base = dict(
        case_id="ICSR-20240315-SYN",
        narrative_hash="a" * 64,
        suspect_drugs=[suspect_drug(start=pd(2024, 1, 10))],
        verbatim_events=[event(onset=pd(2024, 1, 15))],
        extraction_confidence=0.91,
    )
    base.update(overrides)
    return ExtractedCaseEntities(**base)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PartialDate valid variants
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_date_year_only():
    p = pd(2023)
    assert p.year == 2023 and p.month is None and p.day is None


def test_partial_date_year_month():
    p = pd(2023, 6)
    assert p.month == 6 and p.day is None


def test_partial_date_full():
    p = pd(2023, 6, 15)
    assert p.day == 15


# ─────────────────────────────────────────────────────────────────────────────
# 2. PartialDate: day without month raises
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_date_day_without_month_raises():
    with pytest.raises(ValidationError, match="day cannot be specified without month"):
        PartialDate(year=2023, month=None, day=15)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PartialDate: future year raises
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_date_future_year_raises():
    with pytest.raises(ValidationError, match="future"):
        pd(2099)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PartialDate.__le__: year-level comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_date_le_year_level():
    assert pd(2022) <= pd(2023)
    assert not (pd(2023) <= pd(2022))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Partial-date leniency: same year-month, one side missing day → True
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_date_leniency_same_year_month_missing_day():
    """Same year+month with missing day on either side → not a violation (leniency)."""
    start = pd(2024, 3)          # Year+month only, no day
    onset = pd(2024, 3, 15)      # Full date
    assert start <= onset         # Leniency: month matches, start has no day → True

    start2 = pd(2024, 3, 10)
    onset2 = pd(2024, 3)         # Onset has no day
    assert start2 <= onset2       # Leniency: onset has no day → True


# ─────────────────────────────────────────────────────────────────────────────
# 6. PartialDate.__le__: full date comparison
# ─────────────────────────────────────────────────────────────────────────────

def test_partial_date_le_full_date():
    assert pd(2024, 3, 10) <= pd(2024, 3, 15)
    assert not (pd(2024, 3, 20) <= pd(2024, 3, 15))


# ─────────────────────────────────────────────────────────────────────────────
# 7. SuspectDrug: stop before start raises
# ─────────────────────────────────────────────────────────────────────────────

def test_suspect_drug_stop_before_start_raises():
    with pytest.raises(ValidationError, match="stop_date.*cannot be before"):
        SuspectDrug(
            drug_name="Warfarin",
            start_date=pd(2024, 5, 10),
            stop_date=pd(2024, 5, 5),   # Before start
        )


# ─────────────────────────────────────────────────────────────────────────────
# 8. VerbatimEvent: serious=True, no criteria raises
# ─────────────────────────────────────────────────────────────────────────────

def test_serious_event_without_criteria_raises():
    with pytest.raises(ValidationError, match="seriousness_criteria is empty"):
        VerbatimEvent(verbatim_term="Rash", serious=True, seriousness_criteria=[])


# ─────────────────────────────────────────────────────────────────────────────
# 9. VerbatimEvent: serious=False, no criteria = valid non-serious
# ─────────────────────────────────────────────────────────────────────────────

def test_non_serious_event_no_criteria_valid():
    e = VerbatimEvent(verbatim_term="Mild headache", serious=False, seriousness_criteria=[])
    assert e.serious is False
    assert e.seriousness_criteria == []


# ─────────────────────────────────────────────────────────────────────────────
# 10. VerbatimEvent: HOSPITALIZATION without HospitalizationDetails raises
# ─────────────────────────────────────────────────────────────────────────────

def test_hospitalization_criterion_without_details_raises():
    with pytest.raises(ValidationError, match="hospitalization_details is missing"):
        VerbatimEvent(
            verbatim_term="Allergic reaction",
            serious=True,
            seriousness_criteria=[SeriousnessCriterion.HOSPITALIZATION],
            hospitalization_details=None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 11. VerbatimEvent: HospitalizationDetails without criterion raises
# ─────────────────────────────────────────────────────────────────────────────

def test_hospitalization_details_without_criterion_raises():
    hosp = HospitalizationDetails(
        hospital_name="General Hospital",
        hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION,
    )
    with pytest.raises(ValidationError, match="HOSPITALIZATION is not in seriousness_criteria"):
        VerbatimEvent(
            verbatim_term="Rash",
            serious=True,
            seriousness_criteria=[SeriousnessCriterion.LIFE_THREATENING],  # No HOSPITALIZATION
            hospitalization_details=hosp,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 12. HospitalizationDetails: discharge before admission raises
# ─────────────────────────────────────────────────────────────────────────────

def test_hospitalization_discharge_before_admission_raises():
    with pytest.raises(ValidationError, match="cannot be before date_of_admission"):
        HospitalizationDetails(
            hospital_name="City Hospital",
            date_of_admission=pd(2024, 3, 18),
            date_of_discharge=pd(2024, 3, 10),  # Before admission
            hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 13. ExtractedCaseEntities: drug start after event onset raises
# ─────────────────────────────────────────────────────────────────────────────

def test_drug_start_after_event_onset_raises():
    with pytest.raises(ValidationError, match="AFTER event"):
        ExtractedCaseEntities(
            case_id="ICSR-TEST-001",
            narrative_hash="a" * 64,
            suspect_drugs=[suspect_drug(start=pd(2024, 3, 20))],  # Start AFTER onset
            verbatim_events=[event(onset=pd(2024, 3, 10))],       # Onset earlier
            extraction_confidence=0.88,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 14. ExtractedCaseEntities: partial-date leniency in cross-model check
# ─────────────────────────────────────────────────────────────────────────────

def test_cross_model_partial_date_leniency():
    """Same year+month on both sides with missing day — should NOT raise."""
    entities = ExtractedCaseEntities(
        case_id="ICSR-TEST-002",
        narrative_hash="b" * 64,
        suspect_drugs=[suspect_drug(start=pd(2024, 3))],    # Year+month only
        verbatim_events=[event(onset=pd(2024, 3))],          # Same year+month
        extraction_confidence=0.85,
    )
    assert len(entities.suspect_drugs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 15. ExtractedCaseEntities: no SUSPECT drug raises
# ─────────────────────────────────────────────────────────────────────────────

def test_no_suspect_drug_raises():
    with pytest.raises(ValidationError, match="At least one drug must have drug_role=SUSPECT"):
        ExtractedCaseEntities(
            case_id="ICSR-TEST-003",
            narrative_hash="c" * 64,
            suspect_drugs=[suspect_drug(role=DrugRole.CONCOMITANT)],  # No SUSPECT
            verbatim_events=[event()],
            extraction_confidence=0.88,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 16. node_id auto-generated and unique per entity
# ─────────────────────────────────────────────────────────────────────────────

def test_node_ids_auto_generated_and_unique():
    d1 = suspect_drug("Drug A")
    d2 = suspect_drug("Drug B")
    e1 = event("Event A")
    e2 = event("Event B")

    assert d1.node_id.startswith("DRG-")
    assert e1.node_id.startswith("AE-")
    assert d1.node_id != d2.node_id
    assert e1.node_id != e2.node_id


# ─────────────────────────────────────────────────────────────────────────────
# 17. CONCOMITANT drug allowed when SUSPECT also present
# ─────────────────────────────────────────────────────────────────────────────

def test_concomitant_drug_allowed_with_suspect():
    entities = minimal_entities(
        suspect_drugs=[
            suspect_drug("Amoxicillin", start=pd(2024, 1, 10), role=DrugRole.SUSPECT),
            suspect_drug("Ibuprofen", start=pd(2024, 1, 5), role=DrugRole.CONCOMITANT),
        ]
    )
    assert len(entities.suspect_drugs) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 18. Rechallenge enum: all four states valid
# ─────────────────────────────────────────────────────────────────────────────

def test_rechallenge_all_four_states():
    for state in [Rechallenge.YES, Rechallenge.NO, Rechallenge.NOT_REPORTED, Rechallenge.UNKNOWN]:
        d = SuspectDrug(drug_name="TestDrug", rechallenge=state)
        assert d.rechallenge == state
