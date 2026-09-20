"""
schemas/extraction.py
=====================
ExtractedCaseEntities — output of the Extraction Agent.

Key design decisions (v4):
  - PartialDate: supports year-only, year+month, or full date with partial-date leniency
    in cross-field comparisons (same year-month, missing day → not a violation).
  - No future dates are permitted in any PartialDate field.
  - Drug start_date ≤ event onset_date cross-model validator (ExtractedCaseEntities level).
  - Rechallenge enum has four clinically precise states (YES/NO/NOT_REPORTED/UNKNOWN).
    NOT_REPORTED ≠ UNKNOWN — see Rechallenge docstring.
  - serious=False + seriousness_criteria=[] → valid non-serious event (no error).
  - serious=True + seriousness_criteria=[] → schema validation error.
  - HospitalizationDetails required when Hospitalization criterion selected.
  - node_id: str on SuspectDrug and VerbatimEvent — MAGMA provenance IDs.
  - patient_ethnicity included for CIOMS WG XIV fairness/equity bias tracking.
  - partial_e2b: dict populated here with E2B(R3) field codes for direct blackboard writes.

Compliance:
  - ICH E2B(R3) fields cross-referenced in field descriptions.
  - 21 CFR Part 11: all schemas frozen (immutable after creation).
"""
from __future__ import annotations

import uuid
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class DrugRole(str, Enum):
    SUSPECT     = "SUSPECT"
    CONCOMITANT = "CONCOMITANT"
    INTERACTING = "INTERACTING"


class Dechallenge(str, Enum):
    """
    Outcome after suspect drug was stopped.
    YES     — drug stopped → AE resolved or improved
    NO      — drug stopped → AE continued or worsened
    UNKNOWN — drug stopped but AE outcome not documented in source
    NA      — drug was NOT stopped during the reported period
    Maps to ICH E2B(R3) G.k.8.
    """
    YES     = "YES"
    NO      = "NO"
    UNKNOWN = "UNKNOWN"
    NA      = "N/A"


class Rechallenge(str, Enum):
    """
    Outcome after suspect drug was restarted.

    Four clinically precise states — do NOT conflate NOT_REPORTED with UNKNOWN:
      YES          — drug restarted AND the relevant AE re-occurred (rechallenge positive)
      NO           — drug restarted AND the relevant AE did NOT re-occur (rechallenge negative)
      NOT_REPORTED — no information about drug being restarted exists in the source document
      UNKNOWN      — drug was restarted but whether the AE re-occurred is absent from source

    NOT_REPORTED: the rechallenge attempt is not mentioned at all.
    UNKNOWN:      a rechallenge happened, but its outcome is undocumented.

    Maps to ICH E2B(R3) G.k.9.i.1.
    """
    YES          = "YES"
    NO           = "NO"
    NOT_REPORTED = "NOT_REPORTED"
    UNKNOWN      = "UNKNOWN"


class SeriousnessCriterion(str, Enum):
    """
    ICH E2A seriousness criteria.
    Maps to ICH E2B(R3) Section E.i.3.1 - E.i.3.5 + custom extensions.
    """
    DEATH              = "Death"
    LIFE_THREATENING   = "Life-threatening"
    HOSPITALIZATION    = "Hospitalization"
    DISABILITY         = "Disability / Incapacity"
    CONGENITAL_ANOMALY = "Congenital anomaly"
    INTERVENTION_REQD  = "Intervention required"       # Important for medical devices
    OTHER_MEDICALLY_IMP= "Other medically important condition"


# ─────────────────────────────────────────────────────────────────────────────
# Partial-Date Model
# ─────────────────────────────────────────────────────────────────────────────

class PartialDate(BaseModel):
    """
    Represents a date that may be partially known.
    Supports ICH E2B(R3) partial date handling (CCYYMMDD with nullflavors).

    Rules:
      - year is always required
      - month is required when day is provided
      - No component may represent a future date (validated against today)
    """
    year:  int           = Field(..., ge=1900, le=2999)
    month: Optional[int] = Field(None, ge=1, le=12)
    day:   Optional[int] = Field(None, ge=1, le=31)

    @model_validator(mode="after")
    def day_requires_month(self) -> "PartialDate":
        if self.day is not None and self.month is None:
            raise ValueError("day cannot be specified without month")
        return self

    @model_validator(mode="after")
    def no_future_dates(self) -> "PartialDate":
        today = date.today()
        if self.year > today.year:
            raise ValueError(f"Year {self.year} is in the future (today={today})")
        if self.year == today.year and self.month is not None:
            if self.month > today.month:
                raise ValueError(
                    f"Month {self.month}/{self.year} is in the future (today={today})"
                )
            if self.month == today.month and self.day is not None:
                if self.day > today.day:
                    raise ValueError(
                        f"Date {self.day}/{self.month}/{self.year} is in the future "
                        f"(today={today})"
                    )
        return self

    def __le__(self, other: "PartialDate") -> bool:
        """
        Partial-date comparison at the coarsest shared precision.
        Partial-date leniency: if year AND month are the same and either side
        lacks a day component, the comparison returns True (not a violation).
        """
        if self.year != other.year:
            return self.year <= other.year
        # Same year — compare months if both present
        if self.month is None or other.month is None:
            return True   # Year-level precision only — cannot determine order
        if self.month != other.month:
            return self.month <= other.month
        # Same year-month — if day missing on either side, accept (partial-date leniency)
        if self.day is None or other.day is None:
            return True
        return self.day <= other.day

    def to_e2b_str(self) -> str:
        """Serialize to E2B(R3) CCYYMMDD format with nullflavor padding."""
        y = str(self.year)
        m = str(self.month).zfill(2) if self.month else "00"
        d = str(self.day).zfill(2) if self.day else "00"
        return f"{y}{m}{d}"


# ─────────────────────────────────────────────────────────────────────────────
# Hospitalization Details
# ─────────────────────────────────────────────────────────────────────────────

class HospitalizationCausalityFlag(str, Enum):
    EVENT_CAUSED_HOSPITALIZATION          = "Event caused hospitalization"
    EVENT_DID_NOT_CAUSE_HOSPITALIZATION   = "Event did not cause hospitalization"


class HospitalizationDetails(BaseModel):
    """
    Required when seriousness_criteria includes SeriousnessCriterion.HOSPITALIZATION.
    Maps to ICH E2B(R3) Section B.2 (patient outcome).
    """
    hospital_name:        Optional[str]                  = None
    date_of_admission:    Optional[PartialDate]           = None
    date_of_discharge:    Optional[PartialDate]           = None
    hospitalization_flag: HospitalizationCausalityFlag

    @model_validator(mode="after")
    def discharge_after_admission(self) -> "HospitalizationDetails":
        if self.date_of_admission and self.date_of_discharge:
            if not (self.date_of_admission <= self.date_of_discharge):
                raise ValueError(
                    "date_of_discharge cannot be before date_of_admission"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Non-Serious Event Note
# ─────────────────────────────────────────────────────────────────────────────

class NonSeriousEventNote(BaseModel):
    """
    Explicit non-serious event marker.
    Created when serious=False and seriousness_criteria is empty.
    Prevents the agent from treating absent criteria as a schema error.
    """
    verbatim_term: str
    reporter_term: Optional[str]   = None
    onset_date:    Optional[PartialDate] = None
    outcome:       Optional[str]   = None


# ─────────────────────────────────────────────────────────────────────────────
# Verbatim Event
# ─────────────────────────────────────────────────────────────────────────────

class VerbatimEvent(BaseModel):
    """
    Single adverse event extracted from the narrative.

    Seriousness logic:
      serious=False + seriousness_criteria=[] → non-serious event (valid)
      serious=True  + seriousness_criteria=[] → validation error
      serious=True  + HOSPITALIZATION in criteria → hospitalization_details required

    node_id: MAGMA provenance ID — assigned by ExtractionAgent, used by QCAgent
    for exact source sentence citation (<ref:node_id> in linearized context).

    Maps to ICH E2B(R3) Section E.i.
    """
    node_id:                str                    = Field(
        default_factory=lambda: f"AE-{uuid.uuid4().hex[:8].upper()}",
        description="MAGMA provenance ID — unique per extracted event"
    )
    verbatim_term:          str                    = Field(..., min_length=1)
    reporter_term:          Optional[str]          = None
    onset_date:             Optional[PartialDate]  = None   # E2B: E.i.4
    outcome:                Optional[str]          = None   # E2B: E.i.7
    serious:                bool                   = False  # E2B: E.i.3
    seriousness_criteria:   list[SeriousnessCriterion] = Field(default_factory=list)
    hospitalization_details: Optional[HospitalizationDetails] = None

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def serious_requires_criteria(self) -> "VerbatimEvent":
        """
        Only validate when serious=True.
        serious=False with empty criteria is a valid non-serious event.
        """
        if self.serious and not self.seriousness_criteria:
            raise ValueError(
                f"Event '{self.verbatim_term}' is marked serious=True "
                "but seriousness_criteria is empty. "
                "If the event is non-serious, set serious=False."
            )
        return self

    @model_validator(mode="after")
    def hospitalization_requires_details(self) -> "VerbatimEvent":
        if SeriousnessCriterion.HOSPITALIZATION in self.seriousness_criteria:
            if self.hospitalization_details is None:
                raise ValueError(
                    f"Event '{self.verbatim_term}' has Hospitalization criterion "
                    "but hospitalization_details is missing."
                )
        return self

    @model_validator(mode="after")
    def details_require_hospitalization_criterion(self) -> "VerbatimEvent":
        if (
            self.hospitalization_details is not None
            and SeriousnessCriterion.HOSPITALIZATION not in self.seriousness_criteria
        ):
            raise ValueError(
                "hospitalization_details provided but HOSPITALIZATION is not "
                "in seriousness_criteria"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Suspect Drug
# ─────────────────────────────────────────────────────────────────────────────

class SuspectDrug(BaseModel):
    """
    A single suspect (or concomitant / interacting) drug.

    node_id: MAGMA provenance ID for entity graph co-reference resolution.
    rxcui: canonical RxNorm concept identifier (fetched from RxNorm REST API).
    start_date: must NOT be after any linked VerbatimEvent.onset_date.
                Validated at the ExtractedCaseEntities level.

    Maps to ICH E2B(R3) Section G.k.
    """
    node_id:      str            = Field(
        default_factory=lambda: f"DRG-{uuid.uuid4().hex[:8].upper()}",
        description="MAGMA provenance ID — unique per extracted drug"
    )
    drug_name:    str            = Field(..., min_length=1)
    rxcui:        Optional[str]  = Field(None, description="RxNorm RxCUI (G.k.2.1.1)")
    rxnorm_label: Optional[str]  = Field(None, description="Normalized drug label from RxNorm")
    drug_class:   Optional[str]  = Field(None, description="EPC drug class from RxNorm")
    dose:         Optional[str]  = None                             # G.k.4.r.7
    route:        Optional[str]  = None                             # G.k.4.r.10
    indication:   Optional[str]  = None                             # G.k.6
    start_date:   Optional[PartialDate] = Field(
        None,
        description=(
            "Drug start date (G.k.4.r.4). "
            "Must not be after any linked adverse event onset date. "
            "Partial-date leniency applies (same year-month → not a violation). "
            "Cannot be a future date."
        )
    )
    stop_date:    Optional[PartialDate] = Field(None, description="Drug stop date (G.k.4.r.5)")
    drug_role:    DrugRole              = DrugRole.SUSPECT          # G.k.1
    dechallenge:  Dechallenge           = Dechallenge.UNKNOWN       # G.k.8
    rechallenge:  Rechallenge           = Rechallenge.NOT_REPORTED  # G.k.9.i.1

    @model_validator(mode="after")
    def stop_after_start(self) -> "SuspectDrug":
        if self.start_date and self.stop_date:
            if not (self.start_date <= self.stop_date):
                raise ValueError(
                    f"stop_date ({self.stop_date}) cannot be before "
                    f"start_date ({self.start_date}) for drug '{self.drug_name}'"
                )
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Top-Level Extraction Schema
# ─────────────────────────────────────────────────────────────────────────────

class ExtractedCaseEntities(BaseModel):
    """
    Full output of the Extraction Agent.
    Immutable after creation (frozen=True).

    partial_e2b: dict pre-populated with E2B(R3) field codes that the
    ExtractionAgent can write to directly. Downstream agents (CodingAgent,
    CausalityAgent, etc.) add their fields. NarrativeAgent finalizes the object.

    Compliance: 21 CFR Part 11 — frozen schema, SHA-256 narrative hash linkage.
    """
    model_config = {"frozen": True}

    case_id:               str
    narrative_hash:        str   = Field(
        ..., min_length=64, max_length=64,
        description="Must match TriageOutput.narrative_hash (SHA-256)"
    )
    suspect_drugs:         list[SuspectDrug]     = Field(..., min_length=1)
    verbatim_events:       list[VerbatimEvent]   = Field(..., min_length=1)

    # Patient demographics (ICH E2B(R3) Section D)
    patient_age:           Optional[str]  = None   # D.2.2
    patient_sex:           Optional[str]  = None   # D.5
    patient_ethnicity:     Optional[str]  = Field(
        None,
        description=(
            "Patient ethnicity (e.g., Hispanic, Caucasian, Asian, African). "
            "Extract verbatim from narrative. "
            "Used for CIOMS WG XIV fairness/equity bias monitoring in LearningDB."
        )
    )
    patient_weight_kg:     Optional[float] = Field(None, ge=0.5, le=700.0)  # D.3
    patient_height_cm:     Optional[float] = Field(None, ge=20.0, le=280.0) # D.4

    reporter_type:         Optional[str]  = None   # C.2.1 (HCP | Consumer | Lawyer | Other)
    country_of_occurrence: Optional[str]  = None   # C.1.9

    extraction_confidence: float  = Field(..., ge=0.0, le=1.0)
    agent_id:              str    = "extraction-agent-v1"
    prompt_version:        str    = "extraction-prompt-v1.0"

    # E2B(R3) partial blackboard state — agents write directly using E2B field codes
    partial_e2b:           dict   = Field(
        default_factory=dict,
        description=(
            "Partial E2B(R3) JSON object. Keys are E2B field codes (e.g. 'D.1', 'E.i.4'). "
            "Each agent writes its section; NarrativeAgent finalizes."
        )
    )

    # ── Cross-model validators ───────────────────────────────────────────────

    @model_validator(mode="after")
    def drug_start_before_event_onset(self) -> "ExtractedCaseEntities":
        """
        For each suspect drug with a start_date, verify start_date ≤ onset_date
        for every VerbatimEvent with an onset_date.

        Partial-date leniency: PartialDate.__le__() handles same-year-month cases
        (missing day on either side → not a violation).

        Future-date check: enforced individually by PartialDate.no_future_dates.

        HITL behaviour: raises ValueError → routes to HITL with structured message.
        """
        errors: list[str] = []
        for drug in self.suspect_drugs:
            if drug.start_date is None:
                continue
            for event in self.verbatim_events:
                if event.onset_date is None:
                    continue
                if not (drug.start_date <= event.onset_date):
                    errors.append(
                        f"Drug '{drug.drug_name}' "
                        f"(node_id={drug.node_id}, "
                        f"start={drug.start_date.to_e2b_str()}) "
                        f"is AFTER event '{event.verbatim_term}' "
                        f"(node_id={event.node_id}, "
                        f"onset={event.onset_date.to_e2b_str()}). "
                        "A suspect drug must precede or coincide with AE onset. "
                        "Partial-date leniency: same year+month with missing day is accepted."
                    )
        if errors:
            raise ValueError(
                "Drug start_date / Event onset_date conflict(s) detected:\n"
                + "\n".join(errors)
            )
        return self

    @model_validator(mode="after")
    def at_least_one_suspect_drug(self) -> "ExtractedCaseEntities":
        suspects = [d for d in self.suspect_drugs if d.drug_role == DrugRole.SUSPECT]
        if not suspects:
            raise ValueError(
                "At least one drug must have drug_role=SUSPECT. "
                "Concomitant-only drug lists are not valid for ICSR extraction."
            )
        return self
