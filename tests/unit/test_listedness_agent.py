"""
tests/unit/test_listedness_agent.py
=====================================
Unit tests for ListednessAgent.

DailyMed HTTP calls are fully mocked via requests.Session.
Tests verify:
  1. LISTED event → listedness_status=LISTED, expedited_reporting=False
  2. UNLISTED + serious → UNLISTED + expedited_reporting=True
  3. DailyMed network failure → UNKNOWN → pipeline_halted
  4. In-process cache hit (second call does not re-fetch)
  5. partial_e2b written per event
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import MagicMock, patch

import pytest

from agents.listedness_agent import ListednessAgent
from graph.state import initial_state
from schemas.coding import CodedEvent, CodingStatus
from schemas.extraction import (
    Dechallenge, DrugRole, ExtractedCaseEntities,
    HospitalizationCausalityFlag, HospitalizationDetails,
    PartialDate, Rechallenge, SeriousnessCriterion, SuspectDrug, VerbatimEvent,
)
from schemas.listedness import ListednessStatus


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

NARRATIVE = "Patient received Amoxicillin 500mg PO. Developed anaphylaxis."

_AMOXICILLIN_REACTIONS_TEXT = (
    "Adverse Reactions: The following reactions have been reported during "
    "clinical trials and post-marketing: anaphylaxis, urticaria, rash, diarrhoea."
)

_SPL_LIST_RESP = {"data": [{"setid": "test-set-id-001"}]}
_SPL_SECTIONS_RESP = {"data": [
    {"loinc_code": "34084-4", "text": _AMOXICILLIN_REACTIONS_TEXT}
]}


def _make_entities(event_terms: list[tuple[str, bool]] = None) -> ExtractedCaseEntities:
    """
    event_terms: list of (verbatim_term, is_serious)
    """
    event_terms = event_terms or [("anaphylaxis", True)]
    drug = SuspectDrug(
        drug_name   = "Amoxicillin",
        drug_role   = DrugRole.SUSPECT,
        dechallenge = Dechallenge.YES,
        rechallenge = Rechallenge.NOT_REPORTED,
    )
    events = []
    for term, serious in event_terms:
        criteria = [SeriousnessCriterion.HOSPITALIZATION] if serious else []
        hosp = None
        if serious:
            hosp = HospitalizationDetails(
                hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION
            )
        events.append(VerbatimEvent(
            verbatim_term         = term,
            serious               = serious,
            seriousness_criteria  = criteria,
            hospitalization_details = hosp,
        ))
    return ExtractedCaseEntities(
        case_id="ICSR-20240101-TST",
        narrative_hash="a" * 64,
        suspect_drugs=[drug],
        verbatim_events=events,
        extraction_confidence=0.90,
    )


def _make_coded(entities: ExtractedCaseEntities) -> list[CodedEvent]:
    return [
        CodedEvent(
            event_node_id = e.node_id,
            verbatim_term = e.verbatim_term,
            oae_term      = e.verbatim_term,
            coding_status = CodingStatus.AUTO_CODED,
        )
        for e in entities.verbatim_events
    ]


def _make_state(entities, coded) -> dict:
    h = hashlib.sha256(NARRATIVE.encode()).hexdigest()
    state = initial_state(
        case_id="ICSR-20240101-TST",
        raw_narrative=NARRATIVE,
        narrative_hash=h,
    )
    state["extracted_entities"] = entities
    state["coded_events"]       = coded
    return state


def _mock_session(spls_resp: dict, sections_resp: dict) -> MagicMock:
    """Mock requests.Session returning DailyMed JSON responses."""
    session = MagicMock()

    def get_side_effect(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "spls.json" in url:
            resp.json.return_value = spls_resp
        elif "sections.json" in url:
            resp.json.return_value = sections_resp
        else:
            resp.json.return_value = {"data": []}
        return resp

    session.get.side_effect = get_side_effect
    return session


def _mock_audit_db() -> MagicMock:
    db = MagicMock()
    db.insert_entry.return_value = None
    return db


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestListednessAgentListed:
    def test_listed_event_no_expedited(self):
        """Anaphylaxis in DailyMed Adverse Reactions → LISTED, no expedited flag."""
        entities = _make_entities([("anaphylaxis", True)])
        coded    = _make_coded(entities)
        session  = _mock_session(_SPL_LIST_RESP, _SPL_SECTIONS_RESP)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        result = agent.run(_make_state(entities, coded))

        ev = result["listedness_evaluations"][0]
        assert ev.listedness_status == ListednessStatus.LISTED
        assert ev.expedited_reporting is False
        assert result["next_stage"] == "narrative"

    def test_listed_non_serious_no_expedited(self):
        """Listed non-serious event → expedited_reporting=False."""
        entities = _make_entities([("urticaria", False)])
        coded    = _make_coded(entities)
        session  = _mock_session(_SPL_LIST_RESP, _SPL_SECTIONS_RESP)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        result = agent.run(_make_state(entities, coded))
        assert result["listedness_evaluations"][0].expedited_reporting is False


class TestListednessAgentUnlisted:
    def test_unlisted_serious_triggers_expedited(self):
        """Event not in SPL + serious=True → UNLISTED + expedited_reporting=True."""
        sections_without_event = {"data": [
            {"loinc_code": "34084-4", "text": "No adverse reactions listed."}
        ]}
        entities = _make_entities([("Stevens-Johnson syndrome", True)])
        coded    = _make_coded(entities)
        session  = _mock_session(_SPL_LIST_RESP, sections_without_event)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        result = agent.run(_make_state(entities, coded))
        ev = result["listedness_evaluations"][0]
        assert ev.listedness_status == ListednessStatus.UNLISTED
        assert ev.expedited_reporting is True


class TestListednessAgentNetworkFailure:
    def test_dailymed_failure_unknown_routes_to_hitl(self):
        """DailyMed network failure → UNKNOWN → pipeline_halted."""
        session = MagicMock()
        session.get.side_effect = Exception("Connection refused")

        entities = _make_entities([("anaphylaxis", True)])
        coded    = _make_coded(entities)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        result = agent.run(_make_state(entities, coded))
        ev = result["listedness_evaluations"][0]
        assert ev.listedness_status == ListednessStatus.UNKNOWN
        assert result["pipeline_halted"] is True
        assert result["hitl_stage"] == "listedness"

    def test_no_spl_found_returns_unknown(self):
        """Empty DailyMed SPL list → UNKNOWN."""
        session = _mock_session({"data": []}, {"data": []})
        entities = _make_entities([("anaphylaxis", True)])
        coded    = _make_coded(entities)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        result = agent.run(_make_state(entities, coded))
        assert result["listedness_evaluations"][0].listedness_status == ListednessStatus.UNKNOWN


class TestListednessAgentCache:
    def test_cache_hit_avoids_second_request(self):
        """Two events for same drug → DailyMed only fetched once (cache hit)."""
        session  = _mock_session(_SPL_LIST_RESP, _SPL_SECTIONS_RESP)
        entities = _make_entities([("anaphylaxis", True), ("rash", False)])
        coded    = _make_coded(entities)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        agent.run(_make_state(entities, coded))

        # SPL lookup called once (first event), cache used for second
        spls_calls = [c for c in session.get.call_args_list if "spls.json" in str(c)]
        assert len(spls_calls) == 1

    def test_partial_e2b_written_per_event(self):
        """E.i.3a keys written to partial_e2b for each evaluated event."""
        session  = _mock_session(_SPL_LIST_RESP, _SPL_SECTIONS_RESP)
        entities = _make_entities([("anaphylaxis", True)])
        coded    = _make_coded(entities)

        agent = ListednessAgent(
            llm=MagicMock(), audit_db=_mock_audit_db(), session=session
        )
        result = agent.run(_make_state(entities, coded))
        assert any("E.i.3a" in k for k in result["partial_e2b"])
