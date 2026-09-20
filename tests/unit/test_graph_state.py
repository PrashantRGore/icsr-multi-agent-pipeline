"""
tests/unit/test_graph_state.py
================================
Unit tests for graph/state.py

Tests:
  1.  initial_state creates all required fields
  2.  initial_state: error_log is empty list (not None)
  3.  initial_state: partial_e2b is empty dict (not None)
  4.  initial_state: pipeline_halted is False
  5.  initial_state: all Optional agent outputs are None
  6.  initial_state: trace_id auto-generated UUID when not provided
  7.  initial_state: trace_id preserved when provided
  8.  initial_state: narrative_hash stored correctly
  9.  refresh_review_id: returns new review_id, preserves all other fields
  10. refresh_review_id: new review_id is a valid UUID
  11. initial_state: review_id is a valid UUID string
"""
import uuid

import pytest

from graph.state import GraphState, initial_state, refresh_review_id


def _make_state(**overrides) -> GraphState:
    base = dict(
        case_id="ICSR-20240315-SYN",
        raw_narrative="Patient developed anaphylaxis after amoxicillin.",
        narrative_hash="a" * 64,
    )
    base.update(overrides)
    return initial_state(**base)


def _is_valid_uuid(s: str) -> bool:
    try:
        uuid.UUID(s)
        return True
    except (ValueError, TypeError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 1. Required fields present
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_required_fields():
    state = _make_state()
    assert state["case_id"]        == "ICSR-20240315-SYN"
    assert state["raw_narrative"]  == "Patient developed anaphylaxis after amoxicillin."
    assert state["narrative_hash"] == "a" * 64


# ─────────────────────────────────────────────────────────────────────────────
# 2. error_log is empty list
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_error_log_empty_list():
    state = _make_state()
    assert state["error_log"] == []
    assert isinstance(state["error_log"], list)


# ─────────────────────────────────────────────────────────────────────────────
# 3. partial_e2b is empty dict
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_partial_e2b_empty_dict():
    state = _make_state()
    assert state["partial_e2b"] == {}
    assert isinstance(state["partial_e2b"], dict)


# ─────────────────────────────────────────────────────────────────────────────
# 4. pipeline_halted is False
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_pipeline_not_halted():
    state = _make_state()
    assert state["pipeline_halted"] is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. All Optional agent outputs are None
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_agent_outputs_none():
    state = _make_state()
    for field in [
        "triage_output", "extracted_entities", "causality_matrix",
        "coded_events", "listedness_evaluations", "qc_report",
        "narrative_text", "hitl_stage", "hitl_correction", "graph_payload",
        "current_stage", "next_stage", "risk_tier", "oversight_mode",
    ]:
        assert state[field] is None, f"Expected {field} to be None"


# ─────────────────────────────────────────────────────────────────────────────
# 6. trace_id auto-generated when not provided
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_trace_id_auto_generated():
    s1 = _make_state()
    s2 = _make_state()
    assert _is_valid_uuid(s1["trace_id"])
    assert _is_valid_uuid(s2["trace_id"])
    # Two separate calls should produce different trace_ids
    assert s1["trace_id"] != s2["trace_id"]


# ─────────────────────────────────────────────────────────────────────────────
# 7. trace_id preserved when provided
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_trace_id_preserved():
    tid = "my-custom-trace-id-12345"
    state = _make_state(trace_id=tid)
    assert state["trace_id"] == tid


# ─────────────────────────────────────────────────────────────────────────────
# 8. narrative_hash stored correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_narrative_hash():
    h = "b" * 64
    state = _make_state(narrative_hash=h)
    assert state["narrative_hash"] == h


# ─────────────────────────────────────────────────────────────────────────────
# 9. refresh_review_id preserves all other fields
# ─────────────────────────────────────────────────────────────────────────────

def test_refresh_review_id_preserves_other_fields():
    state = _make_state()
    original_trace_id = state["trace_id"]
    original_case_id  = state["case_id"]
    original_review   = state["review_id"]

    new_state = refresh_review_id(state)

    assert new_state["trace_id"] == original_trace_id
    assert new_state["case_id"]  == original_case_id
    assert new_state["review_id"] != original_review   # Must change


# ─────────────────────────────────────────────────────────────────────────────
# 10. refresh_review_id: new review_id is valid UUID
# ─────────────────────────────────────────────────────────────────────────────

def test_refresh_review_id_valid_uuid():
    state     = _make_state()
    new_state = refresh_review_id(state)
    assert _is_valid_uuid(new_state["review_id"])


# ─────────────────────────────────────────────────────────────────────────────
# 11. initial_state review_id is valid UUID
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_state_review_id_valid_uuid():
    state = _make_state()
    assert _is_valid_uuid(state["review_id"])
