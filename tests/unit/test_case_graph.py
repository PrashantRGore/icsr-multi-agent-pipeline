"""
tests/unit/test_case_graph.py
================================
Unit tests for infra/case_graph.py

Tests:
  1.  Empty CaseGraph creates an empty networkx DiGraph
  2.  from_entities builds DRUG_START and AE_ONSET nodes from Case 001
  3.  from_entities: DRUG_START → AE_ONSET PRECEDES edge present
  4.  from_entities: DECHALLENGE node present when drug has dechallenge
  5.  from_entities: RECHALLENGE node present when rechallenge=YES
  6.  assert_temporal_order: valid order → all assertions pass (R1)
  7.  assert_temporal_order: drug start AFTER AE onset → R1 fails
  8.  from_entities with hospitalization details → HOSPITALIZATION node
  9.  serialize_to_json → JSON string parseable
  10. from_json reconstructs correct node count
  11. summary returns dict with num_nodes, num_edges
  12. add_edge with unknown node_id raises ValueError
"""
import json
from pathlib import Path

import pytest

from infra.case_graph import CaseGraph, NodeType, EdgeType, GraphAssertion
from schemas.extraction import (
    ExtractedCaseEntities, SuspectDrug, VerbatimEvent, PartialDate,
    SeriousnessCriterion, Dechallenge, Rechallenge, HospitalizationDetails,
    HospitalizationCausalityFlag,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _drug(
    name="Amoxicillin",
    start_year=2024, start_month=3, start_day=10,
    dechallenge=Dechallenge.NA,
    rechallenge=Rechallenge.NOT_REPORTED,
) -> SuspectDrug:
    return SuspectDrug(
        drug_name=name,
        start_date=PartialDate(year=start_year, month=start_month, day=start_day),
        dechallenge=dechallenge,
        rechallenge=rechallenge,
    )


def _event(
    term="Anaphylaxis",
    onset_year=2024, onset_month=3, onset_day=11,
    serious=True,
) -> VerbatimEvent:
    return VerbatimEvent(
        verbatim_term=term,
        onset_date=PartialDate(year=onset_year, month=onset_month, day=onset_day),
        serious=serious,
        seriousness_criteria=[SeriousnessCriterion.LIFE_THREATENING] if serious else [],
    )


def _entities(drugs=None, events=None) -> ExtractedCaseEntities:
    return ExtractedCaseEntities(
        case_id="ICSR-TEST-001",
        narrative_hash="a" * 64,
        extraction_confidence=0.95,
        suspect_drugs=drugs or [_drug()],
        verbatim_events=events or [_event()],
        reporter_type="Physician",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Empty graph
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_graph():
    g = CaseGraph("test-case")
    assert g._g.number_of_nodes() == 0
    assert g._g.number_of_edges() == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. from_entities builds DRUG_START and AE_ONSET nodes
# ─────────────────────────────────────────────────────────────────────────────

def test_from_entities_creates_drug_start_and_ae_onset():
    entities = _entities()
    g = CaseGraph.from_entities(entities)

    node_types = {d.get("node_type") for _, d in g._g.nodes(data=True)}
    assert NodeType.DRUG_START.value in node_types
    assert NodeType.AE_ONSET.value  in node_types


# ─────────────────────────────────────────────────────────────────────────────
# 3. DRUG_START → AE_ONSET PRECEDES edge present
# ─────────────────────────────────────────────────────────────────────────────

def test_from_entities_precedes_edge_present():
    entities = _entities()
    g = CaseGraph.from_entities(entities)

    precedes_edges = [
        (src, dst)
        for src, dst, data in g._g.edges(data=True)
        if data.get("edge_type") == EdgeType.PRECEDES.value
    ]
    assert len(precedes_edges) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. DECHALLENGE node when drug has dechallenge
# ─────────────────────────────────────────────────────────────────────────────

def test_from_entities_dechallenge_node():
    drug = SuspectDrug(
        drug_name="Amoxicillin",
        start_date=PartialDate(year=2024, month=3, day=10),
        stop_date=PartialDate(year=2024, month=3, day=15),
        dechallenge=Dechallenge.YES,
        rechallenge=Rechallenge.NOT_REPORTED,
    )
    entities = _entities(drugs=[drug])
    g = CaseGraph.from_entities(entities)

    node_types = {d.get("node_type") for _, d in g._g.nodes(data=True)}
    assert NodeType.DECHALLENGE.value in node_types


# ─────────────────────────────────────────────────────────────────────────────
# 5. RECHALLENGE node when rechallenge=YES
# ─────────────────────────────────────────────────────────────────────────────

def test_from_entities_rechallenge_node():
    drug = SuspectDrug(
        drug_name="Atorvastatin",
        start_date=PartialDate(year=2024, month=1, day=5),
        stop_date=PartialDate(year=2024, month=2, day=1),
        dechallenge=Dechallenge.YES,
        rechallenge=Rechallenge.YES,
    )
    entities = _entities(drugs=[drug])
    g = CaseGraph.from_entities(entities)

    node_types = {d.get("node_type") for _, d in g._g.nodes(data=True)}
    assert NodeType.RECHALLENGE.value in node_types


# ─────────────────────────────────────────────────────────────────────────────
# 6. assert_temporal_order: valid order → R1 passes
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_order_valid_passes_r1():
    # Drug start March 10 → AE onset March 11 → valid
    entities = _entities()
    g = CaseGraph.from_entities(entities)
    assertions = g.assert_temporal_order()

    r1_assertions = [a for a in assertions if "R1" in a.rule]
    assert all(a.passed for a in r1_assertions), \
        f"Expected all R1 to pass, got failures: {[a.description for a in r1_assertions if not a.passed]}"


# ─────────────────────────────────────────────────────────────────────────────
# 7. assert_temporal_order: drug start AFTER AE onset → R1 fails
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_order_drug_after_ae_r1_fails():
    """
    The extraction schema's cross-model validator blocks invalid dates at
    construction time, so we build the graph manually to test graph-level
    assertion R1 independently.
    """
    g = CaseGraph("ICSR-TEST-001")
    # Drug started March 20
    g.add_node("DRG-001-START", NodeType.DRUG_START,
               label="Start: Amoxicillin", date_repr="20240320",
               drug_node_id="DRG-001", drug_name="Amoxicillin")
    # AE onset March 11 (BEFORE drug start → violation)
    g.add_node("AE-001", NodeType.AE_ONSET,
               label="AE Onset: Anaphylaxis", date_repr="20240311",
               event_node_id="AE-001", verbatim_term="Anaphylaxis", serious=True)
    g.add_edge("DRG-001-START", "AE-001", EdgeType.PRECEDES)

    assertions = g.assert_temporal_order()
    r1_failed = [a for a in assertions if "R1" in a.rule and not a.passed]
    assert len(r1_failed) >= 1
    assert "AFTER" in r1_failed[0].description


# ─────────────────────────────────────────────────────────────────────────────
# 8. Hospitalization node created
# ─────────────────────────────────────────────────────────────────────────────

def test_from_entities_hospitalization_node():
    event = VerbatimEvent(
        verbatim_term="Drug-induced liver injury",
        onset_date=PartialDate(year=2024, month=1, day=10),
        serious=True,
        seriousness_criteria=[SeriousnessCriterion.HOSPITALIZATION],
        hospitalization_details=HospitalizationDetails(
            hospitalization_flag=HospitalizationCausalityFlag.EVENT_CAUSED_HOSPITALIZATION,
            date_of_admission=PartialDate(year=2024, month=1, day=11),
        ),
    )
    # Drug must start BEFORE event onset (Jan 5 < Jan 10) to pass pydantic cross-validator
    drug = _drug(start_year=2024, start_month=1, start_day=5)
    entities = _entities(drugs=[drug], events=[event])
    g = CaseGraph.from_entities(entities)

    node_types = {d.get("node_type") for _, d in g._g.nodes(data=True)}
    assert NodeType.HOSPITALIZATION.value in node_types


# ─────────────────────────────────────────────────────────────────────────────
# 9. serialize_to_json → JSON string parseable
# ─────────────────────────────────────────────────────────────────────────────

def test_serialize_to_json_parseable():
    g = CaseGraph.from_entities(_entities())
    json_str = g.serialize_to_json()
    data = json.loads(json_str)
    assert "nodes" in data
    # networkx 3.x uses 'edges' key; older versions used 'links'
    assert "edges" in data or "links" in data


# ─────────────────────────────────────────────────────────────────────────────
# 10. from_json reconstructs correct node count
# ─────────────────────────────────────────────────────────────────────────────

def test_from_json_reconstructs_node_count():
    g_orig = CaseGraph.from_entities(_entities())
    n_orig = g_orig._g.number_of_nodes()

    json_str  = g_orig.serialize_to_json()
    g_restored = CaseGraph.from_json("ICSR-TEST-001", json_str)

    assert g_restored._g.number_of_nodes() == n_orig


# ─────────────────────────────────────────────────────────────────────────────
# 11. summary returns expected keys
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_keys():
    g = CaseGraph.from_entities(_entities())
    s = g.summary()
    assert "num_nodes" in s
    assert "num_edges" in s
    assert "node_types" in s
    assert s["case_id"] == "ICSR-TEST-001"


# ─────────────────────────────────────────────────────────────────────────────
# 12. add_edge with unknown node raises ValueError
# ─────────────────────────────────────────────────────────────────────────────

def test_add_edge_unknown_node_raises():
    g = CaseGraph("test")
    g.add_node("A", NodeType.DRUG_START, "Drug Start")
    with pytest.raises(ValueError, match="not in graph"):
        g.add_edge("A", "DOES-NOT-EXIST", EdgeType.PRECEDES)
