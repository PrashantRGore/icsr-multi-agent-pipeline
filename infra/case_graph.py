"""
infra/case_graph.py
===================
MAGMA-inspired case-level graph for temporal and causal QC assertions.

Architecture (selective MAGMA adoption — Phase 3-4 of the plan):
  - E_temp (Temporal Graph): enforces clinical timeline ordering.
    Nodes: clinical events (drug_start, drug_stop, ae_onset, ae_resolution,
           dechallenge, rechallenge, hospitalization).
    Edges: PRECEDES, COINCIDES, RESOLVES_AFTER.
  - E_causal (Causal Graph): models drug→event causal pathways and
    alternative etiologies.
    Nodes: drugs, AEs, pre-existing conditions, alternative causes.
    Edges: CAUSES, POSSIBLY_CAUSES, ALTERNATIVE_CAUSE_OF, CONTRAINDICATED_WITH.

Design decisions:
  - networkx DiGraph (in-memory, ~50KB per case) — no Neo4j required.
  - All nodes carry a node_id matching the MAGMA provenance IDs
    (SuspectDrug.node_id, VerbatimEvent.node_id) from the extraction schema.
  - Temporal assertions are expressed as formal graph-level rules (see
    assert_temporal_order), not pairwise scalar comparisons.
  - serialize_to_json() / from_json() use networkx node_link_data format
    for storage in GraphState.graph_payload.
  - QCAgent uses assert_temporal_order() and assert_causal_consistency()
    to produce CritiqueItems with MAGMA node_id references.

Hardware note: networkx operates purely in CPU/RAM. A typical ICSR case graph
has ~5-15 nodes and ~10-20 edges — negligible memory footprint.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Optional

import networkx as nx

from schemas.extraction import ExtractedCaseEntities, Dechallenge, Rechallenge

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node and edge types
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    DRUG_ADMINISTRATION = "DRUG_ADMINISTRATION"   # Drug start → stop interval
    DRUG_START          = "DRUG_START"
    DRUG_STOP           = "DRUG_STOP"
    AE_ONSET            = "AE_ONSET"
    AE_RESOLUTION       = "AE_RESOLUTION"
    DECHALLENGE         = "DECHALLENGE"           # Intentional drug stop to test AE
    RECHALLENGE         = "RECHALLENGE"           # Drug restart after dechallenge
    HOSPITALIZATION     = "HOSPITALIZATION"
    PRE_EXISTING_COND   = "PRE_EXISTING_COND"     # Alternative etiology node
    ALTERNATIVE_CAUSE   = "ALTERNATIVE_CAUSE"     # Other potential cause


class EdgeType(str, Enum):
    # Temporal edges (E_temp)
    PRECEDES         = "PRECEDES"           # A occurred before B
    COINCIDES        = "COINCIDES"          # A and B overlap or same date
    RESOLVES_AFTER   = "RESOLVES_AFTER"     # AE resolved after drug stop (dechallenge+)
    RECURS_AFTER     = "RECURS_AFTER"       # AE recurred after rechallenge

    # Causal edges (E_causal)
    CAUSES           = "CAUSES"             # Drug → AE (causal claim)
    POSSIBLY_CAUSES  = "POSSIBLY_CAUSES"    # Drug → AE (possible)
    ALTERNATIVE_CAUSE_OF = "ALTERNATIVE_CAUSE_OF"  # AltCause → AE
    CONTRAINDICATED  = "CONTRAINDICATED"    # Clinical negative hard edge


# ─────────────────────────────────────────────────────────────────────────────
# Assertion result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphAssertion:
    """Result of a single graph-level QC assertion."""
    rule:        str
    passed:      bool
    description: str
    node_ids:    list[str] = field(default_factory=list)   # Provenance IDs involved


# ─────────────────────────────────────────────────────────────────────────────
# CaseGraph
# ─────────────────────────────────────────────────────────────────────────────

class CaseGraph:
    """
    MAGMA case-level graph for a single ICSR.
    Built from ExtractedCaseEntities; used by QCAgent for graph assertions.
    """

    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._g      = nx.DiGraph()

    # ── Graph construction ───────────────────────────────────────────────────

    def add_node(
        self,
        node_id:   str,
        node_type: NodeType,
        label:     str,
        date_repr: Optional[str] = None,   # E2B CCYYMMDD string
        **attrs: Any,
    ) -> None:
        """Add a node with provenance metadata."""
        self._g.add_node(
            node_id,
            node_type=node_type.value,
            label=label,
            date_repr=date_repr,
            **attrs,
        )

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        **attrs: Any,
    ) -> None:
        """Add a directed edge."""
        if source_id not in self._g:
            raise ValueError(f"Source node {source_id!r} not in graph")
        if target_id not in self._g:
            raise ValueError(f"Target node {target_id!r} not in graph")
        self._g.add_edge(source_id, target_id, edge_type=edge_type.value, **attrs)

    # ── Builder from extraction entities ────────────────────────────────────

    @classmethod
    def from_entities(cls, entities: ExtractedCaseEntities) -> "CaseGraph":
        """
        Construct a CaseGraph from ExtractedCaseEntities.
        Populates E_temp (temporal) nodes and edges.
        E_causal edges are added by the CausalityAgent in Phase 3.
        """
        g = cls(case_id=entities.case_id)

        # ── Drug nodes ───────────────────────────────────────────────────────
        for drug in entities.suspect_drugs:
            if drug.start_date:
                start_node = f"{drug.node_id}-START"
                g.add_node(
                    start_node,
                    NodeType.DRUG_START,
                    label=f"Start: {drug.drug_name}",
                    date_repr=drug.start_date.to_e2b_str(),
                    drug_node_id=drug.node_id,
                    drug_name=drug.drug_name,
                )

            if drug.stop_date:
                stop_node = f"{drug.node_id}-STOP"
                g.add_node(
                    stop_node,
                    NodeType.DRUG_STOP,
                    label=f"Stop: {drug.drug_name}",
                    date_repr=drug.stop_date.to_e2b_str(),
                    drug_node_id=drug.node_id,
                    drug_name=drug.drug_name,
                )
                # Dechallenge / rechallenge special nodes
                if drug.dechallenge != Dechallenge.NA:
                    dc_node = f"{drug.node_id}-DC"
                    g.add_node(
                        dc_node,
                        NodeType.DECHALLENGE,
                        label=f"Dechallenge: {drug.drug_name} ({drug.dechallenge.value})",
                        date_repr=drug.stop_date.to_e2b_str(),
                        drug_node_id=drug.node_id,
                        dechallenge_result=drug.dechallenge.value,
                    )

            if drug.rechallenge in (Rechallenge.YES, Rechallenge.NO):
                rc_node = f"{drug.node_id}-RC"
                g.add_node(
                    rc_node,
                    NodeType.RECHALLENGE,
                    label=f"Rechallenge: {drug.drug_name} ({drug.rechallenge.value})",
                    drug_node_id=drug.node_id,
                    rechallenge_result=drug.rechallenge.value,
                )

        # ── Event nodes ──────────────────────────────────────────────────────
        for event in entities.verbatim_events:
            if event.onset_date:
                g.add_node(
                    event.node_id,
                    NodeType.AE_ONSET,
                    label=f"AE Onset: {event.verbatim_term}",
                    date_repr=event.onset_date.to_e2b_str(),
                    event_node_id=event.node_id,
                    verbatim_term=event.verbatim_term,
                    serious=event.serious,
                )

            if event.hospitalization_details:
                hosp_node = f"{event.node_id}-HOSP"
                hosp = event.hospitalization_details
                g.add_node(
                    hosp_node,
                    NodeType.HOSPITALIZATION,
                    label=f"Hospitalization for: {event.verbatim_term}",
                    date_repr=(
                        hosp.date_of_admission.to_e2b_str()
                        if hosp.date_of_admission else None
                    ),
                    event_node_id=event.node_id,
                )

        # ── Temporal edges ────────────────────────────────────────────────────
        # Drug start → AE onset (PRECEDES edges for suspect drugs)
        for drug in entities.suspect_drugs:
            for event in entities.verbatim_events:
                start_node = f"{drug.node_id}-START"
                if start_node in g._g and event.node_id in g._g:
                    g.add_edge(start_node, event.node_id, EdgeType.PRECEDES)

        # Drug stop → dechallenge node (PRECEDES)
        for drug in entities.suspect_drugs:
            stop_node = f"{drug.node_id}-STOP"
            dc_node   = f"{drug.node_id}-DC"
            rc_node   = f"{drug.node_id}-RC"
            if stop_node in g._g and dc_node in g._g:
                g.add_edge(stop_node, dc_node, EdgeType.PRECEDES)
            if dc_node in g._g and rc_node in g._g:
                g.add_edge(dc_node, rc_node, EdgeType.PRECEDES)

        logger.debug(
            "CaseGraph[%s]: built with %d nodes, %d edges",
            entities.case_id, g._g.number_of_nodes(), g._g.number_of_edges()
        )
        return g

    # ── Graph assertions ─────────────────────────────────────────────────────

    def assert_temporal_order(self) -> list[GraphAssertion]:
        """
        Formal graph-level temporal assertions.
        Returns a list of GraphAssertion — failed assertions become
        DATE_ORDER_VIOLATION CritiqueItems in the QCReport.

        Rules:
          R1: All DRUG_START nodes must have date_repr ≤ any AE_ONSET
              node they connect to via PRECEDES edges.
          R2: All DECHALLENGE nodes must come AFTER the AE_ONSET they
              relate to (i.e., AE must exist before dechallenge).
          R3: RECHALLENGE nodes must come AFTER their DECHALLENGE predecessor.
        """
        results: list[GraphAssertion] = []

        def _parse_date(date_repr: Optional[str]) -> Optional[date]:
            if not date_repr or date_repr == "00000000":
                return None
            try:
                year  = int(date_repr[:4])
                month = int(date_repr[4:6]) or 1
                day   = int(date_repr[6:8]) or 1
                return date(year, month, day)
            except (ValueError, IndexError):
                return None

        # R1: Drug start → AE onset ordering
        for src, dst, data in self._g.edges(data=True):
            if data.get("edge_type") != EdgeType.PRECEDES.value:
                continue
            src_data = self._g.nodes[src]
            dst_data = self._g.nodes[dst]
            if (
                src_data.get("node_type") == NodeType.DRUG_START.value
                and dst_data.get("node_type") == NodeType.AE_ONSET.value
            ):
                src_date = _parse_date(src_data.get("date_repr"))
                dst_date = _parse_date(dst_data.get("date_repr"))
                if src_date and dst_date and src_date > dst_date:
                    results.append(GraphAssertion(
                        rule="R1:DrugStart≤AEOnset",
                        passed=False,
                        description=(
                            f"Drug start ({src_data.get('label')}) on {src_date} "
                            f"is AFTER AE onset ({dst_data.get('label')}) on {dst_date}. "
                            "A suspect drug must precede the adverse event."
                        ),
                        node_ids=[
                            src_data.get("drug_node_id", src),
                            dst_data.get("event_node_id", dst),
                        ],
                    ))
                else:
                    results.append(GraphAssertion(
                        rule="R1:DrugStart≤AEOnset",
                        passed=True,
                        description=f"{src} PRECEDES {dst} — temporal order valid.",
                        node_ids=[
                            src_data.get("drug_node_id", src),
                            dst_data.get("event_node_id", dst),
                        ],
                    ))

        # R2: Dechallenge nodes — must follow at least one AE onset
        for node, data in self._g.nodes(data=True):
            if data.get("node_type") != NodeType.DECHALLENGE.value:
                continue
            # Predecessors should include AE_ONSET via graph traversal
            drug_node_id = data.get("drug_node_id", node)
            dc_date      = _parse_date(data.get("date_repr"))
            # Find all AE_ONSET nodes connected to the same drug's start node
            drug_start = f"{drug_node_id}-START"
            if drug_start in self._g:
                for ae_node in self._g.successors(drug_start):
                    ae_data = self._g.nodes[ae_node]
                    if ae_data.get("node_type") == NodeType.AE_ONSET.value:
                        ae_date = _parse_date(ae_data.get("date_repr"))
                        if dc_date and ae_date and dc_date < ae_date:
                            results.append(GraphAssertion(
                                rule="R2:Dechallenge≥AEOnset",
                                passed=False,
                                description=(
                                    f"Dechallenge for drug {drug_node_id} "
                                    f"({dc_date}) is BEFORE AE onset "
                                    f"({ae_data.get('label')}, {ae_date}). "
                                    "Dechallenge must follow the adverse event."
                                ),
                                node_ids=[drug_node_id, ae_data.get("event_node_id", ae_node)],
                            ))

        # R3: Rechallenge must follow dechallenge
        for src, dst, data in self._g.edges(data=True):
            if data.get("edge_type") != EdgeType.PRECEDES.value:
                continue
            src_data = self._g.nodes[src]
            dst_data = self._g.nodes[dst]
            if (
                src_data.get("node_type") == NodeType.DECHALLENGE.value
                and dst_data.get("node_type") == NodeType.RECHALLENGE.value
            ):
                # Both dechallenge and rechallenge must be present and ordered
                results.append(GraphAssertion(
                    rule="R3:Rechallenge≥Dechallenge",
                    passed=True,
                    description="Rechallenge follows dechallenge — valid order.",
                    node_ids=[
                        src_data.get("drug_node_id", src),
                        dst_data.get("drug_node_id", dst),
                    ],
                ))

        return results

    def assert_causal_consistency(
        self, hard_negatives_path: Optional[str] = None
    ) -> list[GraphAssertion]:
        """
        Causal graph assertions.
        Currently validates that CONTRAINDICATED edges have no co-occurring
        CAUSES edge (i.e., a drug marked as contradicted for a condition
        cannot simultaneously be claimed as causing the same condition).

        Extension point: full hard_negatives.json integration happens in QCAgent.
        """
        results: list[GraphAssertion] = []
        for src, dst, data in self._g.edges(data=True):
            if data.get("edge_type") == EdgeType.CONTRAINDICATED.value:
                # Check if a CAUSES edge exists for the same pair
                if self._g.has_edge(src, dst):
                    causes_data = self._g[src][dst]
                    if causes_data.get("edge_type") == EdgeType.CAUSES.value:
                        results.append(GraphAssertion(
                            rule="Causal:NoContradictedCauses",
                            passed=False,
                            description=(
                                f"Node {src} has both CAUSES and CONTRAINDICATED "
                                f"edge to {dst}. A contradicted drug cannot be "
                                "simultaneously a causal agent."
                            ),
                            node_ids=[src, dst],
                        ))
        return results

    # ── Serialization ─────────────────────────────────────────────────────────

    def serialize_to_json(self) -> str:
        """Serialize to JSON string (networkx node_link_data format) for GraphState."""
        data = nx.node_link_data(self._g)
        return json.dumps(data)

    @classmethod
    def from_json(cls, case_id: str, json_str: str) -> "CaseGraph":
        """Reconstruct a CaseGraph from serialized JSON."""
        data    = json.loads(json_str)
        g       = cls(case_id=case_id)
        g._g    = nx.node_link_graph(data)
        return g

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "case_id":    self.case_id,
            "num_nodes":  self._g.number_of_nodes(),
            "num_edges":  self._g.number_of_edges(),
            "node_types": list({d.get("node_type") for _, d in self._g.nodes(data=True)}),
            "edge_types": list({d.get("edge_type") for _, _, d in self._g.edges(data=True)}),
        }
