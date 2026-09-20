"""
tests/unit/test_e2b_exporter.py
================================
Unit tests for infra/e2b_exporter.py.

Key parsing note:
  The export() method returns a string starting with an XML declaration
  (<?xml version="1.0" encoding="UTF-8"?>). To parse with ElementTree,
  we must encode to bytes so it respects the declared encoding.
  ET.fromstring(xml_str.encode("utf-8")) handles this correctly.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from infra.e2b_exporter import E2BExporter


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_e2b(xml_str: str) -> ET.Element:
    """Parse E2B(R3) XML string. Encodes to bytes so ET respects the declaration."""
    return ET.fromstring(xml_str.encode("utf-8"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_state():
    return {
        "triage_output": {
            "risk_tier":  "TIER_1",
            "is_serious": True,
        },
        "extracted_entities": {
            "patient_sex": "FEMALE",
            "patient_age": "45",
            "suspect_drugs": [
                {"name": "amoxicillin", "rxnorm_cui": "723"},
            ],
        },
        "causality_matrix": {"amoxicillin": "PROBABLE"},
        "coded_events": [
            {"preferred_term": "Anaphylaxis", "ctcae_code": "10002218"},
        ],
        "final_narrative": (
            "A 45-year-old female experienced anaphylaxis following amoxicillin."
        ),
    }


@pytest.fixture
def exporter():
    return E2BExporter(sender_org="TEST-ORG", receiver_org="TEST-REG")


@pytest.fixture
def xml_root(exporter, minimal_state):
    """Parse the E2B export as an ElementTree for structural inspection."""
    xml_str = exporter.export(state=minimal_state, case_id="ICSR-TEST-001")
    return parse_e2b(xml_str)


# ── XML structure tests ───────────────────────────────────────────────────────

class TestXMLStructure:
    def test_xml_is_well_formed(self, exporter, minimal_state):
        xml_str = exporter.export(state=minimal_state, case_id="ICSR-001")
        parse_e2b(xml_str)   # must not raise

    def test_root_element_is_ichicsr(self, xml_root):
        assert xml_root.tag == "ichicsr"

    def test_has_message_header(self, xml_root):
        assert xml_root.find("ichicsrmessageheader") is not None

    def test_message_type_is_ichicsr(self, xml_root):
        hdr   = xml_root.find("ichicsrmessageheader")
        mtype = hdr.find("messagetype")
        assert mtype is not None
        assert mtype.text == "ichicsr"

    def test_has_safety_report(self, xml_root):
        assert xml_root.find("safetyreport") is not None

    def test_safety_report_id(self, xml_root):
        rid = xml_root.find("safetyreport/safetyreportid")
        assert rid is not None
        assert rid.text == "ICSR-TEST-001"

    def test_has_patient_block(self, xml_root):
        assert xml_root.find("safetyreport/patient") is not None

    def test_has_narrative(self, xml_root):
        narr = xml_root.find("safetyreport/narrativeincludeclinical")
        assert narr is not None
        assert "anaphylaxis" in narr.text.lower()

    def test_has_sender(self, xml_root):
        assert xml_root.find("sender") is not None

    def test_has_receiver(self, xml_root):
        assert xml_root.find("receiver") is not None

    def test_sender_org(self, xml_root):
        org = xml_root.find("sender/senderorganization")
        assert org.text == "TEST-ORG"


# ── Seriousness tests ─────────────────────────────────────────────────────────

class TestSeriousness:
    def test_tier1_is_serious(self, xml_root):
        serious = xml_root.find("safetyreport/serious")
        assert serious.text == "1"

    def test_tier3_is_not_serious(self, exporter, minimal_state):
        state = dict(minimal_state)
        state = {**minimal_state, "triage_output": {"risk_tier": "TIER_3"}}
        root  = parse_e2b(exporter.export(state=state, case_id="ICSR-002"))
        assert root.find("safetyreport/serious").text == "2"


# ── Drug block tests ──────────────────────────────────────────────────────────

class TestDrugBlock:
    def test_drug_block_present(self, xml_root):
        assert xml_root.find("safetyreport/patient/drug") is not None

    def test_drug_name(self, xml_root):
        med = xml_root.find("safetyreport/patient/drug/medicinalproduct")
        assert med.text == "amoxicillin"

    def test_drug_causality(self, xml_root):
        causal = xml_root.find("safetyreport/patient/drug/drugcausalityassessment")
        assert causal.text == "PROBABLE"

    def test_drug_characterization_suspect(self, xml_root):
        char = xml_root.find("safetyreport/patient/drug/drugcharacterization")
        assert char.text == "1"


# ── Reaction block and ADR-001 tests ─────────────────────────────────────────

class TestReactionBlock:
    def test_reaction_block_present(self, xml_root):
        assert xml_root.find("safetyreport/patient/reaction") is not None

    def test_primary_source_reaction(self, xml_root):
        psr = xml_root.find("safetyreport/patient/reaction/primarysourcereaction")
        assert psr.text == "Anaphylaxis"

    def test_meddrapt_uses_ctcae_term(self, xml_root):
        mpt = xml_root.find("safetyreport/patient/reaction/reactionmeddrapt")
        assert mpt is not None
        # .strip() needed: ET appends whitespace tail when a Comment child follows
        assert mpt.text.strip() == "Anaphylaxis"

    def test_adr001_comment_present_in_raw_xml(self, exporter, minimal_state):
        xml_str = exporter.export(state=minimal_state, case_id="ICSR-001")
        assert "MedDRA LLT/PT required" in xml_str
        assert "ADR-001" in xml_str

    def test_meddra_mapper_used_when_supplied(self, minimal_state):
        mapper   = lambda term: "10002218"   # noqa: E731
        exporter = E2BExporter(meddra_mapper=mapper,
                               sender_org="T", receiver_org="R")
        root     = parse_e2b(exporter.export(state=minimal_state, case_id="ICSR-003"))
        mpt      = root.find("safetyreport/patient/reaction/reactionmeddrapt")
        assert mpt.text == "10002218"

    def test_empty_coded_events_no_reaction(self, exporter):
        state = {
            "triage_output":      {"risk_tier": "TIER_3"},
            "extracted_entities": {"patient_sex": "MALE"},
            "causality_matrix":   {},
            "coded_events":       [],
            "final_narrative":    "No events.",
        }
        root    = parse_e2b(exporter.export(state=state, case_id="ICSR-004"))
        patient = root.find("safetyreport/patient")
        assert patient.find("reaction") is None


# ── Preamble / declaration tests ──────────────────────────────────────────────

class TestPreamble:
    def test_xml_declaration_present(self, exporter, minimal_state):
        xml_str = exporter.export(state=minimal_state, case_id="ICSR-001")
        assert xml_str.startswith('<?xml version="1.0"')

    def test_adr_reference_in_header_comment(self, exporter, minimal_state):
        xml_str = exporter.export(state=minimal_state, case_id="ICSR-001")
        assert "governance/decisions.md" in xml_str
