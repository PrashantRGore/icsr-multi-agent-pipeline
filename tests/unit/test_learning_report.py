"""
tests/unit/test_learning_report.py
====================================
Unit tests for scripts/learning_report.py.
Tests report data building and Markdown / JSON rendering
against a real (in-memory fixture) LearningDB.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

from infra.learning_db import LearningDB
from scripts.learning_report import build_report_data, main, render_markdown


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_with_signals(tmp_path):
    """LearningDB populated with 3 representative signals."""
    db = LearningDB(db_path=tmp_path / "learning.db")

    common = dict(
        trace_id              = "TRACE-001",
        review_id             = "REV-001",
        stage                 = "extraction",
        prompt_version        = "v1",
        error_classification  = "OMISSION",
        field_path            = "extracted_entities.suspect_drugs",
        original_value        = "aspirin",
        corrected_value       = "acetylsalicylic acid",
        correction_rationale  = "INN preferred",
        reviewer_id           = "QPPV-01",
        reviewed_at           = "2026-09-10T10:00:00+00:00",
    )

    db.insert_signal(
        agent_id="extraction-agent-v1",
        patient_sex="FEMALE", patient_age_group="ADULT", patient_ethnicity=None,
        **common,
    )
    db.insert_signal(
        agent_id="coding-agent-v1",
        trace_id="TRACE-002", review_id="REV-002",
        stage="coding",
        prompt_version="v1",
        error_classification="CODING_MISMATCH",
        field_path="coded_events.ctcae_code",
        original_value="5.0.1", corrected_value="5.0.2",
        correction_rationale="Wrong grade",
        reviewer_id="QPPV-01",
        reviewed_at="2026-09-11T10:00:00+00:00",
        patient_sex="MALE", patient_age_group="PEDIATRIC", patient_ethnicity="Asian",
    )
    db.insert_signal(
        agent_id="extraction-agent-v1",
        trace_id="TRACE-003", review_id="REV-003",
        stage="extraction",
        prompt_version="v1",
        error_classification="OMISSION",
        field_path="extracted_entities.reporter",
        original_value=None, corrected_value="Dr. Smith",
        correction_rationale="Reporter missing",
        reviewer_id="QPPV-01",
        reviewed_at="2026-09-12T10:00:00+00:00",
        patient_sex=None, patient_age_group="ELDERLY", patient_ethnicity=None,
    )
    return db


@pytest.fixture
def empty_db(tmp_path):
    return LearningDB(db_path=tmp_path / "empty.db")


# ── build_report_data tests ───────────────────────────────────────────────────

class TestBuildReportData:
    def test_total_signals_count(self, db_with_signals):
        data = build_report_data(db_with_signals)
        assert data["total_signals"] == 3

    def test_agent_stats_present(self, db_with_signals):
        data   = build_report_data(db_with_signals)
        agents = {a["agent_id"] for a in data["agent_accuracy"]}
        assert "extraction-agent-v1" in agents
        assert "coding-agent-v1" in agents

    def test_extraction_agent_corrections(self, db_with_signals):
        data = build_report_data(db_with_signals)
        ext  = next(a for a in data["agent_accuracy"]
                    if a["agent_id"] == "extraction-agent-v1")
        assert ext["corrections"] == 2

    def test_correction_rate_calculation(self, db_with_signals):
        data = build_report_data(db_with_signals)
        ext  = next(a for a in data["agent_accuracy"]
                    if a["agent_id"] == "extraction-agent-v1")
        # 2 corrections / 3 total signals = 66.7%
        assert abs(ext["correction_rate_%"] - 66.7) < 1.0

    def test_top_patterns_present(self, db_with_signals):
        data = build_report_data(db_with_signals)
        assert isinstance(data["top_patterns"], list)
        assert len(data["top_patterns"]) > 0

    def test_bias_report_present(self, db_with_signals):
        data = build_report_data(db_with_signals)
        assert isinstance(data["bias_report"], list)
        assert len(data["bias_report"]) > 0

    def test_generated_at_is_iso(self, db_with_signals):
        data = build_report_data(db_with_signals)
        assert "T" in data["generated_at"]    # ISO 8601

    def test_empty_db_zero_signals(self, empty_db):
        data = build_report_data(empty_db)
        assert data["total_signals"] == 0
        # No division-by-zero errors
        for agent in data["agent_accuracy"]:
            assert agent["correction_rate_%"] == 0.0


# ── render_markdown tests ─────────────────────────────────────────────────────

class TestRenderMarkdown:
    def test_markdown_contains_heading(self, db_with_signals):
        data = build_report_data(db_with_signals)
        md   = render_markdown(data)
        assert "# ICSR Pipeline" in md

    def test_markdown_has_agent_table(self, db_with_signals):
        data = build_report_data(db_with_signals)
        md   = render_markdown(data)
        assert "extraction-agent-v1" in md

    def test_markdown_has_bias_section(self, db_with_signals):
        data = build_report_data(db_with_signals)
        md   = render_markdown(data)
        assert "CIOMS WG XIV" in md

    def test_markdown_adr001_reference(self, db_with_signals):
        data = build_report_data(db_with_signals)
        md   = render_markdown(data)
        assert "ADR-001" in md

    def test_empty_db_no_patterns_placeholder(self, empty_db):
        data = build_report_data(empty_db)
        md   = render_markdown(data)
        assert "No patterns recorded" in md


# ── CLI integration tests ─────────────────────────────────────────────────────

class TestCLI:
    def test_cli_md_output_to_stdout(self, db_with_signals, tmp_path, capsys):
        rc = main([
            "--learning-db", str(db_with_signals.db_path),
            "--format", "md",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "# ICSR Pipeline" in out

    def test_cli_json_output(self, db_with_signals, tmp_path, capsys):
        rc = main([
            "--learning-db", str(db_with_signals.db_path),
            "--format", "json",
        ])
        out = capsys.readouterr().out
        assert rc == 0
        parsed = json.loads(out)
        assert "total_signals" in parsed
        assert parsed["total_signals"] == 3

    def test_cli_write_to_file(self, db_with_signals, tmp_path):
        out_file = tmp_path / "report.md"
        rc = main([
            "--learning-db", str(db_with_signals.db_path),
            "--format", "md",
            "--output", str(out_file),
        ])
        assert rc == 0
        assert out_file.exists()
        assert "# ICSR Pipeline" in out_file.read_text()
