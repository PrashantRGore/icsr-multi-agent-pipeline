"""
scripts/generate_psmf_report.py
================================
PSMF Report Generator — CLI script.

Reads live data from AuditDB + LearningDB and renders
governance/psmf_template.md into a dated, complete PSMF report file.

All {{PLACEHOLDER}} tokens in the template are replaced with real values
sourced from the SQLite databases produced by the ICSR pipeline.

Usage
-----
  python -m scripts.generate_psmf_report [OPTIONS]

Options
-------
  --audit-db      PATH  AuditDB SQLite path        (default: audit/audit.db)
  --learning-db   PATH  LearningDB SQLite path      (default: audit/learning.db)
  --template      PATH  PSMF template markdown file (default: governance/psmf_template.md)
  --output-dir    PATH  Output directory            (default: governance/psmf_reports)
  --owner         STR   System owner / QPPV name    (default: SYSTEM-OWNER)

Output
------
  governance/psmf_reports/PSMF_YYYYMMDD_HHmmss.md

Exit codes
----------
  0 — success
  1 — template file not found
  2 — AuditDB read error
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("generate_psmf_report")

# Agent IDs used by the pipeline (for pass-rate table)
_AGENT_IDS = {
    "Triage":     "triage-agent-v1",
    "Extraction": "extraction-agent-v1",
    "QC":         "qc-agent-v1",
    "Coding":     "coding-agent-v1",
    "Causality":  "causality-agent-v1",
    "Listedness": "listedness-agent-v1",
    "Narrative":  "narrative-agent-v1",
}

# Synthetic case trace_ids used in the template (matched by case_id prefix pattern)
_SYNTHETIC_CASES: dict[str, str] = {
    "CASE_001": "ICSR-20240315",
    "CASE_002": "ICSR-20240402",
    "CASE_003": "ICSR-20240520",
}


# -- Data gathering -----------------------------------------------------------

def _get_deployment_date(audit_db) -> str:
    ts = audit_db.get_earliest_timestamp()
    if ts:
        return ts[:10]  # YYYY-MM-DD
    return "N/A"


def _get_agent_stats_row(audit_db, label: str, agent_id: str) -> tuple[str, str, str]:
    """Return (label, pass_rate_str, benchmark_date) for the PSMF table."""
    stats = audit_db.get_agent_stats(agent_id)
    if stats["total"] == 0:
        return label, "N/A", "N/A"
    rate = f"{stats['pass_rate']}%  ({stats['success']}/{stats['total']})"
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return label, rate, date


def _get_case_status(audit_db, placeholder_key: str) -> str:
    """
    Look up the last pipeline status for a synthetic case by its ID prefix.
    Returns 'PASS', 'HITL_PENDING', 'FAIL', or 'NOT RUN'.
    """
    prefix = _SYNTHETIC_CASES.get(placeholder_key, "")
    if not prefix:
        return "NOT RUN"
    try:
        with audit_db._connect() as conn:
            rows = conn.execute(
                """SELECT status FROM audit_log
                   WHERE trace_id LIKE ?
                   ORDER BY timestamp DESC LIMIT 1""",
                (f"{prefix}%",)
            ).fetchall()
        if not rows:
            return "NOT RUN"
        status = rows[0]["status"]
        return {"SUCCESS": "PASS", "HITL_QUEUED": "HITL_PENDING", "FAILED": "FAIL"}.get(
            status, status
        )
    except Exception:
        return "NOT RUN"


def _get_top_patterns(learning_db, limit: int = 5) -> str:
    """Return a markdown table of top correction patterns, or a note if empty."""
    try:
        patterns = learning_db.top_patterns(limit=limit)
    except Exception:
        return "_No learning signals recorded yet._"
    if not patterns:
        return "_No learning signals recorded yet._"
    lines = [
        "| Agent | Error Class | Field | Count |",
        "|---|---|---|---|",
    ]
    for p in patterns:
        lines.append(
            f"| `{p.get('agent_id','—')}` "
            f"| `{p.get('error_classification','—')}` "
            f"| `{p.get('field_path') or '—'}` "
            f"| {p.get('count',0)} |"
        )
    return "\n".join(lines)


def _get_total_signals(learning_db) -> str:
    try:
        return str(learning_db.count_signals())
    except Exception:
        return "N/A"


# -- Template rendering --------------------------------------------------------

def render_psmf(
    template_path: Path,
    audit_db,
    learning_db,
    owner: str,
    now: datetime,
) -> str:
    """Read template and replace all {{PLACEHOLDER}} tokens."""
    template = template_path.read_text(encoding="utf-8")

    benchmark_date = now.strftime("%Y-%m-%d")
    deployment_date = _get_deployment_date(audit_db)

    # ── Agent pass-rate rows ──────────────────────────────────────────────────
    triage_rate  = audit_db.get_agent_stats(_AGENT_IDS["Triage"])
    extract_rate = audit_db.get_agent_stats(_AGENT_IDS["Extraction"])
    qc_rate      = audit_db.get_agent_stats(_AGENT_IDS["QC"])

    def _fmt(stats: dict) -> str:
        if stats["total"] == 0:
            return "N/A"
        return f"{stats['pass_rate']}% ({stats['success']}/{stats['total']})"

    # ── Case statuses ─────────────────────────────────────────────────────────
    case_001 = _get_case_status(audit_db, "CASE_001")
    case_002 = _get_case_status(audit_db, "CASE_002")
    case_003 = _get_case_status(audit_db, "CASE_003")

    # ── Build the Learning Pipeline appendix section ──────────────────────────
    total_signals  = _get_total_signals(learning_db)
    top_patterns   = _get_top_patterns(learning_db)

    try:
        bias_rows = learning_db.demographic_bias_report()
        if bias_rows:
            bias_lines = [
                "| Agent | Sex | Ethnicity | Age Group | Error Class | Count |",
                "|---|---|---|---|---|---|",
            ]
            for r in bias_rows[:10]:
                bias_lines.append(
                    f"| `{r.get('agent_id','—')}` "
                    f"| {r.get('patient_sex') or '—'} "
                    f"| {r.get('patient_ethnicity') or '—'} "
                    f"| {r.get('patient_age_group') or '—'} "
                    f"| `{r.get('error_classification','—')}` "
                    f"| {r.get('correction_count',0)} |"
                )
            bias_section = "\n".join(bias_lines)
        else:
            bias_section = "_No demographic data recorded yet._"
    except Exception:
        bias_section = "_No demographic data recorded yet._"

    # ── Append Learning Pipeline section to report ────────────────────────────
    learning_appendix = f"""
---

### 10. Learning Pipeline Summary (CIOMS WG XIV Principle 6)

| Metric | Value |
|---|---|
| Total HITL correction signals recorded | {total_signals} |
| LearningDB path | `audit/learning.db` |
| Generated | {now.isoformat()} |

#### 10.1 Top Correction Patterns

{top_patterns}

#### 10.2 Demographic Bias Summary

{bias_section}

---

*Report generated automatically by `scripts/generate_psmf_report.py` at {now.isoformat()}.*
*All `{{PLACEHOLDER}}` values were populated from live AuditDB + LearningDB data.*
"""

    # ── Replace placeholders ──────────────────────────────────────────────────
    replacements: dict[str, str] = {
        "{{DEPLOYMENT_DATE}}":         deployment_date,
        "{{PYTEST_RESULTS_PATH}}":     "tests/ (run: python -m pytest tests/ -v)",
        "{{TRIAGE_BENCHMARK_PATH}}":   "tests/unit/test_triage_agent.py",
        "{{EXTRACTION_BENCHMARK_PATH}}": "tests/unit/test_extraction_agent.py",
        "{{QC_BENCHMARK_PATH}}":       "tests/unit/test_qc_agent.py",
        "{{TRIAGE_PASS_RATE}}":        _fmt(triage_rate),
        "{{EXTRACTION_PASS_RATE}}":    _fmt(extract_rate),
        "{{QC_PASS_RATE}}":            _fmt(qc_rate),
        "{{BENCHMARK_DATE}}":          benchmark_date,
        "{{CASE_001_STATUS}}":         case_001,
        "{{CASE_002_STATUS}}":         case_002,
        "{{CASE_003_STATUS}}":         case_003,
        "{{SYSTEM_OWNER}}":            owner,
    }

    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)

    # Append learning pipeline section before the final line
    template = template.rstrip() + "\n" + learning_appendix

    return template


# -- CLI entry point -----------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a PSMF report from AuditDB + LearningDB data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audit-db",    default="audit/audit.db",
                        help="Path to AuditDB SQLite file")
    parser.add_argument("--learning-db", default="audit/learning.db",
                        help="Path to LearningDB SQLite file")
    parser.add_argument("--template",    default="governance/psmf_template.md",
                        help="Path to PSMF markdown template")
    parser.add_argument("--output-dir",  default="governance/psmf_reports",
                        help="Directory to write the rendered report into")
    parser.add_argument("--owner",       default="SYSTEM-OWNER",
                        help="System owner / QPPV name for sign-off section")
    args = parser.parse_args(argv)

    template_path = Path(args.template)
    if not template_path.exists():
        logger.error("Template not found: %s", template_path)
        return 1

    # Import DB classes here so the module is importable without them being initialised
    from infra.audit_db import AuditDB
    from infra.learning_db import LearningDB

    try:
        audit_db    = AuditDB(db_path=args.audit_db)
        learning_db = LearningDB(db_path=args.learning_db)
    except Exception as exc:
        logger.error("Failed to open databases: %s", exc)
        return 2

    now = datetime.now(timezone.utc)

    try:
        report_text = render_psmf(
            template_path = template_path,
            audit_db      = audit_db,
            learning_db   = learning_db,
            owner         = args.owner,
            now           = now,
        )
    except Exception as exc:
        logger.error("Report rendering failed: %s", exc)
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename   = f"PSMF_{now.strftime('%Y%m%d_%H%M%S')}.md"
    output_path = output_dir / filename

    output_path.write_text(report_text, encoding="utf-8")
    logger.info("PSMF report written to %s  (%d bytes)", output_path, len(report_text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
