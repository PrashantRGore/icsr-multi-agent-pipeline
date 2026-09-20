"""
scripts/generate_psmf.py
=========================
Auto-populates the PSMF template with live data from AuditDB and LearningDB.

Replaces all {{PLACEHOLDER}} tokens in governance/psmf_template.md with
real values derived from the running system, producing a regulation-ready
PSMF component document for EU GVP Module II / CIOMS WG XIV submission.

Usage:
  python scripts/generate_psmf.py
  python scripts/generate_psmf.py --system-owner "Dr. Jane Smith" --output governance/psmf.md
  python scripts/generate_psmf.py --audit-db audit/audit.db --learning-db audit/learning.db

See governance/decisions.md ADR-004 for benchmark data methodology.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infra.audit_db import AuditDB
from infra.learning_db import LearningDB

_TEMPLATE = _ROOT / "governance" / "psmf_template.md"

_AGENT_IDS = [
    "triage-agent-v1",
    "extraction-agent-v1",
    "qc-agent-v1",
    "coding-agent-v1",
    "causality-agent-v1",
    "listedness-agent-v1",
    "narrative-agent-v1",
]

_PLACEHOLDER_AGENT_MAP = {
    "TRIAGE_PASS_RATE":     "triage-agent-v1",
    "EXTRACTION_PASS_RATE": "extraction-agent-v1",
    "QC_PASS_RATE":         "qc-agent-v1",
}


def _read_version() -> str:
    pyproject = _ROOT / "pyproject.toml"
    try:
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return data["project"]["version"]
    except Exception:
        return "0.6.0"


def _get_deployment_date(audit_db: AuditDB) -> str:
    """Return ISO date of the earliest audit log entry, or today."""
    try:
        with audit_db._connect() as conn:
            row = conn.execute(
                "SELECT MIN(timestamp) FROM audit_log"
            ).fetchone()
        if row and row[0]:
            return row[0][:10]
    except Exception:
        pass
    return datetime.now(timezone.utc).date().isoformat()


def _agent_pass_rate(learning_db: LearningDB, agent_id: str, total_signals: int) -> str:
    """
    Derive pass rate from correction rate (ADR-004 methodology).
    pass_rate = 100% - correction_rate
    """
    if total_signals == 0:
        return "N/A (no correction data)"
    acc = learning_db.agent_accuracy(agent_id)
    corrections = acc["total_corrections"]
    rate = round(corrections / total_signals * 100, 1)
    pass_rate = round(100.0 - rate, 1)
    return (
        f"{pass_rate}% "
        f"(derived from {corrections}/{total_signals} corrections — see ADR-004)"
    )


def build_replacements(
    audit_db:    AuditDB,
    learning_db: LearningDB,
    system_owner:  str,
    pytest_results: str,
) -> dict[str, str]:
    version        = _read_version()
    now            = datetime.now(timezone.utc)
    today          = now.date().isoformat()
    deployment_date = _get_deployment_date(audit_db)
    total_signals  = learning_db.count_signals()

    replacements = {
        "VERSION":              version,
        "DEPLOYMENT_DATE":      deployment_date,
        "SYSTEM_OWNER":         system_owner,
        "BENCHMARK_DATE":       today,
        "PYTEST_RESULTS_PATH":  pytest_results,
        "CASE_001_STATUS":      "See audit/audit.db (query trace_id=ICSR-20240315-SYN)",
        "CASE_002_STATUS":      "See audit/audit.db (query trace_id=ICSR-20240402-SYN)",
        "CASE_003_STATUS":      "See audit/audit.db (query trace_id=ICSR-20240520-SYN)",
    }

    # Per-agent pass rates
    for placeholder, agent_id in _PLACEHOLDER_AGENT_MAP.items():
        replacements[placeholder] = _agent_pass_rate(learning_db, agent_id, total_signals)

    # Benchmark paths (placeholder — real paths supplied by CI)
    replacements["TRIAGE_BENCHMARK_PATH"]     = "tests/unit/test_triage_agent.py"
    replacements["EXTRACTION_BENCHMARK_PATH"] = "tests/unit/test_extraction_schema.py"
    replacements["QC_BENCHMARK_PATH"]         = "tests/unit/test_qc_agent.py"

    return replacements


def render(template: str, replacements: dict[str, str]) -> str:
    for key, value in replacements.items():
        template = template.replace(f"{{{{{key}}}}}", value)
    # Report any unreplaced placeholders
    remaining = re.findall(r"\{\{(\w+)\}\}", template)
    if remaining:
        for r in remaining:
            print(f"WARNING: unreplaced placeholder: {{{{{r}}}}}", file=sys.stderr)
    return template


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "generate_psmf",
        description = "Auto-populate PSMF template from live AuditDB / LearningDB data",
    )
    p.add_argument("--audit-db",     default=os.getenv("AUDIT_DB_PATH",    "audit/audit.db"))
    p.add_argument("--learning-db",  default=os.getenv("LEARNING_DB_PATH", "audit/learning.db"))
    p.add_argument("--system-owner", default=os.getenv("SYSTEM_OWNER",     "{{SYSTEM_OWNER}}"),
                   help="Name of the system owner / data scientist for QPPV sign-off")
    p.add_argument("--pytest-results", default="tests/ (pytest)",
                   help="Path or description of pytest results for Section 5.1")
    p.add_argument("--template",     default=str(_TEMPLATE),
                   help="PSMF template path (default: governance/psmf_template.md)")
    p.add_argument("--output",       default=None,
                   help="Output path (default: governance/psmf_<version>_<date>.md)")
    return p


def main(argv: list[str] | None = None) -> int:
    args     = build_parser().parse_args(argv)
    template = Path(args.template).read_text(encoding="utf-8")

    audit_db    = AuditDB(db_path=args.audit_db)
    learning_db = LearningDB(db_path=args.learning_db)

    replacements = build_replacements(
        audit_db       = audit_db,
        learning_db    = learning_db,
        system_owner   = args.system_owner,
        pytest_results = args.pytest_results,
    )
    output_text = render(template, replacements)

    version = replacements.get("VERSION", "0.6.0")
    today   = replacements.get("BENCHMARK_DATE", "unknown")
    out_path = Path(args.output) if args.output else (
        _ROOT / "governance" / f"psmf_{version}_{today}.md"
    )
    out_path.write_text(output_text, encoding="utf-8")
    print(f"PSMF written to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
