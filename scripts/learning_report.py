"""
scripts/learning_report.py
===========================
HITL Learning Pipeline Report Generator.

Reads LearningDB and produces a structured governance report covering:
  1. Per-agent correction rates and top error classifications
  2. Field-level error frequency heatmap
  3. Demographic stratification (CIOMS WG XIV Principle 6 bias monitoring)
  4. Weekly correction trend (model drift detection)

Output formats:
  --format md    (default) — Markdown, suitable for PSMF / Confluence
  --format json             — Machine-readable, suitable for dashboards

Usage:
  python scripts/learning_report.py
  python scripts/learning_report.py --format json
  python scripts/learning_report.py --learning-db audit/learning.db --format md
  python scripts/learning_report.py --output report.md

Note on E2B(R3):
  This report covers system performance metrics and is an internal governance
  document. ICH E2B(R3) XML applies to individual ICSR case transmission, not
  system-level accuracy reports. See governance/decisions.md ADR-001.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from infra.learning_db import LearningDB

# All known agent IDs in pipeline order
_ALL_AGENTS = [
    "triage-agent-v1",
    "extraction-agent-v1",
    "qc-agent-v1",
    "coding-agent-v1",
    "causality-agent-v1",
    "listedness-agent-v1",
    "narrative-agent-v1",
    "secondary-qc-auditor-v1",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data builders
# ─────────────────────────────────────────────────────────────────────────────

def build_report_data(db: LearningDB) -> dict:
    """Aggregate all LearningDB data into a report dict."""
    total_signals = db.count_signals()

    # ── Per-agent accuracy ────────────────────────────────────────────────────
    agent_stats = []
    for agent_id in _ALL_AGENTS:
        acc = db.agent_accuracy(agent_id)
        corrections = acc["total_corrections"]
        top_class = (
            max(acc["by_classification"], key=acc["by_classification"].get)
            if acc["by_classification"] else "N/A"
        )
        # Correction rate = corrections / total_signals (proxy accuracy lower bound)
        rate = round(corrections / total_signals * 100, 1) if total_signals else 0.0
        agent_stats.append({
            "agent_id":          agent_id,
            "corrections":       corrections,
            "correction_rate_%": rate,
            "top_error_class":   top_class,
            "by_classification": acc["by_classification"],
        })

    # ── Top field-level error patterns ───────────────────────────────────────
    top_patterns = db.top_patterns(limit=20)

    # ── Demographic bias ──────────────────────────────────────────────────────
    bias_rows = db.demographic_bias_report()

    return {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "total_signals":  total_signals,
        "agent_accuracy": agent_stats,
        "top_patterns":   top_patterns,
        "bias_report":    bias_rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(data: dict) -> str:
    lines: list[str] = []

    lines += [
        "# ICSR Pipeline — HITL Learning Report",
        f"**Generated:** {data['generated_at']}  ",
        f"**Total correction signals:** {data['total_signals']}  ",
        "",
        "> This report covers system performance metrics and is an internal governance",
        "> document (not an ICH E2B(R3) ICSR submission). See `governance/decisions.md` ADR-001.",
        "",
        "---",
        "",
    ]

    # ── Section 1: Per-agent accuracy ─────────────────────────────────────────
    lines += [
        "## 1. Per-Agent Correction Rates",
        "",
        "Correction rate is computed as corrections / total_signals. "
        "A higher rate indicates more frequent human intervention for that agent "
        "(conservative lower bound on accuracy per ADR-004).",
        "",
        "| Agent | Corrections | Correction Rate | Top Error Class |",
        "|---|---|---|---|",
    ]
    for row in data["agent_accuracy"]:
        lines.append(
            f"| `{row['agent_id']}` | {row['corrections']} "
            f"| {row['correction_rate_%']}% | {row['top_error_class']} |"
        )
    lines.append("")

    # ── Section 2: Field-level error heatmap ──────────────────────────────────
    lines += [
        "## 2. Field-Level Error Frequency",
        "",
        "Top recurring correction patterns across all agents:",
        "",
        "| Rank | Agent | Field Path | Error Class | Count | Last Seen |",
        "|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(data["top_patterns"], start=1):
        field = p.get("field_path") or "—"
        lines.append(
            f"| {i} | `{p['agent_id']}` | `{field}` "
            f"| {p['error_classification']} | {p['count']} | {p['last_seen'][:10]} |"
        )
    if not data["top_patterns"]:
        lines.append("| — | No patterns recorded yet | | | | |")
    lines.append("")

    # ── Section 3: Demographic bias (CIOMS WG XIV Principle 6) ────────────────
    lines += [
        "## 3. Demographic Stratification (CIOMS WG XIV Principle 6)",
        "",
        "Correction counts stratified by patient demographic group. Disproportionate",
        "correction rates in any group may indicate systematic model bias.",
        "",
        "| Agent | Sex | Age Group | Ethnicity | Error Class | Corrections |",
        "|---|---|---|---|---|---|",
    ]
    for row in data["bias_report"]:
        lines.append(
            f"| `{row['agent_id']}` | {row.get('patient_sex') or '—'} "
            f"| {row.get('patient_age_group') or '—'} "
            f"| {row.get('patient_ethnicity') or '—'} "
            f"| {row['error_classification']} | {row['correction_count']} |"
        )
    if not data["bias_report"]:
        lines.append("| — | Insufficient demographic data recorded | | | | |")
    lines.append("")

    lines += [
        "---",
        "",
        "*This report was generated by `scripts/learning_report.py`. "
        "For MedDRA / E2B(R3) considerations, refer to `governance/decisions.md` ADR-001.*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "learning_report",
        description = "Generate HITL learning pipeline governance report",
    )
    p.add_argument(
        "--learning-db",
        default = os.getenv("LEARNING_DB_PATH", "audit/learning.db"),
        metavar = "PATH",
        help    = "Path to learning.db (default: LEARNING_DB_PATH env or audit/learning.db)",
    )
    p.add_argument(
        "--format",
        choices = ["md", "json"],
        default = "md",
        help    = "Output format: md (Markdown) or json (default: md)",
    )
    p.add_argument(
        "--output",
        default = None,
        metavar = "FILE",
        help    = "Write output to FILE instead of stdout",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db   = LearningDB(db_path=args.learning_db)
    data = build_report_data(db)

    if args.format == "json":
        output = json.dumps(data, indent=2, default=str)
    else:
        output = render_markdown(data)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
