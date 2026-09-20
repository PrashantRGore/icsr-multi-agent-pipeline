# Architectural Decision Records (ADR)
# ICSR Multi-Agent HITL Processing & Governance Engine
# =====================================================
# Reference: [ADR-NNN] format — each decision is numbered sequentially.
# These records are living documents. Once "Accepted", status changes only if
# the decision is formally superseded (status → "Superseded by ADR-NNN").
#
# Template:
#   ## ADR-NNN: [Short title]
#   **Date:** YYYY-MM-DD | **Status:** Proposed | Accepted | Superseded
#   **Deciders:** [Role]
#   ### Context … ### Decision … ### Consequences … ### Upgrade Path …

---

## ADR-001: MedDRA Terminology — Exclusion in Favour of Zero-Cost Alternatives

**Date:** 2026-09-16
**Status:** Accepted
**Deciders:** System Architect / QPPV

### Context

MedDRA (Medical Dictionary for Regulatory Activities) is the globally mandated controlled
vocabulary for adverse event coding in ICH E2B(R3) regulatory submissions. It is maintained
by the MedDRA Maintenance and Support Services Organization (MSSO) and is subject to annual
commercial licensing fees payable by any organisation that encodes, stores, or transmits
MedDRA-coded safety data in a production environment.

This project is designed as a **zero-cost, open-source pharmacovigilance engine**. Its core
architectural principle is that the full ICSR pipeline — including adverse event coding,
drug identification, causality assessment, and regulatory report generation — must operate
exclusively using open, royalty-free data sources and terminologies. This principle ensures
the system is accessible to academic institutions, independent researchers, smaller
pharmaceutical companies, and jurisdictions with limited technology budgets.

The terminologies currently in use are:

| Terminology | Source | License |
|---|---|---|
| Ontology of Adverse Events (OAE) | OBO Foundry | CC BY 4.0 |
| NCI CTCAE v5 | National Cancer Institute | US Gov — Public Domain |
| RxNorm | NLM RxNorm REST API | NLM Terms of Service (free, no key) |
| WHO-UMC Causality Scale | World Health Organization | Publicly available |
| FDA DailyMed SPL | FDA | US Gov — Public Domain |

### Decision

MedDRA is **not integrated** in this system. The ICH E2B(R3) XML export uses the CTCAE v5
preferred term as the value of `<reactionmeddrapt>`, accompanied by a structured XML comment
at the element level that:

1. States that the value is a CTCAE term, not a MedDRA LLT/PT code.
2. Instructs the QPPV to complete MedDRA coding before regulatory transmission.
3. Documents the upgrade path for licence holders.

This makes E2B(R3) output suitable for **internal validation, sandbox regulatory submissions,
and QPPV pre-review workflows**. It is not suitable for direct submission to EudraVigilance,
MedWatch, PMDA, or other ICH M2-compliant gateways without manual MedDRA completion.

### Consequences

**Positive:**
- System remains fully zero-cost and open-source.
- No licensing compliance risk for users operating without a MedDRA licence.
- E2B(R3) XML is structurally valid and well-formed; only MedDRA codes are absent.
- QPPV review step (already mandatory by design) is the natural point for MedDRA completion.

**Negative / Limitations:**
- E2B(R3) XML cannot be submitted directly to regulatory authorities without manual QPPV coding.
- Automated MedDRA code validation is not possible.

### Upgrade Path (for MedDRA Licence Holders)

Organisations holding a valid MedDRA licence can integrate MedDRA coding without modifying
core pipeline logic. The `E2BExporter` class (in `infra/e2b_exporter.py`) accepts an
optional `meddra_mapper: Callable[[str], str] | None = None` constructor parameter.
Supplying a mapper function that accepts a CTCAE term and returns a MedDRA LLT code will
produce **fully production-grade, regulatory-submission-ready E2B(R3) XML**.

Example integration point for a licence holder:
```python
from infra.e2b_exporter import E2BExporter
from my_org.meddra import ctcae_to_meddra_llt  # Organisation's licensed mapping

exporter = E2BExporter(meddra_mapper=ctcae_to_meddra_llt)
xml_str  = exporter.export(state=pipeline_state, case_id="ICSR-001")
```

A suitable open mapping source for non-commercial research purposes is the NCI Metathesaurus
CTCAE-to-MedDRA crosswalk, available via UMLS (requires UMLS licence registration — free
for non-commercial use). Integration of this crosswalk is deferred to a dedicated validation
sprint.

---

## ADR-002: Single Uvicorn Worker — Concurrency Constraint

**Date:** 2026-09-12
**Status:** Accepted
**Deciders:** System Architect

### Context

The `OllamaClient` uses a `threading.Semaphore(1)` to serialise all LLM requests. This
prevents GPU memory contention on 16 GB RAM hardware. Multiple Uvicorn workers (separate
processes) would each hold their own `Semaphore(1)`, allowing concurrent LLM calls and
causing OOM errors on constrained hardware.

### Decision

The server is configured with `workers=1` in both the Dockerfile `CMD` and
`hitl/main.py` CLI entry point. This is a deliberate hardware-driven constraint, not a
scalability limitation of the architecture.

### Upgrade Path

For higher-throughput deployments on hardware with more GPU memory (or GPU-partitioned
multi-instance inference), replace `Semaphore(1)` with a proper async task queue
(e.g., Celery + Redis, or a dedicated LLM inference server like vLLM).

---

## ADR-003: SQLite as Primary Storage — No External Database

**Date:** 2026-09-12
**Status:** Accepted
**Deciders:** System Architect

### Context

The zero-cost, local-only constraint (no external API calls for PV data) extends to
storage. External databases (PostgreSQL, MySQL) require running infrastructure and
introduce operational complexity inconsistent with single-machine deployment.

### Decision

All persistence uses SQLite with WAL mode and immutability triggers:
- `audit/audit.db` — immutable audit trail (21 CFR Part 11)
- `audit/auth.db` — reviewer key store (bcrypt-hashed keys)
- `audit/learning.db` — HITL learning signals
- `checkpoints/pipeline.db` — LangGraph checkpointer

### Upgrade Path

All DB classes use a `db_path` constructor parameter. Migration to PostgreSQL requires
only implementing a `PostgresAuditDB` adapter conforming to the same interface —
no changes to pipeline, agent, or route logic.

---

## ADR-004: PSMF Benchmark — Correction-Rate Derived Accuracy

**Date:** 2026-09-16
**Status:** Accepted
**Deciders:** System Architect / QPPV

### Context

The PSMF template includes per-agent benchmark pass rates. A formal benchmark requires
a labelled ICSR test set with ground-truth annotations — a significant effort requiring
clinical expert involvement.

### Decision

For the initial PSMF draft, per-agent "accuracy" is derived from LearningDB correction
rates: if an agent's outputs are corrected in X% of reviewed cases, the reported accuracy
is (100 - X)%. This is a conservative lower bound, as not all HITL corrections represent
agent errors (some are QPPV preference adjustments).

### Upgrade Path

A formal benchmark test suite with ~50 labelled synthetic ICSR cases will be developed
in a dedicated validation sprint, replacing this proxy metric. The PSMF will be updated
accordingly before any regulatory submission.
