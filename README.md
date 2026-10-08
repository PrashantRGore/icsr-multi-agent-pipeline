# ICSR Engine — Zero-Cost Multi-Agent Pharmacovigilance Pipeline

> **v0.6.0** · Zero-cost · Privacy-first · Runs entirely on your hardware

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-4A154B)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-black?logo=ollama)](https://ollama.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Tests](https://img.shields.io/badge/Tests-447%20passing-brightgreen?logo=pytest)](./tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)
[![Security](https://img.shields.io/badge/Secrets-None%20Committed-brightgreen)](#security)

An end-to-end **Individual Case Safety Report (ICSR)** processing pipeline built on a 7-agent LangGraph architecture with Human-in-the-Loop (HITL) review, audit trail controls, and ICH E2B(R3) XML export — all running **locally, at zero ongoing cost**, using open-source LLMs via Ollama.

> [!CAUTION]
> **Research and Portfolio Prototype — Not for Production Use with Real Patient Data**
>
> This repository is a **research and portfolio prototype**. It must **not** be used with real patient data or for regulatory submission without:
> - Appropriate independent validation and security assessment
> - Production-grade security controls (TLS, network isolation, enterprise IAM)
> - Required organisational and regulatory authorisation
> - A valid MedDRA licence (required for ICH E2B(R3) regulatory transmission — see [ADR-001](./governance/decisions.md))
>
> See the [LICENSE](./LICENSE) for the full disclaimer and the [Security](#security) section for deployment guidance.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Option A: Docker (Recommended)](#option-a-docker-recommended)
  - [Option B: Local Development](#option-b-local-development)
- [First-Time Setup](#first-time-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Directory Structure](#directory-structure)
- [Governance & Compliance](#governance--compliance)
- [Testing](#testing)
- [Known Limitations & Upgrade Paths](#known-limitations--upgrade-paths)
- [Security](#security)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Overview

The **ICSR Engine** automates the end-to-end processing of individual case safety reports for pharmacovigilance workflows. It ingests unstructured adverse event narratives, processes them through a deterministic 7-agent pipeline, queues uncertain cases for QPPV human review, and exports finalized reports as ICH E2B(R3) XML — the globally mandated interchange format for regulatory submissions to EudraVigilance, MedWatch, and PMDA.

The project is designed with a single non-negotiable constraint: **zero recurring cost**. Every component — the LLM, terminologies, databases, and APIs — uses open-source or royalty-free resources that can be self-hosted on standard hardware (16 GB RAM).

---

## Key Features

| Feature | Description |
|---|---|
| 🤖 **7-Agent LangGraph Pipeline** | Extraction → RxNorm → Triage → Coding → Causality → Narrative → QC Auditor |
| 🔒 **Audit Trail Controls** | Append-only SQLite `audit_log` with UPDATE/DELETE triggers and content hashing; inspired by 21 CFR Part 11 controls — see [governance/decisions.md](./governance/decisions.md) |
| 👁️ **HITL Review API** | FastAPI server with `X-API-Key` auth; QPPV review, correction, and sign-off |
| 📄 **ICH E2B(R3) XML Export** | E2B(R3)-structured XML with CTCAE-coded reactions and ADR-001 MedDRA note |
| 🏷️ **Adverse Event Coding** | CTCAE v5 + OAE (CC BY 4.0) via FAISS semantic search |
| 💊 **Drug Normalisation** | RxNorm REST API (NLM) — no API key, rate-limited at 85 req/min |
| 🔐 **PII De-identification** | Microsoft Presidio + spaCy `en_core_web_lg`; Fernet-encrypted entity maps |
| 🔑 **Column-Level Encryption** | Fernet AES-128 on sensitive audit columns; optional, off by default |
| 📊 **Prometheus Metrics** | `/metrics` endpoint with case counts, latency histograms, HITL queue depth |
| 📈 **Learning Feedback Loop** | LearningDB captures HITL corrections; per-agent accuracy & bias reports |
| 📋 **PSMF Auto-generation** | Script populates Pharmacovigilance System Master File template from live data |
| 🐳 **Docker Compose** | Two-service stack (Ollama + ICSR HITL); named volumes; GPU-ready |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ICSR Engine — System Overview                    │
└─────────────────────────────────────────────────────────────────────┘

  Adverse Event
  Narrative (text)
       │
       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline (7 Agents)                  │
│                                                                   │
│  ① Extraction ──► ② RxNorm ──► ③ Triage ──► ④ Coding           │
│       │                             │             │               │
│  (entities)              (risk tier)       (CTCAE/OAE)           │
│                                                   │               │
│                  ⑤ Causality ◄───────────────────┘               │
│                       │                                           │
│                  (WHO-UMC scale)                                  │
│                       │                                           │
│                  ⑥ Narrative                                      │
│                       │                                           │
│                  (CIOMS-I text)                                   │
│                       │                                           │
│                  ⑦ QC Auditor                                     │
│                       │                                           │
│          ┌────────────┴────────────┐                              │
│      PASS (≥threshold)        HITL_QUEUED                         │
└──────────┼─────────────────────────┼──────────────────────────────┘
           │                         │
           ▼                         ▼
    ┌──────────────┐         ┌──────────────────┐
    │  AuditDB     │         │  HITL Review API  │
    │  (SQLite,    │         │  (FastAPI :8000)  │
    │  immutable)  │         │                  │
    └──────┬───────┘         │  QPPV Review ──► │
           │                 │  Correction       │
           │                 │  Sign-off         │
           │                 └────────┬─────────┘
           │                          │
           │◄─────────────────────────┘
           │           (approved case written to AuditDB)
           │
           ▼
    ┌──────────────────────────────────────────┐
    │  Export                                  │
    │  GET /api/v1/cases/{id}/export/e2b  →   │
    │  ICH E2B(R3) XML  (regulatory ready*)   │
    │  GET /api/v1/cases/{id}/export/json →   │
    │  Structured JSON (ICH field names)       │
    └──────────────────────────────────────────┘

  * MedDRA LLT/PT coding required before regulatory submission (ADR-001)
```

### Agent Responsibilities

| # | Agent | Model Input | Output |
|---|---|---|---|
| ① | **Extraction** | Raw narrative | `extracted_entities` (patient, drugs, events) |
| ② | **RxNorm** | Drug names | RxNorm CUI + normalized name |
| ③ | **Triage** | Entities + drugs | `risk_tier` (TIER_1/2/3) + `is_serious` flag |
| ④ | **Coding** | Event terms | CTCAE v5 / OAE preferred terms + codes |
| ⑤ | **Causality** | Drug + events | WHO-UMC causality scale per drug |
| ⑥ | **Narrative** | All prior outputs | CIOMS-I formatted narrative |
| ⑦ | **QC Auditor** | Full pipeline state | Pass / HITL queue decision |

---

## Technology Stack

| Layer | Technology | Licence / Cost |
|---|---|---|
| **LLM** | Llama 3.1 8B Instruct Q4_K_M via [Ollama](https://ollama.com/) | Meta LLAMA 3 Community · **Free** |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | MIT · **Free** |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | MIT · **Free** |
| **AE Coding** | CTCAE v5 (US Gov public domain) + OAE (CC BY 4.0) | **Free** |
| **Drug Coding** | [RxNorm REST API](https://rxnav.nlm.nih.gov/) (NLM) | NLM free-use · **Free** |
| **Causality** | WHO-UMC scale (publicly available) | **Free** |
| **Vector Search** | [FAISS](https://github.com/facebookresearch/faiss) + `sentence-transformers` | MIT · **Free** |
| **PII Protection** | [Microsoft Presidio](https://github.com/microsoft/presidio) | MIT · **Free** |
| **Databases** | SQLite (AuditDB · AuthDB · LearningDB) | Public domain · **Free** |
| **Encryption** | [cryptography](https://cryptography.io/) (Fernet / AES-128) | Apache 2.0 · **Free** |
| **Auth** | bcrypt API-key hashing | Apache 2.0 · **Free** |
| **Observability** | Prometheus-format `/metrics` | Apache 2.0 · **Free** |
| **Containers** | Docker + Docker Compose | Apache 2.0 · **Free** |

---

## Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| RAM | **16 GB** | Llama 3.1 8B Q4_K_M fits comfortably; 8 GB may work at reduced parallelism |
| Disk | **10 GB** | ~5 GB model weights + Python env + FAISS indices |
| Python | **3.11+** | Tested on 3.11 and 3.13 |
| Docker | **24+** | Required for the Docker path; optional for local dev |
| Ollama | **0.3+** | Install from [ollama.com](https://ollama.com/download) |
| OS | Linux / macOS / Windows (WSL2) | Native Windows via PowerShell also supported |

---

## Quick Start

### Option A: Docker (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/PrashantRGore/icsr-multi-agent-pipeline.git
cd icsr-multi-agent-pipeline

# 2. Copy environment template
cp .env.example .env
# Edit .env — at minimum, generate and set DB_ENCRYPTION_KEY (see Configuration)

# 3. Start the full stack (must start before running docker_init)
docker compose up -d

# 4. First-time initialisation (pulls Ollama model + creates QPPV reviewer)
#    Requires the stack to be running (uses docker compose exec internally)
#    Linux / macOS:
./scripts/docker_init.sh
#    Windows (PowerShell):
.\scripts\docker_init.ps1

# 5. Verify the service is healthy
curl http://localhost:8000/health
# Expected output (healthy, Ollama running):
# {
#   "status": "healthy",
#   "version": "0.6.0",
#   "checks": {
#     "audit_db": {"status": "ok", "entry_count": 0},
#     "auth_db":  {"status": "ok", "reviewer_count": 1},
#     "hitl_queue": {"status": "ok", "pending": 0},
#     "ollama":   {"status": "ok", "model": "llama3.1:8b-instruct-q4_K_M", "latency_ms": 42.1}
#   },
#   "encryption_active": true,
#   "pii_deidentifier_active": true
# }
# Note: status="degraded" (not 200→5503) if Ollama is unreachable but all other checks pass.
```

The stack exposes:
- `http://localhost:8000` — HITL Review API (localhost only by default)
- Ollama is **internal** to Docker — not reachable from the host

### Option B: Local Development

```bash
# 1. Clone and create virtual environment
git clone https://github.com/PrashantRGore/icsr-multi-agent-pipeline.git
cd icsr-multi-agent-pipeline
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -e .

# 3. Install spaCy model (required by Presidio)
python -m spacy download en_core_web_lg

# 4. Install and start Ollama, then pull the model
ollama pull llama3.1:8b-instruct-q4_K_M

# 5. Build FAISS terminology indices
python scripts/build_ctcae_index.py
python scripts/build_oae_index.py

# 6. Copy and configure environment
cp .env.example .env

# 7. Create your first reviewer
python scripts/manage_reviewers.py add --reviewer-id QPPV-01 --role QPPV
# Output: api_key: <your-key>   ← store this securely

# 8. Start the HITL API server
uvicorn hitl.main:app --host 0.0.0.0 --port 8000 --workers 1

# 9. Verify
curl http://localhost:8000/health
```

---

## First-Time Setup

### Generate an Encryption Key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Copy the output into `DB_ENCRYPTION_KEY` in your `.env` file.

> ⚠️ **Warning:** This key encrypts sensitive audit columns. Losing it makes that data **unrecoverable**. Back up your `.env` separately from your database volumes.

### Reviewer Management

```bash
# Add a new reviewer
python scripts/manage_reviewers.py add --reviewer-id QPPV-01 --role QPPV

# List all active reviewers
python scripts/manage_reviewers.py list

# Rotate a reviewer's API key (issues new key, invalidates old one)
python scripts/manage_reviewers.py rotate --reviewer-id QPPV-01

# Deactivate a reviewer (soft-delete)
python scripts/manage_reviewers.py deactivate --reviewer-id QPPV-01
```

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Model to use for all agents |
| `OLLAMA_TEMPERATURE` | `0.0` | Deterministic output (recommended for ICSR) |
| `AUDIT_DB_PATH` | `audit/audit.db` | Append-only audit log (Part 11-inspired controls) |
| `AUTH_DB_PATH` | `audit/auth.db` | Reviewer API key store (bcrypt-hashed) |
| `LEARNING_DB_PATH` | `audit/learning.db` | HITL correction signal store |
| `DB_ENCRYPTION_KEY` | *(empty)* | Fernet key for column-level encryption — **set in production** |
| `PII_ENCRYPTION_KEY` | *(empty)* | Fernet key for PII entity maps |
| `SYSTEM_OWNER` | *(empty)* | QPPV name for PSMF generation |
| `E2B_SENDER_ORG` | `ICSR-HITL-SYSTEM` | Sender organisation in E2B(R3) XML |
| `E2B_RECEIVER_ORG` | `REGULATORY-AUTHORITY` | Receiver (regulatory body) in E2B(R3) XML |
| `CONFIDENCE_THRESHOLD_TIER1` | `0.95` | Min confidence for auto-pass (TIER_1 cases) |
| `CONFIDENCE_THRESHOLD_TIER2` | `0.85` | Min confidence for auto-pass (TIER_2 cases) |
| `CONFIDENCE_THRESHOLD_TIER3` | `0.75` | Min confidence for auto-pass (TIER_3 cases) |

### GPU Support (Optional)

The stack runs on CPU by default (appropriate for 16 GB RAM). To enable NVIDIA GPU acceleration for Ollama, uncomment the `deploy` block in `docker-compose.yml` (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)).

---

## API Reference

All endpoints require `X-API-Key: <reviewer-key>` unless stated otherwise.

### Case Submission & Review

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/cases/run` | Submit a new adverse event narrative for processing |
| `GET` | `/api/v1/review/queue` | List all cases currently waiting for human review |
| `GET` | `/api/v1/review/{review_id}` | Get full state snapshot for one queued case |
| `POST` | `/api/v1/review/{review_id}/submit` | Submit QPPV correction and sign-off |

### Export

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/cases/{case_id}/export/e2b` | Download ICH E2B(R3) XML (completed cases only) |
| `GET` | `/api/v1/cases/{case_id}/export/json` | Download structured JSON with ICH field names |

### Learning & Governance Reports

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/learning/report` | Full governance report (agent accuracy, field errors, bias) |
| `GET` | `/api/v1/learning/agents` | Per-agent correction rate summary |
| `GET` | `/api/v1/learning/bias` | Demographic stratification table (CIOMS WG XIV Principle 6) |

### Observability

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | None | Service health + HITL queue depth |
| `GET` | `/metrics` | `X-API-Key` | Prometheus-format metrics |

### Example: Submit a Case

```bash
curl -X POST http://localhost:8000/api/v1/cases/run \
  -H "X-API-Key: <your-reviewer-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "case_id": "ICSR-2026-001",
    "raw_narrative": "A 58-year-old female patient developed anaphylaxis within 30 minutes of receiving amoxicillin 500mg orally. She was hospitalised for 2 days. Previous penicillin allergy noted in records.",
    "source_type": "spontaneous",
    "country": "GB",
    "received_date": "2026-10-01"
  }'
```

### Example: Export E2B(R3) XML

```bash
curl http://localhost:8000/api/v1/cases/ICSR-2026-001/export/e2b \
  -H "X-API-Key: <your-reviewer-key>" \
  -o ICSR-2026-001.xml

# Validate the output is well-formed
python -c "import xml.etree.ElementTree as ET; ET.parse('ICSR-2026-001.xml'); print('Valid E2B(R3) XML')"
```

---

## Directory Structure

```
icsr-engine/
│
├── agents/                     # 7 LangGraph agent definitions
│   ├── extraction_agent.py     # Entity extraction (patient, drug, event)
│   ├── rxnorm_client.py        # Drug normalisation via NLM RxNorm REST
│   ├── triage_agent.py         # Risk tier + seriousness classification
│   ├── coding_agent.py         # CTCAE/OAE adverse event coding
│   ├── causality_agent.py      # WHO-UMC causality assessment
│   ├── narrative_agent.py      # CIOMS-I narrative generation
│   └── secondary_qc_auditor.py # Pipeline output QC + HITL routing
│
├── graph/                      # LangGraph pipeline assembly
│   ├── pipeline.py             # Compiled graph + run() entrypoint
│   ├── nodes.py                # NodeFactory — agent wiring
│   └── hitl_interrupt.py       # HITL interrupt + SqliteSaver checkpoint
│
├── hitl/                       # HITL Review API (FastAPI)
│   ├── main.py                 # App factory + lifespan + router registration
│   ├── auth.py                 # X-API-Key dependency (bcrypt verification)
│   └── routes/
│       ├── review.py           # Case submission, HITL review, correction
│       ├── export.py           # E2B(R3) XML + JSON export
│       ├── learning.py         # Governance report endpoints
│       └── metrics.py          # Prometheus /metrics endpoint
│
├── infra/                      # Infrastructure layer
│   ├── audit_db.py             # Append-only AuditDB (Part 11-inspired controls)
│   ├── auth_db.py              # Reviewer key management (bcrypt)
│   ├── learning_db.py          # HITL correction signal store
│   ├── encrypted_db.py         # Fernet column-level encryption mixin
│   ├── pii_deidentifier.py     # Presidio PII de-identification
│   └── e2b_exporter.py         # ICH E2B(R3) XML serializer
│
├── schemas/                    # Pydantic data models
│   ├── audit.py                # AuditLogEntry, HITLReviewRecord, ...
│   └── icsr.py                 # ICSR pipeline state schema
│
├── scripts/                    # Operational CLI scripts
│   ├── manage_reviewers.py     # QPPV reviewer lifecycle management
│   ├── learning_report.py      # CLI: Markdown/JSON governance report
│   ├── generate_psmf.py        # CLI: PSMF template auto-population
│   ├── build_ctcae_index.py    # Build CTCAE FAISS index from NCI source
│   ├── build_oae_index.py      # Build OAE FAISS index from OWL ontology
│   ├── docker_init.sh          # First-time init (Linux/macOS)
│   └── docker_init.ps1         # First-time init (Windows)
│
├── governance/
│   ├── decisions.md            # Architectural Decision Record (ADR) registry
│   └── psmf_template.md        # PSMF v0.6.0 template
│
├── tests/
│   ├── unit/                   # Unit tests (pytest, no network/LLM)
│   └── integration/            # Integration tests (TestClient, real SQLite)
│
├── Dockerfile                  # Multi-stage build (builder + runtime)
├── docker-compose.yml          # Production stack (ollama + icsr-hitl)
├── docker-compose.dev.yml      # Dev overrides (volume mounts, hot reload)
├── pyproject.toml              # Project metadata + dependencies
└── .env.example                # Environment variable template
```

---

## Governance & Compliance

### 21 CFR Part 11 Alignment

The audit database (`infra/audit_db.py`) implements the following controls:

- **Immutability** — SQLite `BEFORE UPDATE` and `BEFORE DELETE` triggers raise `ABORT` with the message `21CFR11: audit_log rows are immutable`. No row can be modified once written.
- **Content Hashing** — Every entry carries a SHA-256 hash of the serialised agent output. Tamper detection is performed at retrieval time.
- **Timestamps** — All timestamps are UTC ISO-8601 (`strftime('%Y-%m-%dT%H:%M:%SZ', 'now')`).
- **Human Review Log** — A separate `hitl_review_log` table (also append-only) records every reviewer action with reviewer ID, timestamp, original value, corrected value, and rationale.

### Architectural Decision Records

Key decisions are documented in [`governance/decisions.md`](./governance/decisions.md):

| ADR | Decision | Rationale |
|---|---|---|
| **ADR-001** | Use CTCAE v5 terms as `<reactionmeddrapt>` placeholder | MedDRA requires a commercial licence; zero-cost constraint |
| **ADR-002** | Single Uvicorn worker (`--workers 1`) | `asyncio.Semaphore(1)` on OllamaClient; prevents race conditions |
| **ADR-003** | SQLite for all databases | Self-contained; no external DB server; 16 GB RAM target |
| **ADR-004** | PSMF benchmark derived from correction rates | Full agent benchmark deferred; correction rate is a valid proxy |

### MedDRA Note (ADR-001)

E2B(R3) exports produced by this system use CTCAE v5 preferred terms in the `<reactionmeddrapt>` field. A structured XML comment is embedded at the element level flagging that MedDRA coding is required before regulatory transmission:

```xml
<!-- MedDRA LLT/PT required for regulatory submission.
     Current value is CTCAE v5 preferred term.
     See governance/decisions.md ADR-001 for rationale and upgrade path. -->
<reactionmeddrapt>Anaphylaxis</reactionmeddrapt>
```

Organisations holding a valid [MedDRA licence](https://www.meddra.org/) can supply a `meddra_mapper: Callable[[str], str]` to `E2BExporter.__init__()` to enable full production-grade MedDRA coding without any structural change to the pipeline.

### Pharmacovigilance System Master File (PSMF)

A PSMF template is maintained at [`governance/psmf_template.md`](./governance/psmf_template.md). To generate a dated, auto-populated PSMF from live database data:

```bash
python scripts/generate_psmf.py --system-owner "Dr. Smith, QPPV"
# Writes: governance/psmf_0.6.0_<date>.md
```

### Governance Reports

```bash
# Markdown report (suitable for PSMF appendices or Confluence)
python scripts/learning_report.py --format md

# Machine-readable JSON (suitable for dashboards)
python scripts/learning_report.py --format json
```

Report sections:
1. **Per-agent accuracy** — correction rate per `agent_id`
2. **Field-level error frequency** — which output fields are corrected most often
3. **Demographic stratification** — correction rates by sex, age group, ethnicity (CIOMS WG XIV Principle 6 bias monitoring)
4. **Weekly trend** — correction count per ISO week (model drift detection)

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -q

# Run unit tests only (no network, no LLM)
python -m pytest tests/unit/ -v

# Run integration tests only
python -m pytest tests/integration/ -v

# Run a specific module
python -m pytest tests/unit/test_e2b_exporter.py -v
```

**Current status:** `447 passed, 0 failed` (Python 3.13.5, pytest 9.0.2)

Test counts are generated from `pytest --collect-only -q`. The table lists every test file; the overall total is 447.

| Test File | Tests | Focus |
|---|---|---|
| `unit/test_extraction_agent` | 10 | Entity extraction accuracy |
| `unit/test_extraction_schema` | 20 | Pydantic schema validation |
| `unit/test_triage_agent` | 9 | Risk tier + seriousness logic |
| `unit/test_triage_schema` | 11 | Triage schema edge cases |
| `unit/test_coding_agent` | 8 | CTCAE/OAE code mapping |
| `unit/test_causality_agent` | 6 | WHO-UMC scale assignment |
| `unit/test_causality_schema` | 11 | Causality schema validation |
| `unit/test_narrative_agent` | 8 | CIOMS-I narrative generation |
| `unit/test_qc_agent` | 7 | QC auditor routing logic |
| `unit/test_qc_schema` | 10 | QC schema validation |
| `unit/test_listedness_agent` | 7 | Listedness evaluation logic |
| `unit/test_audit_db` | 13 | Append-only trigger enforcement + hashing |
| `unit/test_audit_db_queue` | 11 | HITL queue persistence |
| `unit/test_auth_db` | 16 | Reviewer key lifecycle |
| `unit/test_encrypted_db` | 18 | Fernet column encryption |
| `unit/test_learning_db` | — | *(no unit tests; covered by integration)* |
| `unit/test_learning_pipeline` | 31 | End-to-end learning pipeline |
| `unit/test_learning_report` | 16 | Markdown + JSON report generation |
| `unit/test_e2b_exporter` | 25 | E2B(R3) XML structure + ADR-001 |
| `unit/test_manage_reviewers` | 18 | CLI: add, list, rotate, deactivate |
| `unit/test_pii_deidentifier` | 15 | Presidio PII detection + encryption |
| `unit/test_rxnorm_client` | 10 | Drug normalisation client |
| `unit/test_ollama_client` | 13 | LLM client + semaphore logic |
| `unit/test_case_graph` | 12 | LangGraph pipeline compilation |
| `unit/test_graph_state` | 11 | GraphState schema transitions |
| `unit/test_json_formatter` | 19 | Structured log formatter |
| `unit/test_log_filter` | 13 | PII log filter |
| `unit/test_metrics` | 24 | Prometheus metrics counters |
| `integration/test_health` | 9 | API health, auth, queue endpoints |
| `integration/test_hitl_api` | 14 | Full HITL review workflow |
| `integration/test_export_endpoint` | 12 | E2B XML + JSON export API |
| `integration/test_learning_endpoint` | 9 | Learning report API endpoints |
| `integration/test_pipeline_case001` | 11 | Pipeline: anaphylaxis case |
| `integration/test_pipeline_case002` | 10 | Pipeline: hepatotoxicity case |
| `integration/test_pipeline_case003` | 10 | Pipeline: paediatric case |

---

## Known Limitations & Upgrade Paths

| Limitation | Impact | Upgrade Path |
|---|---|---|
| **MedDRA not integrated** (ADR-001) | E2B(R3) exports require manual MedDRA coding before regulatory submission | Inject a `meddra_mapper` callable into `E2BExporter` using the NCI CTCAE→MedDRA crosswalk (requires UMLS + MedDRA licences) |
| **Single Uvicorn worker** (ADR-002) | Throughput limited to one in-flight ICSR at a time | Replace `asyncio.Semaphore(1)` with a worker pool or deploy multiple containers behind a load balancer |
| **SQLite databases** (ADR-003) | Not suitable for multi-node deployments or >100k cases | Migrate `AuditDB`, `AuthDB`, `LearningDB` to PostgreSQL with the same schema |
| **CPU-only inference** | Llama 3.1 8B generates at ~15–25 tok/s on modern x86 | Enable NVIDIA GPU in `docker-compose.yml` (uncomment `deploy` block) |
| **FAISS index is static** | New CTCAE/OAE terms require index rebuild | Schedule `build_ctcae_index.py` / `build_oae_index.py` as periodic jobs |
| **No E2B(R3) gateway** | XML is generated locally; electronic submission to EudraVigilance/MedWatch requires a certified gateway | Integrate [EVWEB](https://eudravigilance.ema.europa.eu/) or an E2B gateway provider |

---

## Security

> **Before pushing to a public repository, verify the following.**

| Item | Status | Notes |
|---|---|---|
| `.gitignore` excludes `audit/*.db` | ✅ | `auth.db` contains bcrypt-hashed reviewer keys — never commit |
| `.gitignore` excludes `data/pii_maps/` | ✅ | May contain patient entity maps from live runs |
| `.gitignore` excludes `checkpoints/` | ✅ | Contains pipeline state snapshots |
| `.gitignore` excludes `.env` | ✅ | Only `.env.example` (with blank sensitive fields) is committed |
| No hardcoded API keys in source | ✅ | All secrets loaded via `os.getenv()` |
| No paid external APIs called at runtime | ✅ | Only RxNorm (NLM — free) and local Ollama |
| `DB_ENCRYPTION_KEY` is blank in `.env.example` | ✅ | Users generate their own key locally |
| `APP_ENV=production` enforces encryption | ✅ | Server refuses to start if keys are absent |
| Ollama not exposed to host network | ✅ | Accessible only on Docker internal network |
| API bound to `127.0.0.1:8000` by default | ✅ | Local-only; for remote access use TLS + reverse proxy (see `docker-compose.yml`) |
| `LICENSE` file present | ✅ | MIT with full third-party notices |

### What should never be committed

```
audit/audit.db       ← operational audit log (test or real data)
audit/auth.db        ← bcrypt-hashed reviewer API keys
audit/learning.db    ← HITL correction signals
data/rxnorm_cache.db ← cached drug lookups
data/pii_maps/       ← patient PII entity maps
checkpoints/         ← LangGraph pipeline state
.env                 ← your real encryption keys and config
```

### Note on `TEST_API_KEY` in test files

`tests/integration/test_hitl_api.py` contains:
```python
TEST_API_KEY = "test-api-key-for-integration-tests"
```
This is a **deliberate, synthetic placeholder** — it is not a real API key and has no associated account or billing. It is inserted into a temporary in-memory test database for the duration of each test run and has no relation to any external service.

---

## Acknowledgements

**Terminologies and Data Sources**

| Resource | Provider | Licence |
|---|---|---|
| NCI CTCAE v5 | U.S. National Cancer Institute | US Gov — Public Domain |
| OAE (Ontology of Adverse Events) | OBO Foundry | CC BY 4.0 — He Y et al., *J Biomed Semantics* 2014 |
| RxNorm | U.S. National Library of Medicine | NLM Terms of Service (free) |
| WHO-UMC Causality Scale | Uppsala Monitoring Centre | Research/demo use — see note below |
| FDA DailyMed SPL | U.S. FDA / NLM | US Gov — Public Domain |

> **RxNorm/NLM attribution (required by [RxNav Terms of Service](https://lhncbc.nlm.nih.gov/RxNav/TermsofService.html)):**
> "This product uses publicly available data from the U.S. National Library of Medicine (NLM), National Institutes of Health, Department of Health and Human Services; NLM is not responsible for the product and does not endorse or recommend this or any other product."

> **WHO-UMC note:** The causality-assessment implementation is included for **research and demonstration purposes only**. Organisations planning commercial deployment should verify applicable WHO-UMC terms before use.

**Open-Source Software**

| Library | Provider | Licence |
|---|---|---|
| Llama 3.1 (model weights) | Meta AI | Meta LLAMA 3 Community Licence |
| LangGraph / LangChain | LangChain AI | MIT |
| FastAPI | Sebastián Ramírez | MIT |
| Microsoft Presidio | Microsoft Corporation | MIT |
| spaCy + `en_core_web_lg` | Explosion AI GmbH | MIT |
| FAISS | Meta Platforms Inc. | MIT |
| sentence-transformers / `all-MiniLM-L6-v2` | UKP Lab, TU Darmstadt | Apache 2.0 |
| cryptography (Fernet / AES-128) | cryptography.io | Apache 2.0 / BSD |
| Ollama (Python client) | Ollama contributors | MIT |
| bcrypt | The bcrypt authors | Apache 2.0 |

> **Note on model weights:** Llama 3.1 and `all-MiniLM-L6-v2` model weights are **not distributed with this repository**. They are downloaded independently via Ollama and `sentence-transformers` respectively, under their own licence terms.

**Regulatory Frameworks Referenced**

- **ICH E2B(R3)** — Electronic Standards for the Transfer of Regulatory Information (2016). *International Council for Harmonisation (ICH)*.
- **ICH M2** — Electronic Standards for the Transfer of Regulatory Information. *ICH*.
- **21 CFR Part 11** — Electronic Records; Electronic Signatures. *U.S. FDA* (referenced as design inspiration; system is not formally validated).
- **CIOMS WG XIV** — *Artificial Intelligence in Pharmacovigilance*. Council for International Organizations of Medical Sciences, Principle 6 (Demographic Stratification).

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](./LICENSE) file for full terms, third-party component notices, terminology source licences, and the regulatory disclaimer.

---

<div align="center">

Built with the constraint that **pharmacovigilance tooling should be accessible to every organisation, regardless of budget.**

</div>
