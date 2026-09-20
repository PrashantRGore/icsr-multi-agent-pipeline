# Pharmacovigilance System Master File (PSMF) — AI System Component Documentation
## Prepared under: EU GVP Module II Requirements / CIOMS WG XIV Governance Principle

> **Status**: DRAFT — To be completed by the Qualified Person for Pharmacovigilance (QPPV)
> **Regulation**: EU GVP Module II, Article 8(2) of Regulation (EC) No 726/2004

---

### 1. AI System Identification

| Field | Value |
|---|---|
| **System Name** | Multi-Agent ICSR Processing & Governance Engine |
| **System Version** | 0.6.0 |
| **Deployment Date** | `{{DEPLOYMENT_DATE}}` |
| **Environment** | Local / Private — Zero external API calls for PV data |
| **Data Classification** | Synthetic CIOMS-I test cases only (NO real patient data) |
| **GAMP 5 Category** | Category 4 (Configured software) / Category 5 (Custom software) |
| **ADR Registry** | `governance/decisions.md` |

---

### 2. AI System Purpose

This system automates the processing of Individual Case Safety Reports (ICSRs) in conformance with:
- **ICH E2B(R3)** — Electronic Transmission of Individual Case Safety Reports
- **ICH E2A** — Clinical Safety Data Management: Definitions and Standards for Expedited Reporting
- **21 CFR Part 11** — Electronic Records; Electronic Signatures
- **21 CFR 314.81(b)(1)** — 15-day expedited reporting for serious unexpected AEs
- **CIOMS-I** — Guidelines for Preparing Core Clinical-Safety Information

The system does **not** autonomously submit regulatory reports. All outputs require QPPV or designated medical reviewer approval before regulatory transmission (Human-in-the-Loop / Human-in-Command mode).

---

### 3. AI Components

| Component | Type | Model / Version | Temperature |
|---|---|---|---|
| Triage Agent | Generative LLM | `llama3.1:8b-instruct-q4_K_M` (Ollama) | 0.0 (deterministic) |
| Extraction Agent | Generative LLM | `llama3.1:8b-instruct-q4_K_M` (Ollama) | 0.0 |
| QC Agent | Generative LLM + Rule-based | `llama3.1:8b-instruct-q4_K_M` + Clinical Negatives Ontology | 0.0 |
| Coding Agent | FAISS k-NN + LLM | OAE FAISS index + CTCAE FAISS index | 0.0 |
| Causality Agent | Generative LLM | `llama3.1:8b-instruct-q4_K_M` (Ollama) | 0.0 |
| Listedness Agent | Rule-based + LLM | FDA DailyMed SPL XML + LLM verification | 0.0 |
| Narrative Agent | Generative LLM | `llama3.1:8b-instruct-q4_K_M` (Ollama) | 0.0 |
| Secondary QC Auditor | Rule-based (no LLM) | Deterministic field-diff classifier | N/A |
| E2B(R3) Exporter | Serializer (no LLM) | `infra/e2b_exporter.py` — ICH E2B(R3) XML | N/A |

---

### 4. Terminology Sources

| Terminology | Source | License | Update Frequency |
|---|---|---|---|
| Ontology of Adverse Events (OAE) | OBO Foundry — oae.owl | CC BY 4.0 | At system update |
| NCI CTCAE v5 | NCI — ctcae_v5.xlsx | Public Domain (US Gov) | At version release |
| RxNorm | NLM RxNorm REST API | NLM Terms of Service (free, no key) | Runtime (cached in SQLite) |
| Reference Safety Information (RSI) | FDA DailyMed SPL XMLs | Public Domain | Quarterly download |
| Clinical Negatives Ontology | Internal (`data/clinical_negatives.json`) | Internal | Manual review at system update |

---

### 5. Validation Summary

#### 5.1 Schema Validation
- All agent outputs are validated against Pydantic V2 schemas before persistence
- Any schema violation → `ErrorClassification.SCHEMA_VIOLATION` → HITL routing
- Validation test results: `{{PYTEST_RESULTS_PATH}}`

#### 5.2 Performance Benchmarks
| Agent | Test Case Set | Pass Rate | Date |
|---|---|---|---|
| Triage Agent | `{{TRIAGE_BENCHMARK_PATH}}` | `{{TRIAGE_PASS_RATE}}` | `{{BENCHMARK_DATE}}` |
| Extraction Agent | `{{EXTRACTION_BENCHMARK_PATH}}` | `{{EXTRACTION_PASS_RATE}}` | `{{BENCHMARK_DATE}}` |
| QC Agent | `{{QC_BENCHMARK_PATH}}` | `{{QC_PASS_RATE}}` | `{{BENCHMARK_DATE}}` |

#### 5.3 Synthetic Test Cases
| Case ID | Description | Expected Outcome | Status |
|---|---|---|---|
| ICSR-20240315-SYN | Valid SAE — amoxicillin anaphylaxis, TIER_1 | VALID, HITL mandatory | `{{CASE_001_STATUS}}` |
| ICSR-20240402-SYN | Invalid — missing reporter | INVALID, early termination | `{{CASE_002_STATUS}}` |
| ICSR-20240520-SYN | Valid TIER_2 — atorvastatin DILI, rechallenge positive | VALID, UNLISTED event | `{{CASE_003_STATUS}}` |

---

### 6. Audit Trail

- **Audit DB path**: `audit/audit.db`
- **Immutability**: SQLite UPDATE/DELETE triggers prohibit modification of any audit record
- **Content hash**: SHA-256 of serialized agent output on every entry
- **Prompt versioning**: `prompt_version` field on every `AuditLogEntry`
- **HITL log**: `hitl_review_log` table in `audit/audit.db` — append-only
- **Learning signals**: `audit/learning.db` — demographic-stratified correction history

---

### 7. Human Oversight Configuration

| Risk Tier | Oversight Mode | Confidence Threshold | Description |
|---|---|---|---|
| TIER_1 (Death / Life-threatening) | HITL — Mandatory | 0.95 | Every case reviewed by QPPV before submission |
| TIER_2 (Other serious SAE) | HITL — Conditional | 0.85 | HITL if confidence < 0.85 or QC finds BLOCKER |
| TIER_3 (Non-serious AE) | HOTL — Monitoring | 0.75 | Auto-submit if all thresholds met; QPPV monitors |

---

### 8. Change Control

| Version | Date | Changed By | Summary of Changes |
|---|---|---|---|
| v4.0 | `{{DEPLOYMENT_DATE}}` | `{{SYSTEM_OWNER}}` | Initial deployment: LangGraph + MAGMA case graph + HITL API |
| v5.0 | `{{BENCHMARK_DATE}}` | `{{SYSTEM_OWNER}}` | Auth hardening: bcrypt API keys, PIIDeidentifier (Presidio), column-level Fernet encryption |
| v6.0 | `{{BENCHMARK_DATE}}` | `{{SYSTEM_OWNER}}` | Operational readiness: JSON structured logging, Prometheus metrics, enriched health check, Docker |
| v6.1 | `{{BENCHMARK_DATE}}` | `{{SYSTEM_OWNER}}` | Learning feedback loop: LearningDB reports, E2B(R3) export (CTCAE terms; MedDRA pending — ADR-001), PSMF auto-population |

---

### 9. QPPV Sign-off

| Role | Name | Date | Signature |
|---|---|---|---|
| Qualified Person for Pharmacovigilance | | | |
| System Owner / Data Scientist | | | |

---

*This PSMF component entry was auto-generated from `governance/psmf_template.md`.*
*Fill all `{{PLACEHOLDER}}` values before regulatory submission.*
*For architectural decisions (MedDRA, storage, benchmark methodology), see `governance/decisions.md`.*
