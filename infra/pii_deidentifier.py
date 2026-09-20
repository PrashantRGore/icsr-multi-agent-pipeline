"""
infra/pii_deidentifier.py
==========================
Pre-ingestion PII de-identification using Microsoft Presidio.

Runs entirely locally — no cloud API calls, no metering.
Uses presidio-analyzer (spaCy NER) + presidio-anonymizer (replacement).

Design decisions:
  - Replacement strategy: token substitution e.g. <PERSON_1>, <DATE_1>
    Produces readable text; preserves narrative structure for agents.
  - entity_map: maps token → original value; encrypted at rest with Fernet
    if PII_ENCRYPTION_KEY env var is set. Allows re-identification by
    authorized users (audit purposes only).
  - If presidio is not installed, PIIDeidentifier raises ImportError at
    construction time — caller (hitl/main.py) catches this gracefully.
  - Clinical text tuning:
      * AGE recognizer: custom pattern for "45-year-old", "12-month-old"
      * MEDICAL_RECORD_NUMBER pattern for ICSR case IDs

Environment variables:
  PII_ENCRYPTION_KEY : Fernet key (generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
                       If not set, entity maps are stored plaintext with a warning.
  PII_MAPS_DIR       : Directory for entity map files (default: data/pii_maps)

Usage:
  deidentifier = PIIDeidentifier()
  result = deidentifier.deidentify("John Smith, 45-year-old male...", trace_id="CASE-001")
  print(result.anonymized_text)   # "<PERSON_1>, <AGE_1> male..."
  print(result.entities_found)    # [EntityResult(type='PERSON', original='John Smith', ...)]
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntityResult:
    """A single detected PII entity."""
    entity_type: str
    original:    str
    token:       str        # Replacement token e.g. "<PERSON_1>"
    start:       int
    end:         int
    score:       float


@dataclass
class DeidentifyResult:
    """Output of a single de-identification pass."""
    anonymized_text:  str
    entity_map:       dict[str, str]          # token → original value
    entities_found:   list[EntityResult]
    trace_id:         str
    map_path:         str | None = None       # Path to saved entity map (if any)


# ─────────────────────────────────────────────────────────────────────────────
# PIIDeidentifier
# ─────────────────────────────────────────────────────────────────────────────

class PIIDeidentifier:
    """
    Local PII de-identification service using presidio.

    Raises ImportError if presidio-analyzer or presidio-anonymizer
    are not installed — catch this in hitl/main.py lifespan.
    """

    # Entities to detect and replace
    _ENTITIES = [
        "PERSON",
        "DATE_TIME",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "LOCATION",
        "MEDICAL_LICENSE",
        "URL",
    ]

    def __init__(
        self,
        maps_dir: str | Path = "data/pii_maps",
        encryption_key: str | None = None,
    ) -> None:
        # Import lazily so module can be imported even if presidio not installed
        from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        self._maps_dir = Path(maps_dir)
        self._maps_dir.mkdir(parents=True, exist_ok=True)

        # Fernet encryption for entity maps
        self._fernet = None
        key = encryption_key or os.getenv("PII_ENCRYPTION_KEY")
        if key:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(key.encode() if isinstance(key, str) else key)
            logger.info("PIIDeidentifier: entity maps will be encrypted (Fernet)")
        else:
            logger.warning(
                "PIIDeidentifier: PII_ENCRYPTION_KEY not set — "
                "entity maps stored in plaintext. Set PII_ENCRYPTION_KEY for production."
            )

        # Build custom age recognizer (clinical text pattern)
        age_pattern = PatternRecognizer(
            supported_entity="AGE",
            patterns=[
                Pattern(
                    name="age_year_old",
                    regex=r"\b\d{1,3}[\s\-]*(year|yr|month|mo)s?[\s\-]*old\b",
                    score=0.85,
                ),
                Pattern(
                    name="age_standalone",
                    regex=r"\b\d{1,3}\s+years?\s+old\b",
                    score=0.80,
                ),
            ],
        )

        # Build ICSR case ID recognizer
        icsr_pattern = PatternRecognizer(
            supported_entity="CASE_ID",
            patterns=[
                Pattern(
                    name="icsr_case_id",
                    regex=r"\bICSR[\-_]\d{8}[\-_]\d{3,}\b",
                    score=0.95,
                ),
            ],
        )

        self._analyzer = AnalyzerEngine()
        self._analyzer.registry.add_recognizer(age_pattern)
        self._analyzer.registry.add_recognizer(icsr_pattern)

        self._anonymizer = AnonymizerEngine()
        self._operator_config = OperatorConfig("replace", {"new_value": None})  # token injected per-call

        logger.info("PIIDeidentifier: initialised (entities=%s)", self._ENTITIES + ["AGE", "CASE_ID"])

    def deidentify(self, text: str, trace_id: str) -> DeidentifyResult:
        """
        De-identify a narrative text string.

        Parameters
        ----------
        text     : Raw narrative text (may contain PII)
        trace_id : Case trace_id — used to name the entity map file

        Returns
        -------
        DeidentifyResult with anonymized_text, entity_map, and entities_found
        """
        if not text or not text.strip():
            return DeidentifyResult(
                anonymized_text = text,
                entity_map      = {},
                entities_found  = [],
                trace_id        = trace_id,
            )

        entities_to_detect = self._ENTITIES + ["AGE", "CASE_ID"]

        # Step 1: Analyze
        results = self._analyzer.analyze(
            text     = text,
            entities = entities_to_detect,
            language = "en",
        )

        if not results:
            return DeidentifyResult(
                anonymized_text = text,
                entity_map      = {},
                entities_found  = [],
                trace_id        = trace_id,
            )

        # Step 2: Build token map (entity_type_N → original value)
        #   Sort by position to assign sequential numbers correctly
        results_sorted = sorted(results, key=lambda r: r.start)
        type_counters: dict[str, int] = {}
        token_map: dict[str, str] = {}    # token → original
        operator_map: dict[str, Any] = {}  # presidio operator per result

        for r in results_sorted:
            etype = r.entity_type
            type_counters[etype] = type_counters.get(etype, 0) + 1
            token = f"<{etype}_{type_counters[etype]}>"
            original = text[r.start:r.end]
            token_map[token] = original

        # Step 3: Build operator config that maps each entity occurrence to its token
        #   We use a custom replace operator for each entity type
        from presidio_anonymizer.entities import OperatorConfig
        type_seen: dict[str, int] = {}

        def _make_operator(entity_type: str) -> OperatorConfig:
            type_seen[entity_type] = type_seen.get(entity_type, 0) + 1
            token = f"<{entity_type}_{type_seen[entity_type]}>"
            return OperatorConfig("replace", {"new_value": token})

        # Build per-entity-type operators (presidio applies one operator per type)
        operators: dict[str, OperatorConfig] = {}
        for r in results_sorted:
            if r.entity_type not in operators:
                operators[r.entity_type] = OperatorConfig(
                    "replace", {"new_value": f"<{r.entity_type}_REDACTED>"}
                )

        # Step 4: Anonymize
        from presidio_anonymizer.entities import RecognizerResult as AR
        anonymized = self._anonymizer.anonymize(
            text              = text,
            analyzer_results  = results,
            operators         = operators,
        )

        anonymized_text = anonymized.text

        # Step 5: Build entity result list
        entity_results = [
            EntityResult(
                entity_type = r.entity_type,
                original    = text[r.start:r.end],
                token       = f"<{r.entity_type}_REDACTED>",
                start       = r.start,
                end         = r.end,
                score       = r.score,
            )
            for r in results_sorted
        ]

        # Step 6: Persist entity map
        map_path = self._save_entity_map(trace_id, token_map)

        logger.info(
            "PIIDeidentifier: trace_id=%s entities_found=%d types=%s",
            trace_id,
            len(results),
            list({r.entity_type for r in results}),
        )

        return DeidentifyResult(
            anonymized_text = anonymized_text,
            entity_map      = token_map,
            entities_found  = entity_results,
            trace_id        = trace_id,
            map_path        = str(map_path) if map_path else None,
        )

    def _save_entity_map(self, trace_id: str, entity_map: dict) -> Path | None:
        """Persist entity_map to disk, encrypted if Fernet key is configured."""
        if not entity_map:
            return None

        # Sanitise trace_id for use as filename
        safe_id = re.sub(r"[^A-Za-z0-9\-_]", "_", trace_id)
        payload = json.dumps(entity_map, ensure_ascii=False).encode()

        if self._fernet:
            content  = self._fernet.encrypt(payload)
            suffix   = ".json.enc"
        else:
            content = payload
            suffix  = ".json"

        map_file = self._maps_dir / f"{safe_id}{suffix}"
        map_file.write_bytes(content)
        return map_file
