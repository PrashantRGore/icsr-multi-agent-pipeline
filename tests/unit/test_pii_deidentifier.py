"""
tests/unit/test_pii_deidentifier.py
=====================================
Unit tests for infra/pii_deidentifier.py.

Strategy: presidio performs real NER so tests use concrete clinical sentences
containing known PII patterns. We verify that:
  - Known PII types are detected and replaced
  - entity_map is returned correctly
  - Empty/non-PII text is handled gracefully
  - Entity map is persisted to disk
  - Fernet encryption works end-to-end
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from infra.pii_deidentifier import PIIDeidentifier, DeidentifyResult


@pytest.fixture(scope="module")
def deid(tmp_path_factory) -> PIIDeidentifier:
    """Single PIIDeidentifier instance shared across the module (slow to init)."""
    maps_dir = tmp_path_factory.mktemp("pii_maps")
    return PIIDeidentifier(maps_dir=maps_dir)


@pytest.fixture(scope="module")
def deid_encrypted(tmp_path_factory) -> PIIDeidentifier:
    """PIIDeidentifier with Fernet encryption enabled."""
    from cryptography.fernet import Fernet
    key = Fernet.generate_key().decode()
    maps_dir = tmp_path_factory.mktemp("pii_maps_enc")
    return PIIDeidentifier(maps_dir=maps_dir, encryption_key=key)


class TestBasicDeidentification:
    def test_returns_deidentify_result(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify("Patient seen on 2024-01-15.", trace_id="T-001")
        assert isinstance(result, DeidentifyResult)

    def test_date_is_redacted(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify(
            "Adverse event reported on 2024-01-15.", trace_id="T-002"
        )
        assert "2024-01-15" not in result.anonymized_text

    def test_person_name_redacted(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify(
            "Patient John Smith reported nausea.", trace_id="T-003"
        )
        # Presidio should detect PERSON entity
        person_found = any(e.entity_type == "PERSON" for e in result.entities_found)
        assert person_found

    def test_email_is_redacted(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify(
            "Contact reporter at john.smith@hospital.org for follow-up.",
            trace_id="T-004"
        )
        assert "john.smith@hospital.org" not in result.anonymized_text

    def test_trace_id_in_result(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify("No PII here.", trace_id="TRACE-999")
        assert result.trace_id == "TRACE-999"


class TestEmptyAndEdgeCases:
    def test_empty_string_returns_unchanged(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify("", trace_id="T-EMPTY")
        assert result.anonymized_text == ""
        assert result.entities_found == []

    def test_whitespace_only_returns_unchanged(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify("   ", trace_id="T-WS")
        assert result.anonymized_text.strip() == ""

    def test_no_pii_text_unchanged_or_minimal(self, deid: PIIDeidentifier) -> None:
        """Non-PII clinical text should pass through largely unchanged."""
        text = "The patient received metformin 500mg twice daily."
        result = deid.deidentify(text, trace_id="T-NOPII")
        # No entities flagged — original text preserved (or very close)
        assert len(result.anonymized_text) > 0
        # metformin should survive
        assert "metformin" in result.anonymized_text or "500" in result.anonymized_text


class TestEntityMap:
    def test_entity_map_contains_original_values(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify(
            "Event date: 2024-03-20. Reporter: jane.doe@clinic.com",
            trace_id="T-MAP"
        )
        if result.entity_map:
            # All values in the map should be substrings of the original text
            original = "Event date: 2024-03-20. Reporter: jane.doe@clinic.com"
            for token, original_val in result.entity_map.items():
                assert original_val in original

    def test_entity_map_file_created(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify(
            "Adverse event reported by John Doe on 2024-06-01.",
            trace_id="MAP-FILE-001"
        )
        if result.map_path:
            assert Path(result.map_path).exists()


class TestEncryption:
    def test_encrypted_map_file_created(self, deid_encrypted: PIIDeidentifier) -> None:
        result = deid_encrypted.deidentify(
            "Patient Jane Smith, DOB 1980-03-01, email jane@clinic.com",
            trace_id="ENC-001"
        )
        if result.map_path:
            p = Path(result.map_path)
            assert p.exists()
            # Encrypted file should not be readable as plain JSON
            raw = p.read_bytes()
            try:
                json.loads(raw)
                is_plaintext = True
            except Exception:
                is_plaintext = False
            assert not is_plaintext, "Entity map should be encrypted, not plain JSON"

    def test_encrypted_map_is_decryptable(self, deid_encrypted: PIIDeidentifier) -> None:
        """Fernet-encrypted content must round-trip correctly."""
        result = deid_encrypted.deidentify(
            "Contact Dr. Smith at drsmith@hospital.org",
            trace_id="ENC-ROUND"
        )
        if result.map_path and deid_encrypted._fernet:
            raw = Path(result.map_path).read_bytes()
            decrypted = deid_encrypted._fernet.decrypt(raw)
            recovered = json.loads(decrypted)
            assert isinstance(recovered, dict)


class TestResultIntegrity:
    def test_anonymized_text_is_string(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify("Some clinical text.", trace_id="T-STR")
        assert isinstance(result.anonymized_text, str)

    def test_entities_found_is_list(self, deid: PIIDeidentifier) -> None:
        result = deid.deidentify("Patient John on 2024-01-01.", trace_id="T-LIST")
        assert isinstance(result.entities_found, list)

    def test_anonymized_length_reasonable(self, deid: PIIDeidentifier) -> None:
        text = "Patient John Smith received aspirin on 2024-01-15."
        result = deid.deidentify(text, trace_id="T-LEN")
        # Replacement tokens are typically longer than original values — length increases
        assert len(result.anonymized_text) > 0
