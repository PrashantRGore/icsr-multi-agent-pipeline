"""
tests/unit/test_rxnorm_client.py
==================================
Unit tests for infra/rxnorm_client.py (all HTTP calls mocked)

Tests:
  1.  Cache DB created on init
  2.  lookup() hits API on cache miss (mocked)
  3.  lookup() returns from_cache=True on second call (no extra HTTP)
  4.  lookup() returns None rxcui when drug not found in RxNorm
  5.  normalize() strips dosage suffixes from drug name
  6.  batch_lookup() returns results for all drugs
  7.  lookup() caches even NULL results (avoids repeated failed lookups)
  8.  cache_count() increments after each unique lookup
  9.  RxCUI lookup failure is soft (no exception raised)
  10. label lookup failure is soft (returns None label)
"""
import sqlite3
from unittest.mock import MagicMock, patch, call

import pytest

from infra.rxnorm_client import RxNormClient, RxNormResult, _normalize


@pytest.fixture()
def client(tmp_path):
    return RxNormClient(db_path=tmp_path / "rxnorm_test.db")


def _rxcui_response(rxcui: str):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"idGroup": {"rxnormId": [rxcui]}}
    return m


def _rxcui_empty_response():
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"idGroup": {}}
    return m


def _label_response(label: str):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "propConceptGroup": {
            "propConcept": [{"propName": "RxNorm Name", "propValue": label}]
        }
    }
    return m


def _class_response(drug_class: str):
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {
        "rxclassDrugInfoList": {
            "rxclassDrugInfo": [
                {
                    "rela": "has_EPC",
                    "rxclassMinConceptItem": {"className": drug_class},
                }
            ]
        }
    }
    return m


def _class_empty_response():
    m = MagicMock()
    m.raise_for_status = MagicMock()
    m.json.return_value = {"rxclassDrugInfoList": {"rxclassDrugInfo": []}}
    return m


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cache DB created on init
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_db_created(tmp_path):
    db_path = tmp_path / "rx.db"
    RxNormClient(db_path=db_path)
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    conn.close()
    assert "rxnorm_cache" in tables


# ─────────────────────────────────────────────────────────────────────────────
# 2. lookup() hits API on cache miss
# ─────────────────────────────────────────────────────────────────────────────

def test_lookup_hits_api_on_cache_miss(client):
    with patch.object(client._session, "get") as mock_get:
        mock_get.side_effect = [
            _rxcui_response("723"),
            _label_response("Amoxicillin"),
            _class_response("Penicillin-type Antibiotic"),
        ]
        result = client.lookup("amoxicillin")

    assert result.rxcui == "723"
    assert result.label == "Amoxicillin"
    assert result.drug_class == "Penicillin-type Antibiotic"
    assert result.from_cache is False
    assert mock_get.call_count == 3


# ─────────────────────────────────────────────────────────────────────────────
# 3. lookup() returns from_cache=True on second call
# ─────────────────────────────────────────────────────────────────────────────

def test_lookup_cache_hit_on_second_call(client):
    with patch.object(client._session, "get") as mock_get:
        mock_get.side_effect = [
            _rxcui_response("723"),
            _label_response("Amoxicillin"),
            _class_response("Penicillin-type Antibiotic"),
        ]
        client.lookup("amoxicillin")  # First call — API
        api_calls_first = mock_get.call_count

    # Second call — should hit cache, no new HTTP calls
    result2 = client.lookup("amoxicillin")
    assert result2.from_cache is True
    assert result2.rxcui == "723"


# ─────────────────────────────────────────────────────────────────────────────
# 4. lookup() returns None rxcui when drug not found
# ─────────────────────────────────────────────────────────────────────────────

def test_lookup_unknown_drug_returns_none_rxcui(client):
    with patch.object(client._session, "get") as mock_get:
        mock_get.return_value = _rxcui_empty_response()
        result = client.lookup("fakecillin-3000")

    assert result.rxcui is None
    assert result.label is None
    assert result.from_cache is False


# ─────────────────────────────────────────────────────────────────────────────
# 5. _normalize strips dosage suffixes
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_strips_dosage_suffix():
    assert _normalize("Amoxicillin 500mg tablet") == "amoxicillin"
    assert _normalize("Atorvastatin 20mg oral")   == "atorvastatin"
    assert _normalize("  Metformin 1000 mg  ")    == "metformin"
    assert _normalize("Ibuprofen 400 mg capsule") == "ibuprofen"
    assert _normalize("Warfarin")                 == "warfarin"


# ─────────────────────────────────────────────────────────────────────────────
# 6. batch_lookup returns results for all drugs
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_lookup(client):
    side_effects = [
        _rxcui_response("723"), _label_response("Amoxicillin"), _class_response("Antibiotic"),
        _rxcui_response("83367"), _label_response("Atorvastatin"), _class_empty_response(),
    ]
    with patch.object(client._session, "get", side_effect=side_effects):
        results = client.batch_lookup(["amoxicillin", "atorvastatin"])

    assert "amoxicillin" in results
    assert "atorvastatin" in results
    assert results["amoxicillin"].rxcui == "723"
    assert results["atorvastatin"].rxcui == "83367"


# ─────────────────────────────────────────────────────────────────────────────
# 7. lookup() caches NULL results
# ─────────────────────────────────────────────────────────────────────────────

def test_lookup_caches_null_results(client):
    with patch.object(client._session, "get", return_value=_rxcui_empty_response()):
        client.lookup("fakecillin-3000")

    # Second call should come from cache (NULL cached), no HTTP call
    with patch.object(client._session, "get") as mock_get:
        result = client.lookup("fakecillin-3000")

    assert result.from_cache is True
    assert result.rxcui is None
    assert not mock_get.called


# ─────────────────────────────────────────────────────────────────────────────
# 8. cache_count increments after each unique lookup
# ─────────────────────────────────────────────────────────────────────────────

def test_cache_count_increments(client):
    assert client.cache_count() == 0

    se = [
        _rxcui_response("723"), _label_response("Amoxicillin"), _class_empty_response(),
        _rxcui_response("83367"), _label_response("Atorvastatin"), _class_empty_response(),
    ]
    with patch.object(client._session, "get", side_effect=se):
        client.lookup("amoxicillin")
        client.lookup("atorvastatin")

    assert client.cache_count() == 2


# ─────────────────────────────────────────────────────────────────────────────
# 9. RxCUI lookup failure is soft (no exception)
# ─────────────────────────────────────────────────────────────────────────────

def test_rxcui_api_failure_is_soft(client):
    import requests as req
    with patch.object(
        client._session, "get",
        side_effect=req.ConnectionError("Network unreachable")
    ):
        result = client.lookup("amoxicillin")

    # Must not raise; returns None fields
    assert result.rxcui is None
    assert result.from_cache is False


# ─────────────────────────────────────────────────────────────────────────────
# 10. Label lookup failure is soft (returns None label)
# ─────────────────────────────────────────────────────────────────────────────

def test_label_failure_is_soft(client):
    import requests as req

    call_count = [0]

    def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return _rxcui_response("723")
        raise req.ConnectionError("Network error on label call")

    with patch.object(client._session, "get", side_effect=side_effect):
        result = client.lookup("amoxicillin")

    assert result.rxcui == "723"
    assert result.label is None   # soft failure — label not retrieved
