"""
infra/rxnorm_client.py
======================
NLM RxNorm REST API client with SQLite caching.

NLM RxNorm API:
  - Base URL: https://rxnav.nlm.nih.gov/REST
  - No API key required
  - Rate limit: ~20 req/s (we enforce 85 req/min = ~1.4 req/s for safety)
  - Docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html

Endpoints used:
  GET /rxcui?name={name}&search=2&allsrc=0   → find RxCUI by name (approximate)
  GET /rxcui/{rxcui}/allProperties?prop=ALL  → get all properties for RxCUI
  GET /rxcui/{rxcui}/related?tty=IN+BN       → get ingredient and brand names
  GET /rxcui/{rxcui}/proprietary             → get drug class (via EPC classification)

SQLite cache (audit/rxnorm_cache.db):
  - Caches (normalized_name → rxcui, label, drug_class) lookups permanently
  - Cache never expires (RxNorm CUI assignments are stable for approved drugs)
  - Thread-safe: each call opens its own connection in WAL mode

Hardware note: RxNorm API calls are strictly sequential
(rate limiter + SQLite cache hit avoids redundant HTTP calls).
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, NamedTuple, Optional

import requests

logger = logging.getLogger(__name__)

_BASE_URL       = os.getenv("RXNORM_BASE_URL", "https://rxnav.nlm.nih.gov/REST")
_RATE_LIMIT_MIN = int(os.getenv("RXNORM_RATE_LIMIT_PER_MIN", "85"))
_MIN_INTERVAL   = 60.0 / _RATE_LIMIT_MIN   # seconds between calls
_DEFAULT_DB     = os.getenv("RXNORM_CACHE_PATH", "audit/rxnorm_cache.db")


# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

class RxNormResult(NamedTuple):
    """Resolved RxNorm data for a drug name."""
    rxcui:       Optional[str]   # RxNorm Concept Unique Identifier
    label:       Optional[str]   # Normalized drug label from RxNorm
    drug_class:  Optional[str]   # EPC drug class (if available)
    from_cache:  bool            # True if result came from local SQLite cache


# ─────────────────────────────────────────────────────────────────────────────
# DDL
# ─────────────────────────────────────────────────────────────────────────────

_DDL_CACHE = """
CREATE TABLE IF NOT EXISTS rxnorm_cache (
    normalized_name  TEXT PRIMARY KEY,
    rxcui            TEXT,
    label            TEXT,
    drug_class       TEXT,
    fetched_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""

_DDL_CACHE_IDX = """
CREATE INDEX IF NOT EXISTS idx_rxnorm_rxcui ON rxnorm_cache (rxcui);
"""


def _normalize(name: str) -> str:
    """Lowercase, strip extra whitespace, remove common suffixes for cache key."""
    name = name.lower().strip()
    # Remove trailing dose + unit: "500mg", "500 mg", "20mcg", etc.
    name = re.sub(r"\s+\d+\s*(mg|mcg|ug|ml|g|iu|mmol)\b.*$", "", name)
    # Remove common trailing form/route identifiers
    name = re.sub(r"\s+(tablet|capsule|injection|oral|solution|extended.release|er|sr)\b.*$", "", name)
    return re.sub(r"\s+", " ", name).strip()


# ─────────────────────────────────────────────────────────────────────────────
# RxNormClient
# ─────────────────────────────────────────────────────────────────────────────

class RxNormClient:
    """
    NLM RxNorm REST client with SQLite caching and rate limiting.

    Usage:
        client = RxNormClient()
        result = client.lookup("amoxicillin")
        print(result.rxcui, result.label, result.drug_class)
    """

    def __init__(self, db_path: str | Path = _DEFAULT_DB) -> None:
        self.db_path      = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._session     = requests.Session()
        self._session.headers["Accept"] = "application/json"
        self._last_call: float = 0.0
        self._initialize()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(_DDL_CACHE)
            conn.execute(_DDL_CACHE_IDX)
        logger.debug("RxNormClient: cache initialized at %s", self.db_path)

    # ── Rate limiter ─────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Block until at least _MIN_INTERVAL seconds since the last API call."""
        elapsed = time.monotonic() - self._last_call
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    # ── Cache read/write ─────────────────────────────────────────────────────

    def _cache_get(self, normalized: str) -> Optional[RxNormResult]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT rxcui, label, drug_class FROM rxnorm_cache WHERE normalized_name = ?",
                (normalized,)
            ).fetchone()
        if row:
            return RxNormResult(
                rxcui=row["rxcui"],
                label=row["label"],
                drug_class=row["drug_class"],
                from_cache=True,
            )
        return None

    def _cache_set(
        self,
        normalized: str,
        rxcui:      Optional[str],
        label:      Optional[str],
        drug_class: Optional[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO rxnorm_cache
                   (normalized_name, rxcui, label, drug_class)
                   VALUES (?, ?, ?, ?)""",
                (normalized, rxcui, label, drug_class),
            )

    # ── API calls ────────────────────────────────────────────────────────────

    def _get_rxcui(self, name: str) -> Optional[str]:
        """
        Resolve drug name → RxCUI using the approximate search endpoint.
        search=2 → approximate matching; allsrc=0 → RxNorm sources only.
        """
        self._throttle()
        try:
            resp = self._session.get(
                f"{_BASE_URL}/rxcui",
                params={"name": name, "search": 2, "allsrc": 0},
                timeout=15,
            )
            resp.raise_for_status()
            ids = resp.json().get("idGroup", {}).get("rxnormId", [])
            if ids:
                return ids[0]
        except Exception as exc:
            logger.warning("RxNorm /rxcui lookup failed for %r: %s", name, exc)
        return None

    def _get_label(self, rxcui: str) -> Optional[str]:
        """Fetch the preferred name for an RxCUI."""
        self._throttle()
        try:
            resp = self._session.get(
                f"{_BASE_URL}/rxcui/{rxcui}/allProperties",
                params={"prop": "NAMES"},
                timeout=15,
            )
            resp.raise_for_status()
            props = resp.json().get("propConceptGroup", {}).get("propConcept", [])
            for p in props:
                if p.get("propName") == "RxNorm Name":
                    return p.get("propValue")
        except Exception as exc:
            logger.warning("RxNorm label lookup failed for rxcui=%s: %s", rxcui, exc)
        return None

    def _get_drug_class(self, rxcui: str) -> Optional[str]:
        """
        Fetch EPC (Established Pharmacologic Class) drug class for an RxCUI.
        Uses the /class/byRxcui endpoint — falls back to None if not available.
        """
        self._throttle()
        try:
            resp = self._session.get(
                f"{_BASE_URL}/rxclass/class/byRxcui",
                params={"rxcui": rxcui, "relaSource": "FDASPL"},
                timeout=15,
            )
            resp.raise_for_status()
            classes = (
                resp.json()
                    .get("rxclassDrugInfoList", {})
                    .get("rxclassDrugInfo", [])
            )
            # Prefer EPC (Established Pharmacologic Class) entries
            epc = [
                c["rxclassMinConceptItem"]["className"]
                for c in classes
                if c.get("rela") == "has_EPC"
            ]
            if epc:
                return epc[0]
            # Fall back to first available class
            if classes:
                return classes[0]["rxclassMinConceptItem"]["className"]
        except Exception as exc:
            logger.debug(
                "RxNorm drug_class lookup failed for rxcui=%s: %s", rxcui, exc
            )
        return None

    # ── Public API ───────────────────────────────────────────────────────────

    def lookup(self, drug_name: str) -> RxNormResult:
        """
        Resolve a drug name to its RxNorm RxCUI, normalized label, and drug class.

        Order of operations:
          1. Normalize the drug name for cache key
          2. Cache hit → return immediately (no HTTP call)
          3. Cache miss → call NLM RxNorm API (rate-limited)
          4. Cache the result (even NULL results, to avoid repeated failed lookups)
          5. Return RxNormResult

        Failures are soft (returns None fields, from_cache=False).
        Hard failures only on network exceptions — never raises for unknown drugs.
        """
        normalized = _normalize(drug_name)

        # 1. Cache check
        cached = self._cache_get(normalized)
        if cached is not None:
            logger.debug("RxNorm cache hit for %r → rxcui=%s", normalized, cached.rxcui)
            return cached

        # 2. Resolve RxCUI
        rxcui = self._get_rxcui(drug_name)
        label: Optional[str] = None
        drug_class: Optional[str] = None

        if rxcui:
            label      = self._get_label(rxcui)
            drug_class = self._get_drug_class(rxcui)

        # 3. Cache and return
        self._cache_set(normalized, rxcui, label, drug_class)

        result = RxNormResult(rxcui=rxcui, label=label, drug_class=drug_class, from_cache=False)
        logger.debug(
            "RxNorm resolved %r → rxcui=%s label=%r class=%r",
            normalized, rxcui, label, drug_class
        )
        return result

    def batch_lookup(self, drug_names: list[str]) -> dict[str, RxNormResult]:
        """
        Resolve multiple drug names. Cache hits are returned without any HTTP call.
        HTTP calls are made sequentially (never concurrent) — rate limit respected.
        """
        return {name: self.lookup(name) for name in drug_names}

    def cache_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM rxnorm_cache").fetchone()[0]
