"""
infra/faiss_index.py
====================
FAISS-based semantic search wrapper for OAE and CTCAE term coding.

Design:
  - Wraps a pre-built FAISS flat index (IndexFlatIP — inner product, normalised
    vectors = cosine similarity) + a parallel JSON metadata list.
  - Embeddings are generated with sentence-transformers
    (all-MiniLM-L6-v2 by default — 22MB model, ~50ms per query, CPU-only).
  - k-NN search returns top-k results filtered by a minimum cosine similarity
    threshold (default 0.80 for OAE; 0.75 for CTCAE fallback).
  - Thread-safe for concurrent reads (FAISS flat indexes are read-safe).
    Writes only happen during index build (scripts/build_*_index.py).

Memory footprint:
  - OAE index (~3000 terms × 384-dim float32) ≈ 4.6MB
  - CTCAE index (~800 terms × 384-dim float32) ≈ 1.2MB
  - Sentence-transformer model: ~80MB resident
  - Total: ~85MB — well within the 16GB RAM envelope

Lazy loading:
  - The model is loaded on first search(), not at import time.
  - Subsequent searches reuse the cached model (module-level singleton).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Lazy-load sentence-transformers and faiss
# ─────────────────────────────────────────────────────────────────────────────

_EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embed_model = None   # Module-level singleton — loaded once

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        logger.info("FAISSIndex: loading embedding model %s ...", _EMBED_MODEL_NAME)
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        logger.info("FAISSIndex: embedding model loaded.")
    return _embed_model


# ─────────────────────────────────────────────────────────────────────────────
# Search result
# ─────────────────────────────────────────────────────────────────────────────

class SearchResult(NamedTuple):
    term:        str          # Preferred term from the ontology
    term_id:     str          # OAE ID (OAE:0001234) or CTCAE term ID
    score:       float        # Cosine similarity [0.0 – 1.0]
    metadata:    dict         # Full metadata entry from the index JSON


# ─────────────────────────────────────────────────────────────────────────────
# FAISSIndex
# ─────────────────────────────────────────────────────────────────────────────

class FAISSIndex:
    """
    Read-only FAISS flat index wrapper for ontology term search.

    Usage:
        oae_idx = FAISSIndex.load(
            index_path="data/oae.faiss",
            meta_path="data/oae_meta.json",
        )
        results = oae_idx.search("anaphylactic reaction", k=3, threshold=0.80)
        for r in results:
            print(r.term_id, r.term, r.score)
    """

    def __init__(self, index, metadata: list[dict]) -> None:
        self._index    = index        # faiss.IndexFlatIP (or wrapped)
        self._metadata = metadata     # list of dicts, parallel to index vectors

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, index_path: str | Path, meta_path: str | Path) -> "FAISSIndex":
        """
        Load a pre-built FAISS index and its metadata JSON.
        Raises FileNotFoundError if either file is absent
        (means build_*_index.py has not been run yet).
        """
        import faiss  # type: ignore

        index_path = Path(index_path)
        meta_path  = Path(meta_path)

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}. "
                "Run scripts/build_oae_index.py or scripts/build_ctcae_index.py first."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found: {meta_path}. "
                "Run the index build script first."
            )

        index    = faiss.read_index(str(index_path))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        if index.ntotal != len(metadata):
            raise ValueError(
                f"FAISS index has {index.ntotal} vectors but metadata has "
                f"{len(metadata)} entries. Rebuild the index."
            )

        logger.info(
            "FAISSIndex: loaded %s (%d terms)", index_path.name, index.ntotal
        )
        return cls(index, metadata)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(
        self,
        query:     str,
        k:         int   = 5,
        threshold: float = 0.80,
    ) -> list[SearchResult]:
        """
        Semantic search for a verbatim term.

        Args:
            query:     The verbatim adverse event term from the narrative.
            k:         Maximum number of candidates to return.
            threshold: Minimum cosine similarity required to include a result.
                       OAE: 0.80 (high precision for primary coding).
                       CTCAE fallback: 0.75 (wider net).

        Returns:
            List of SearchResult ordered by score descending.
            Empty list if no result meets the threshold.
        """
        model = _get_embed_model()

        # Embed and normalise query vector
        vec = model.encode([query], normalize_embeddings=True).astype("float32")

        # FAISS inner product search on normalised vectors = cosine similarity
        scores, indices = self._index.search(vec, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for unfilled slots
            if float(score) < threshold:
                continue
            meta = self._metadata[idx]
            results.append(SearchResult(
                term=meta.get("term", ""),
                term_id=meta.get("term_id", ""),
                score=float(score),
                metadata=meta,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        logger.debug(
            "FAISSIndex.search(%r): %d results above threshold %.2f",
            query, len(results), threshold
        )
        return results

    def top_match(
        self,
        query:     str,
        threshold: float = 0.80,
    ) -> Optional[SearchResult]:
        """Return the single highest-scoring match, or None if below threshold."""
        results = self.search(query, k=1, threshold=threshold)
        return results[0] if results else None

    @property
    def size(self) -> int:
        """Number of terms in the index."""
        return self._index.ntotal
