"""
scripts/build_oae_index.py
==========================
Build the OAE (Ontology of Adverse Events) FAISS index.

Run this ONCE before starting the pipeline:
    python scripts/build_oae_index.py

What it does:
  1. Downloads oae.owl from the OBO Foundry GitHub release (CC BY 4.0)
  2. Parses OWL/RDF with rdflib to extract:
       - OAE class ID  (e.g. OAE:0001234)
       - preferred label (rdfs:label)
       - synonyms (oboInOwl:hasExactSynonym, oboInOwl:hasBroadSynonym)
  3. Embeds all terms + synonyms with sentence-transformers/all-MiniLM-L6-v2
  4. Builds a FAISS IndexFlatIP index (inner product on L2-normalised vectors
     = cosine similarity search)
  5. Writes:
       data/oae.faiss    — binary FAISS index
       data/oae_meta.json — JSON array parallel to index vectors

Memory budget:
  - OAE has ~3,000 terms; each 384-dim float32 = 1.5KB → ~4.6MB total
  - Embedding batch: 512 terms at a time to keep RAM < 500MB during build

OAE download URL: https://github.com/OAEdev/oae/releases/latest/download/oae.owl
Fallback:         http://purl.obolibrary.org/obo/oae.owl
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_oae_index")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parent.parent
_OWL_PATH    = _ROOT / "data" / "oae.owl"
_FAISS_PATH  = _ROOT / "data" / "oae.faiss"
_META_PATH   = _ROOT / "data" / "oae_meta.json"
_BATCH_SIZE  = 512

_OAE_URL_PRIMARY  = "https://github.com/OAEdev/oae/releases/latest/download/oae.owl"
_OAE_URL_FALLBACK = "http://purl.obolibrary.org/obo/oae.owl"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download OAE OWL
# ─────────────────────────────────────────────────────────────────────────────

def download_oae() -> None:
    if _OWL_PATH.exists():
        logger.info("OAE OWL already exists at %s — skipping download.", _OWL_PATH)
        return

    import requests  # type: ignore

    _OWL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for url in [_OAE_URL_PRIMARY, _OAE_URL_FALLBACK]:
        logger.info("Downloading OAE from %s ...", url)
        try:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(_OWL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
            logger.info("Downloaded OAE OWL to %s (%s bytes)", _OWL_PATH, _OWL_PATH.stat().st_size)
            return
        except Exception as exc:
            logger.warning("Download from %s failed: %s", url, exc)

    raise RuntimeError(
        "Failed to download oae.owl from all URLs. "
        "Manual download: http://purl.obolibrary.org/obo/oae.owl → data/oae.owl"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Parse OWL
# ─────────────────────────────────────────────────────────────────────────────

def parse_oae_owl() -> list[dict]:
    """
    Parse OAE OWL file using rdflib.
    Returns list of dicts: {term_id, term, synonyms, text_to_embed}.
    """
    import rdflib  # type: ignore
    from rdflib import OWL, RDF, RDFS, Namespace

    OBO_IN_OWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")
    OAE_NS     = Namespace("http://purl.obolibrary.org/obo/OAE_")

    logger.info("Parsing OAE OWL from %s ...", _OWL_PATH)
    g = rdflib.Graph()
    g.parse(str(_OWL_PATH), format="xml")

    entries: list[dict] = []
    seen_ids: set[str] = set()

    for cls in g.subjects(RDF.type, OWL.Class):
        cls_str = str(cls)
        if "OAE_" not in cls_str:
            continue

        # Extract OAE ID in OAE:XXXXXXX format
        oae_id = "OAE:" + cls_str.split("OAE_")[1]

        if oae_id in seen_ids:
            continue
        seen_ids.add(oae_id)

        # Preferred label
        labels = list(g.objects(cls, RDFS.label))
        if not labels:
            continue
        pref_label = str(labels[0])

        # Synonyms
        synonyms = []
        for syn_prop in [OBO_IN_OWL.hasExactSynonym, OBO_IN_OWL.hasBroadSynonym,
                         OBO_IN_OWL.hasNarrowSynonym, OBO_IN_OWL.hasRelatedSynonym]:
            synonyms.extend([str(s) for s in g.objects(cls, syn_prop)])

        # Definition (for richer semantic embedding)
        definitions = list(g.objects(cls, OBO_IN_OWL.hasDefinition))
        definition  = str(definitions[0]) if definitions else ""

        # Build text to embed: preferred label + synonyms + first 200 chars of definition
        syn_text = " | ".join(synonyms[:5]) if synonyms else ""
        text_to_embed = pref_label
        if syn_text:
            text_to_embed += " " + syn_text
        if definition:
            text_to_embed += " " + definition[:200]

        entries.append({
            "term_id":       oae_id,
            "term":          pref_label,
            "synonyms":      synonyms,
            "definition":    definition[:500] if definition else "",
            "text_to_embed": text_to_embed,
        })

    logger.info("Parsed %d OAE terms.", len(entries))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Embed + Build FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(entries: list[dict]) -> None:
    import faiss                          # type: ignore
    import numpy as np                    # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    logger.info("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [e["text_to_embed"] for e in entries]
    dim   = 384   # all-MiniLM-L6-v2 output dimension

    logger.info("Embedding %d terms in batches of %d ...", len(texts), _BATCH_SIZE)
    all_vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vectors.append(vecs)
        logger.info("  Embedded %d / %d", min(i + _BATCH_SIZE, len(texts)), len(texts))

    matrix = np.vstack(all_vectors).astype("float32")

    # FAISS flat inner-product index (cosine similarity on L2-normalised vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    logger.info("Writing FAISS index to %s ...", _FAISS_PATH)
    faiss.write_index(index, str(_FAISS_PATH))

    logger.info("Writing metadata to %s ...", _META_PATH)
    meta = [
        {k: v for k, v in e.items() if k != "text_to_embed"}
        for e in entries
    ]
    _META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "OAE FAISS index built: %d terms, %d dims → %s (%.1f KB)",
        index.ntotal, dim, _FAISS_PATH,
        _FAISS_PATH.stat().st_size / 1024,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if _FAISS_PATH.exists() and _META_PATH.exists():
        logger.info(
            "OAE index already exists (%s, %s). "
            "Delete them to rebuild.", _FAISS_PATH.name, _META_PATH.name
        )
        return

    download_oae()
    entries = parse_oae_owl()

    if not entries:
        logger.error("No OAE terms parsed — check oae.owl format.")
        sys.exit(1)

    build_index(entries)
    logger.info("✅ OAE FAISS index build complete.")


if __name__ == "__main__":
    main()
