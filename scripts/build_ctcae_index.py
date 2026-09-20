"""
scripts/build_ctcae_index.py
=============================
Build the NCI CTCAE v5 FAISS fallback index.

Run this ONCE before starting the pipeline:
    python scripts/build_ctcae_index.py

What it does:
  1. Downloads the NCI CTCAE v5.0 Excel file from NCI EVS (public domain, US Gov)
  2. Parses the Excel to extract: MedDRA SOC, CTCAE Term, Grade descriptions
  3. Embeds all terms with sentence-transformers/all-MiniLM-L6-v2
  4. Builds a FAISS IndexFlatIP index
  5. Writes:
       data/ctcae.faiss    — binary FAISS index
       data/ctcae_meta.json — JSON array parallel to index vectors

NCI CTCAE v5.0 source:
  https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm
  Direct file: https://ctep.cancer.gov/protocoldevelopment/electronic_applications/docs/CTCAE_v5_Quick_Reference_8.5x11.xlsx
  Alternative: https://evs.nci.nih.gov/ftp1/CTCAE/CTCAE_5.0/NCIt_CTCAE_5.0.xlsx

Note on CTCAE structure:
  The Excel has columns: SOC | CTCAE Term | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Grade 5 | Definition
  We embed "CTCAE Term" + first 150 chars of "Definition" for semantic richness.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_ctcae_index")

# ─────────────────────────────────────────────────────────────────────────────
# Paths and URLs
# ─────────────────────────────────────────────────────────────────────────────

_ROOT          = Path(__file__).resolve().parent.parent
_XLSX_PATH     = _ROOT / "data" / "ctcae_v5.xlsx"
_FAISS_PATH    = _ROOT / "data" / "ctcae.faiss"
_META_PATH     = _ROOT / "data" / "ctcae_meta.json"
_BATCH_SIZE    = 256

# NCI hosts multiple mirrors; try them in order
_CTCAE_URLS = [
    "https://evs.nci.nih.gov/ftp1/CTCAE/CTCAE_5.0/NCIt_CTCAE_5.0.xlsx",
    "https://ctep.cancer.gov/protocoldevelopment/electronic_applications/docs/CTCAE_v5_Quick_Reference_8.5x11.xlsx",
]

# Known column names in the NCI Excel (may vary by mirror — we search case-insensitively)
_TERM_COL_CANDIDATES   = ["CTCAE Term", "Term", "AE Term"]
_SOC_COL_CANDIDATES    = ["MedDRA SOC", "SOC", "System Organ Class"]
_GRADE_COL_CANDIDATES  = ["Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5"]
_DEF_COL_CANDIDATES    = ["Definition", "CTCAE Definition"]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Download CTCAE Excel
# ─────────────────────────────────────────────────────────────────────────────

def download_ctcae() -> None:
    if _XLSX_PATH.exists():
        logger.info("CTCAE Excel already exists at %s — skipping download.", _XLSX_PATH)
        return

    import requests  # type: ignore

    _XLSX_PATH.parent.mkdir(parents=True, exist_ok=True)

    for url in _CTCAE_URLS:
        logger.info("Downloading CTCAE v5 from %s ...", url)
        try:
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(_XLSX_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    f.write(chunk)
            logger.info(
                "Downloaded CTCAE Excel to %s (%d bytes)",
                _XLSX_PATH, _XLSX_PATH.stat().st_size
            )
            return
        except Exception as exc:
            logger.warning("Download from %s failed: %s", url, exc)

    raise RuntimeError(
        "Failed to download CTCAE v5 Excel from NCI. "
        "Manually place the Excel file at data/ctcae_v5.xlsx "
        "from https://ctep.cancer.gov/protocoldevelopment/electronic_applications/ctc.htm"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Parse Excel
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(columns: list[str], candidates: list[str]) -> str | None:
    """Case-insensitive column name matcher."""
    lower_cols = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return None


def parse_ctcae_excel() -> list[dict]:
    """
    Parse NCI CTCAE v5 Excel.
    Returns list of dicts: {term_id, term, soc, definition, grade_descriptions, text_to_embed}.
    """
    import openpyxl  # type: ignore

    logger.info("Parsing CTCAE Excel from %s ...", _XLSX_PATH)
    wb = openpyxl.load_workbook(str(_XLSX_PATH), read_only=True, data_only=True)

    # Try each sheet — CTCAE files sometimes use different sheet names
    entries: list[dict] = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            continue

        # Find header row (first row with "CTCAE" or "Term" in it)
        header_row_idx = None
        for i, row in enumerate(rows[:10]):
            row_strs = [str(c).strip() for c in row if c]
            if any("term" in s.lower() or "ctcae" in s.lower() for s in row_strs):
                header_row_idx = i
                break

        if header_row_idx is None:
            continue

        headers    = [str(c).strip() if c else "" for c in rows[header_row_idx]]
        data_rows  = rows[header_row_idx + 1:]

        term_col = _find_col(headers, _TERM_COL_CANDIDATES)
        soc_col  = _find_col(headers, _SOC_COL_CANDIDATES)
        def_col  = _find_col(headers, _DEF_COL_CANDIDATES)

        if not term_col:
            logger.warning("Sheet %r: CTCAE Term column not found. Trying next sheet.", sheet_name)
            continue

        term_idx = headers.index(term_col)
        soc_idx  = headers.index(soc_col) if soc_col else None
        def_idx  = headers.index(def_col) if def_col else None

        grade_indices = {}
        for g in _GRADE_COL_CANDIDATES:
            g_col = _find_col(headers, [g])
            if g_col:
                grade_indices[g] = headers.index(g_col)

        seen_terms: set[str] = set()
        for row in data_rows:
            if not row or len(row) <= term_idx:
                continue
            term = str(row[term_idx]).strip() if row[term_idx] else ""
            if not term or term.lower() in ("nan", "none", ""):
                continue
            if term in seen_terms:
                continue
            seen_terms.add(term)

            soc        = str(row[soc_idx]).strip() if (soc_idx is not None and row[soc_idx]) else ""
            definition = str(row[def_idx]).strip() if (def_idx is not None and row[def_idx]) else ""

            grade_descs = []
            for g_name, g_idx in sorted(grade_indices.items()):
                if g_idx < len(row) and row[g_idx]:
                    g_text = str(row[g_idx]).strip()
                    if g_text and g_text.lower() not in ("nan", "none", "-", ""):
                        grade_descs.append(f"{g_name}: {g_text[:80]}")

            # Assign a local term ID (no universal CTCAE ID; we use SOC+term hash)
            import hashlib
            term_id = "CTCAE:" + hashlib.md5(f"{soc}|{term}".encode()).hexdigest()[:8].upper()

            text_to_embed = term
            if definition:
                text_to_embed += " " + definition[:150]

            entries.append({
                "term_id":         term_id,
                "term":            term,
                "soc":             soc,
                "definition":      definition[:500],
                "grade_descriptions": grade_descs,
                "text_to_embed":   text_to_embed,
            })

        if entries:
            logger.info("Parsed %d CTCAE terms from sheet %r.", len(entries), sheet_name)
            break

    wb.close()
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Embed + Build FAISS index
# ─────────────────────────────────────────────────────────────────────────────

def build_index(entries: list[dict]) -> None:
    import faiss                          # type: ignore
    import numpy as np                    # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    logger.info("Loading sentence-transformers model ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [e["text_to_embed"] for e in entries]
    dim   = 384

    logger.info("Embedding %d CTCAE terms in batches of %d ...", len(texts), _BATCH_SIZE)
    all_vectors = []
    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i: i + _BATCH_SIZE]
        vecs  = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vectors.append(vecs)
        logger.info("  Embedded %d / %d", min(i + _BATCH_SIZE, len(texts)), len(texts))

    matrix = np.vstack(all_vectors).astype("float32")

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    logger.info("Writing FAISS index to %s ...", _FAISS_PATH)
    faiss.write_index(index, str(_FAISS_PATH))

    logger.info("Writing metadata to %s ...", _META_PATH)
    meta = [{k: v for k, v in e.items() if k != "text_to_embed"} for e in entries]
    _META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "CTCAE FAISS index built: %d terms → %s (%.1f KB)",
        index.ntotal, _FAISS_PATH, _FAISS_PATH.stat().st_size / 1024,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if _FAISS_PATH.exists() and _META_PATH.exists():
        logger.info(
            "CTCAE index already exists (%s, %s). "
            "Delete them to rebuild.", _FAISS_PATH.name, _META_PATH.name
        )
        return

    download_ctcae()
    entries = parse_ctcae_excel()

    if not entries:
        logger.error("No CTCAE terms parsed from Excel — check file format.")
        sys.exit(1)

    build_index(entries)
    logger.info("✅ CTCAE FAISS index build complete.")


if __name__ == "__main__":
    main()
