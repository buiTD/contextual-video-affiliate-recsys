"""Dual embeddings (raw + enriched) with L2 norm and versioned mean-centering.

Products get emb_raw (title+description) and emb_ctx (enriched composite);
scripts get a QUERY-task embedding. Mean-centering (item_mean_v{N}) is applied
to product vectors for item-item use (MMR/ILD/profile), not for context score.
Embeddings are stored as .npy (parquet deferred to the v3 scale stage).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from google import genai
from google.genai import types

EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
DIM = 768
BATCH = 100
MEAN_VERSION = 1


def _client() -> genai.Client:
    return genai.Client(http_options=types.HttpOptions(timeout=60000))


def _embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    client = _client()
    rows: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        for attempt in range(30):
            try:
                resp = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=chunk,
                    config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=DIM),
                )
                break
            except Exception as exc:
                # free tier caps embed requests/min; server asks ~33s, self-throttle
                if getattr(exc, "code", None) == 429 and attempt < 29:
                    time.sleep(35)
                    continue
                raise
        rows.extend([e.values for e in resp.embeddings])
    return np.asarray(rows, dtype=np.float64)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Scale rows to unit length so dot product equals cosine."""
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-10)


def embed_batch_l2(texts: list[str], task_type: str) -> np.ndarray:
    """Embed in batches and L2-normalize; asymmetric task_type per side."""
    return l2_normalize(_embed_texts(texts, task_type))


def mean_center(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the catalog mean and re-normalize; returns (centered, mean)."""
    mean = v.mean(axis=0)
    return l2_normalize(v - mean), mean


def build_composite_text_for_product(prod: dict, ctx: dict) -> str:
    parts = [prod["name"], prod["category"], *ctx.get("usage_scenarios", []), *ctx.get("solved_problems", [])]
    return " ".join(p for p in parts if p)


def build_script_text(script: dict) -> str:
    return " ".join([script["hook"], *script["scenes"], script["voiceover"]])


def main(root: Path | None = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    products = json.loads((root / "data" / "raw" / "products_raw.json").read_text(encoding="utf-8"))
    enriched = {r["id"]: r for r in json.loads((proc / "products_enriched.json").read_text(encoding="utf-8"))}
    scripts = json.loads((root / "data" / "raw" / "scripts_raw.json").read_text(encoding="utf-8"))
    raw_texts = [f"{p['name']} {p.get('description', '')}" for p in products]
    ctx_texts = [build_composite_text_for_product(p, enriched[p["id"]]) for p in products]
    script_texts = [build_script_text(s) for s in scripts]
    emb_raw = embed_batch_l2(raw_texts, "RETRIEVAL_DOCUMENT")
    emb_ctx = embed_batch_l2(ctx_texts, "RETRIEVAL_DOCUMENT")
    centered, mean = mean_center(emb_ctx)
    emb_script = embed_batch_l2(script_texts, "RETRIEVAL_QUERY")
    np.save(proc / "product_emb_raw.npy", emb_raw)
    np.save(proc / "product_emb_ctx.npy", emb_ctx)
    np.save(proc / "product_emb_centered.npy", centered)
    np.save(proc / "script_emb.npy", emb_script)
    np.save(proc / f"item_mean_v{MEAN_VERSION}.npy", mean)
    print("products", len(products), "scripts", len(scripts), "mean_version", MEAN_VERSION)


if __name__ == "__main__":
    main()