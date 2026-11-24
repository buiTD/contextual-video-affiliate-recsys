"""Idempotent sync of the offline catalog into Supabase (products + scripts).

Vectors are stored as .npy in the same row order as the raw JSON, so alignment is
positional and a shape mismatch must fail loud rather than silently mislabel rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from supabase import Client, create_client

from pipelines.crawlers.product_loader import load_and_validate_products
from pipelines.crawlers.script_loader import load_curated_scripts
from pipelines.embedder import DIM, MEAN_VERSION

BATCH = 100


def _vector_literal(v: np.ndarray) -> str:
    """pgvector text input; 9 significant digits round-trip a float32 exactly."""
    return "[" + ",".join(f"{x:.9g}" for x in v.astype(np.float32)) + "]"


def build_product_rows(root: Path) -> list[dict]:
    """Join the validated catalog, enrichment and the three embedding matrices."""
    proc = root / "data" / "processed"
    products = load_and_validate_products(root / "data" / "raw" / "products_raw.json")
    enriched = {
        r["id"]: r
        for r in json.loads((proc / "products_enriched.json").read_text(encoding="utf-8"))
    }
    emb = {k: np.load(proc / f"product_emb_{k}.npy") for k in ("raw", "ctx", "centered")}
    for name, matrix in emb.items():
        if matrix.shape != (len(products), DIM):
            raise ValueError(f"product_emb_{name} shape {matrix.shape} != {(len(products), DIM)}")

    rows = []
    for i, product in enumerate(products):
        ctx = enriched[product.id]
        rows.append(
            {
                "id": product.id,
                "name": product.name,
                "category": product.category,
                "price": product.price,
                "description": product.description,
                "target_personas": ctx["target_personas"],
                "usage_scenarios": ctx["usage_scenarios"],
                "solved_problems": ctx["solved_problems"],
                "emb_raw": _vector_literal(emb["raw"][i]),
                "emb_ctx": _vector_literal(emb["ctx"][i]),
                "emb_centered": _vector_literal(emb["centered"][i]),
                "mean_version": MEAN_VERSION,
            }
        )
    # no counter columns: re-syncing the catalog must not reset beta statistics
    return rows


def build_script_rows(root: Path) -> list[dict]:
    """Join scripts with their query embedding; unenriched goldens stay NULL."""
    proc = root / "data" / "processed"
    scripts = load_curated_scripts(root / "data" / "raw" / "scripts_raw.json")
    enriched = {
        r["id"]: r for r in json.loads((proc / "scripts_enriched.json").read_text(encoding="utf-8"))
    }
    emb = np.load(proc / "script_emb.npy")
    if emb.shape != (len(scripts), DIM):
        raise ValueError(f"script_emb shape {emb.shape} != {(len(scripts), DIM)}")

    rows = []
    for i, script in enumerate(scripts):
        ctx = enriched.get(script.id)
        rows.append(
            {
                "id": script.id,
                "split": script.split,
                "category": script.category,
                "hook": script.hook,
                "scenes": script.scenes,
                "voiceover": script.voiceover,
                "visual_cue": script.visual_cue,
                "emb_query": _vector_literal(emb[i]),
                "dominant_mood": ctx["dominant_mood"] if ctx else None,
                "latent_needs": ctx["latent_needs"] if ctx else None,
                "recommended_categories": ctx["recommended_categories"] if ctx else None,
                "bridge_copy": ctx["bridge_copy"] if ctx else None,
            }
        )
    return rows


def upsert_rows(client: Client, table: str, rows: list[dict]) -> int:
    """Upsert on the primary key in batches, so a re-run updates instead of duplicating."""
    for i in range(0, len(rows), BATCH):
        client.table(table).upsert(rows[i : i + BATCH], on_conflict="id").execute()
    return len(rows)


def main(root: Path | None = None) -> None:
    if root is None:
        root = Path(__file__).resolve().parents[1]
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    products = build_product_rows(root)
    scripts = build_script_rows(root)
    upsert_rows(client, "products", products)
    upsert_rows(client, "scripts", scripts)
    print("products", len(products), "scripts", len(scripts), "mean_version", MEAN_VERSION)


if __name__ == "__main__":
    main()
