"""Unit tests for the idempotent catalog sync (alignment, nulls, batching)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pipelines import sync_db
from pipelines.embedder import DIM, l2_normalize


def _seed(tmp_path: Path, n_products: int = 2, n_scripts: int = 2) -> Path:
    """Write a miniature catalog with the same layout the real pipeline produces."""
    raw = tmp_path / "data" / "raw"
    proc = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    rng = np.random.default_rng(0)

    products = [
        {"id": f"p{i}", "name": f"N{i}", "category": "book", "price": 1.5, "description": ""}
        for i in range(n_products)
    ]
    enriched = [
        {
            "id": p["id"],
            "target_personas": ["a"],
            "usage_scenarios": ["u"],
            "solved_problems": ["s"],
        }
        for p in products
    ]
    scripts = [
        {
            "id": f"s{i}",
            "split": "feed" if i == 0 else "test",
            "category": "book",
            "hook": f"H{i}",
            "scenes": [f"S{i}"],
            "voiceover": f"V{i}",
            "visual_cue": f"C{i}",
        }
        for i in range(n_scripts)
    ]
    script_enriched = [
        {
            "id": "s0",
            "dominant_mood": "inspirational",
            "latent_needs": ["n"],
            "recommended_categories": [{"category": "book", "weight": 0.9}],
            "bridge_copy": "b",
        }
    ]
    (raw / "products_raw.json").write_text(json.dumps(products), encoding="utf-8")
    (raw / "scripts_raw.json").write_text(json.dumps(scripts), encoding="utf-8")
    (proc / "products_enriched.json").write_text(json.dumps(enriched), encoding="utf-8")
    (proc / "scripts_enriched.json").write_text(json.dumps(script_enriched), encoding="utf-8")
    for name in ("raw", "ctx", "centered"):
        np.save(proc / f"product_emb_{name}.npy", l2_normalize(rng.normal(size=(n_products, DIM))))
    np.save(proc / "script_emb.npy", l2_normalize(rng.normal(size=(n_scripts, DIM))))
    return tmp_path


class FakeClient:
    """Captures upsert calls instead of talking to PostgREST."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict], str]] = []
        self._table = ""
        self._rows: list[dict] = []
        self._conflict = ""

    def table(self, name: str) -> FakeClient:
        self._table = name
        return self

    def upsert(self, rows: list[dict], on_conflict: str) -> FakeClient:
        self._rows = rows
        self._conflict = on_conflict
        return self

    def execute(self) -> None:
        self.calls.append((self._table, self._rows, self._conflict))


def test_product_rows_align_vectors_by_position(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    rows = sync_db.build_product_rows(root)
    matrix = np.load(root / "data" / "processed" / "product_emb_ctx.npy").astype(np.float32)
    assert len(rows) == 2
    assert rows[1]["id"] == "p1"
    assert np.array_equal(np.asarray(json.loads(rows[1]["emb_ctx"]), dtype=np.float32), matrix[1])


def test_product_rows_carry_enrichment_and_mean_version(tmp_path: Path) -> None:
    row = sync_db.build_product_rows(_seed(tmp_path))[0]
    assert row["usage_scenarios"] == ["u"]
    assert row["mean_version"] == sync_db.MEAN_VERSION
    assert len(json.loads(row["emb_raw"])) == DIM


def test_product_rows_omit_counters(tmp_path: Path) -> None:
    row = sync_db.build_product_rows(_seed(tmp_path))[0]
    assert not {"n_impr", "n_click", "n_dislike"} & row.keys()


def test_product_shape_mismatch_fails_loud(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    np.save(root / "data" / "processed" / "product_emb_ctx.npy", np.zeros((1, DIM)))
    with pytest.raises(ValueError, match="product_emb_ctx"):
        sync_db.build_product_rows(root)


def test_missing_product_enrichment_fails_loud(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    proc = root / "data" / "processed"
    (proc / "products_enriched.json").write_text("[]", encoding="utf-8")
    with pytest.raises(KeyError):
        sync_db.build_product_rows(root)


def test_unenriched_golden_scripts_keep_null_latent(tmp_path: Path) -> None:
    rows = sync_db.build_script_rows(_seed(tmp_path))
    assert rows[0]["dominant_mood"] == "inspirational"
    assert rows[1]["split"] == "test"
    assert rows[1]["latent_needs"] is None
    assert rows[1]["bridge_copy"] is None


def test_script_shape_mismatch_fails_loud(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    np.save(root / "data" / "processed" / "script_emb.npy", np.zeros((1, DIM)))
    with pytest.raises(ValueError, match="script_emb"):
        sync_db.build_script_rows(root)


def test_cross_split_duplicate_content_fails_loud(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    path = root / "data" / "raw" / "scripts_raw.json"
    scripts = json.loads(path.read_text(encoding="utf-8"))
    scripts[1].update({k: scripts[0][k] for k in ("hook", "scenes", "voiceover", "visual_cue")})
    path.write_text(json.dumps(scripts), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        sync_db.build_script_rows(root)


def test_upsert_batches_on_primary_key() -> None:
    client = FakeClient()
    rows = [{"id": str(i)} for i in range(250)]
    assert sync_db.upsert_rows(client, "products", rows) == 250
    assert [len(call[1]) for call in client.calls] == [100, 100, 50]
    assert {call[0] for call in client.calls} == {"products"}
    assert {call[2] for call in client.calls} == {"id"}


def test_rerun_produces_identical_payload(tmp_path: Path) -> None:
    root = _seed(tmp_path)
    client = FakeClient()
    sync_db.upsert_rows(client, "products", sync_db.build_product_rows(root))
    sync_db.upsert_rows(client, "products", sync_db.build_product_rows(root))
    assert client.calls[0] == client.calls[1]
