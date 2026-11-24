"""Unit tests for the embedder (L2, batching, mean-centering, text builders)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pipelines import embedder as em


def test_l2_normalize_unit_rows() -> None:
    v = np.array([[3.0, 4.0], [0.0, 5.0]])
    out = em.l2_normalize(v)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_embed_batch_l2_batches_and_normalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    chunks: list[int] = []

    class Embeddings:
        def __init__(self, n: int) -> None:
            self.embeddings = [SimpleNamespace(values=[1.0] * em.DIM) for _ in range(n)]

    class Models:
        def embed_content(self, model: str, contents: list[str], config: object) -> Embeddings:
            chunks.append(len(contents))
            return Embeddings(len(contents))

    class Client:
        models = Models()

    monkeypatch.setattr(em, "_client", lambda: Client())
    out = em.embed_batch_l2(["t"] * 250, "RETRIEVAL_DOCUMENT")
    assert chunks == [100, 100, 50]
    assert out.shape == (250, em.DIM)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_mean_center_reduces_common_direction() -> None:
    rng = np.random.default_rng(0)
    base = np.zeros((40, em.DIM))
    base[:, 0] = 1.0  # strong common direction (anisotropy)
    v = em.l2_normalize(base + rng.normal(0, 0.05, base.shape))
    centered, mean = em.mean_center(v)
    assert np.allclose(np.linalg.norm(centered, axis=1), 1.0)
    assert np.linalg.norm(centered.mean(axis=0)) < np.linalg.norm(v.mean(axis=0))
    assert mean.shape == (em.DIM,)


def test_composite_text_uses_enrichment() -> None:
    prod = {"name": "Serum C", "category": "cosmetics"}
    ctx = {"usage_scenarios": ["duong da sang"], "solved_problems": ["da xin mau"]}
    text = em.build_composite_text_for_product(prod, ctx)
    assert "Serum C" in text and "cosmetics" in text
    assert "duong da sang" in text and "da xin mau" in text


def test_script_text_joins_scenes() -> None:
    s = {"hook": "H", "scenes": ["S1", "S2"], "voiceover": "V"}
    assert em.build_script_text(s) == "H S1 S2 V"


def test_embed_429_self_throttle(monkeypatch: pytest.MonkeyPatch) -> None:
    class Err429(Exception):
        code = 429

    calls: list[int] = []

    class Models:
        def embed_content(self, model: str, contents: list[str], config: object) -> SimpleNamespace:
            calls.append(1)
            if len(calls) == 1:
                raise Err429()
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[1.0] * em.DIM) for _ in contents]
            )

    class Client:
        models = Models()

    monkeypatch.setattr(em, "_client", lambda: Client())
    slept: list[int] = []
    monkeypatch.setattr(em.time, "sleep", lambda s: slept.append(s))
    out = em.embed_batch_l2(["a", "b"], "RETRIEVAL_DOCUMENT")
    assert out.shape == (2, em.DIM)
    assert slept == [35]
    assert len(calls) == 2


def test_main_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "data" / "raw"
    proc = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    products = [
        {"id": "p0", "name": "A", "category": "book", "price": 1, "description": "d"},
        {"id": "p1", "name": "B", "category": "tech", "price": 2, "description": "d"},
    ]
    enriched = [
        {"id": "p0", "usage_scenarios": ["u"], "solved_problems": ["s"]},
        {"id": "p1", "usage_scenarios": ["u"], "solved_problems": ["s"]},
    ]
    scripts = [
        {
            "id": "s0",
            "split": "feed",
            "category": "book",
            "hook": "H",
            "scenes": ["S"],
            "voiceover": "V",
            "visual_cue": "C",
        }
    ]
    (raw / "products_raw.json").write_text(json.dumps(products), encoding="utf-8")
    (proc / "products_enriched.json").write_text(json.dumps(enriched), encoding="utf-8")
    (raw / "scripts_raw.json").write_text(json.dumps(scripts), encoding="utf-8")
    monkeypatch.setattr(em, "_embed_texts", lambda texts, task_type: np.ones((len(texts), em.DIM)))
    em.main(root=tmp_path)
    assert np.load(proc / "product_emb_raw.npy").shape == (2, em.DIM)
    assert np.load(proc / "product_emb_ctx.npy").shape == (2, em.DIM)
    assert np.load(proc / "product_emb_centered.npy").shape == (2, em.DIM)
    assert np.load(proc / "script_emb.npy").shape == (1, em.DIM)
    assert (proc / "item_mean_v1.npy").exists()


def test_jina_fallback_task_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JINA_API_KEY", "k")
    seen: dict = {}

    class Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"data": [{"embedding": [1.0] * em.DIM} for _ in seen["input"]]}

    def fake_post(url: str, **kwargs: object) -> Resp:
        seen.update(kwargs["json"])
        seen["url"] = url
        return Resp()

    monkeypatch.setattr(em.httpx, "post", fake_post)
    out = em.embed_batch_l2(["a"], "RETRIEVAL_QUERY")
    assert seen["url"] == em.JINA_URL
    assert seen["task"] == "retrieval.query"
    assert seen["dimensions"] == em.DIM
    assert out.shape == (1, em.DIM)
