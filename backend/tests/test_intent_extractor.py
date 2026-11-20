"""Unit tests for the batched extractor (batching, checkpoint resume, backoff)."""

import json
from pathlib import Path

import pytest
from pipelines.extractors import intent_extractor as ie


class FakeResp:
    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls = 0

    def generate_content(self, model: str, contents: str, config: object) -> FakeResp:
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResp(json.dumps(item))


class FakeClient:
    def __init__(self, responses: list) -> None:
        self.models = FakeModels(responses)


def _items(n: int) -> list[dict]:
    return [{"id": f"p{i}", "name": f"Product {i}"} for i in range(n)]


def _results(ids: list[str]) -> dict:
    return {
        "results": [
            {"id": i, "target_personas": ["x"], "usage_scenarios": ["y"], "solved_problems": ["z"]}
            for i in ids
        ]
    }


def test_batching_and_checkpoint_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _items(25)
    fake = FakeClient(
        [_results([f"p{i}" for i in range(20)]), _results([f"p{i}" for i in range(20, 25)])]
    )
    monkeypatch.setattr(ie, "_client", lambda: fake)
    cp = tmp_path / "ck.jsonl"
    out = ie.extract_product_contexts(items, cp)
    assert len(out) == 25
    assert fake.models.calls == 2
    assert len(cp.read_text(encoding="utf-8").splitlines()) == 25
    fake2 = FakeClient([])
    monkeypatch.setattr(ie, "_client", lambda: fake2)
    out2 = ie.extract_product_contexts(items, cp)
    assert len(out2) == 25
    assert fake2.models.calls == 0


def test_backoff_retries_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Err503(Exception):
        code = 503

    fake = FakeClient([Err503(), _results(["p0"])])
    monkeypatch.setattr(ie, "_client", lambda: fake)
    slept: list[int] = []
    monkeypatch.setattr(ie.time, "sleep", lambda s: slept.append(s))
    out = ie.extract_product_contexts(_items(1), tmp_path / "ck.jsonl")
    assert len(out) == 1
    assert slept == [1]
    assert fake.models.calls == 2


def test_missing_id_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient([_results(["p0"])])  # batch is p0,p1 but model returns only p0
    monkeypatch.setattr(ie, "_client", lambda: fake)
    with pytest.raises(ValueError):
        ie.extract_product_contexts(_items(2), tmp_path / "ck.jsonl")


def test_malformed_checkpoint_line_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cp = tmp_path / "ck.jsonl"
    good = {"id": "p0", "target_personas": [], "usage_scenarios": [], "solved_problems": []}
    cp.write_text(json.dumps(good) + '\n{"id": "trunc', encoding="utf-8")
    fake = FakeClient([_results(["p1"])])
    monkeypatch.setattr(ie, "_client", lambda: fake)
    out = ie.extract_product_contexts(_items(2), cp)
    assert len(out) == 2
    assert fake.models.calls == 1


def test_empty_response_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyResp:
        text = None

    class Models:
        def generate_content(self, **kwargs: object) -> EmptyResp:
            return EmptyResp()

    class Client:
        models = Models()

    monkeypatch.setattr(ie, "_client", lambda: Client())
    with pytest.raises(ValueError):
        ie.extract_product_contexts(_items(1), tmp_path / "ck.jsonl")
