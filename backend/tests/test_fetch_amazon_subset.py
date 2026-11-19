"""Unit tests for the Amazon subset collector (filter, dedupe, early stop)."""

import json

import pytest
from pipelines.crawlers import fetch_amazon_subset as fas


def _lines(records: list[dict]) -> list[bytes]:
    return [json.dumps(r).encode("utf-8") for r in records]


def test_collect_filters_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"parent_asin": "A", "title": "Good Book", "price": 10},
        {"parent_asin": "B", "title": "Null Price", "price": None},
        {"parent_asin": "C", "title": "Zero Price", "price": 0},
        {"parent_asin": "D", "title": "good book", "price": 20},
        {"parent_asin": "E", "title": "", "price": 5},
    ]
    monkeypatch.setattr(fas, "_stream", lambda url: iter(_lines(records)))
    out = fas.collect("book", "meta_Books.jsonl", None)
    assert [r["id"] for r in out] == ["A"]


def test_collect_keyword_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"parent_asin": "A", "title": "Perfume X", "price": 10},
        {"parent_asin": "B", "title": "Lipstick Y", "price": 10},
    ]
    monkeypatch.setattr(fas, "_stream", lambda url: iter(_lines(records)))
    out = fas.collect("perfume", "meta_All_Beauty.jsonl", ("perfume",))
    assert [r["id"] for r in out] == ["A"]


def test_collect_early_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [
        {"parent_asin": f"A{i}", "title": f"Book {i}", "price": 10}
        for i in range(fas.PER_CATEGORY + 10)
    ]
    monkeypatch.setattr(fas, "_stream", lambda url: iter(_lines(records)))
    out = fas.collect("book", "meta_Books.jsonl", None)
    assert len(out) == fas.PER_CATEGORY
