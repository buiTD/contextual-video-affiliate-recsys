"""Unit tests for the raw product loader (NFC, dedupe, validation)."""

import json
import unicodedata
from pathlib import Path

import pytest
from pipelines.crawlers.product_loader import load_and_validate_products


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "products_raw.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_nfc_normalization(tmp_path: Path) -> None:
    nfd_name = unicodedata.normalize("NFD", "Sách lịch sử")
    path = _write(tmp_path, [{"id": "1", "name": nfd_name, "category": "book", "price": 100000}])
    products = load_and_validate_products(path)
    assert len(products) == 1
    assert unicodedata.is_normalized("NFC", products[0].name)
    assert products[0].name == unicodedata.normalize("NFC", "Sách lịch sử")


def test_dedupe_by_casefolded_name(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "name": "Serum", "category": "cosmetics", "price": 100},
        {"id": "2", "name": "serum", "category": "cosmetics", "price": 200},
    ]
    products = load_and_validate_products(_write(tmp_path, rows))
    assert len(products) == 1
    assert products[0].id == "1"


def test_drop_invalid_price_and_category(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "name": "A", "category": "book", "price": 0},
        {"id": "2", "name": "B", "category": "toys", "price": 10},
        {"id": "3", "name": "C", "category": "tech", "price": 50},
    ]
    products = load_and_validate_products(_write(tmp_path, rows))
    assert [p.id for p in products] == ["3"]


def test_empty_catalog(tmp_path: Path) -> None:
    assert load_and_validate_products(_write(tmp_path, [])) == []


def test_non_string_name_is_skipped_not_crash(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "name": None, "category": "book", "price": 10},
        {"id": "2", "name": "OK", "category": "book", "price": 20},
    ]
    products = load_and_validate_products(_write(tmp_path, rows))
    assert [p.id for p in products] == ["2"]


def test_non_array_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "products_raw.json"
    path.write_text('{"not": "an array"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_and_validate_products(path)
