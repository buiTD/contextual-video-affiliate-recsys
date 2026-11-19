"""Unit tests for the script loader (NFC, validation, cross-split leakage)."""

import json
import unicodedata
from pathlib import Path

import pytest
from pipelines.crawlers.script_loader import load_curated_scripts


def _script(sid: str, split: str, hook: str = "Hook mặc định") -> dict:
    return {
        "id": sid,
        "split": split,
        "category": "book",
        "hook": hook,
        "scenes": ["Cảnh một", "Cảnh hai"],
        "voiceover": "Lời thoại",
        "visual_cue": "Gợi ý hình",
    }


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "scripts_raw.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_loads_and_normalizes_nfc(tmp_path: Path) -> None:
    nfd_hook = unicodedata.normalize("NFD", "Sách hay")
    path = _write(tmp_path, [_script("s1", "feed", nfd_hook)])
    scripts = load_curated_scripts(path)
    assert len(scripts) == 1
    assert unicodedata.is_normalized("NFC", scripts[0].hook)


def test_disjoint_splits_ok(tmp_path: Path) -> None:
    rows = [
        _script("a", "feed", "Hook A"),
        _script("b", "dev", "Hook B"),
        _script("c", "test", "Hook C"),
    ]
    scripts = load_curated_scripts(_write(tmp_path, rows))
    assert {s.split for s in scripts} == {"feed", "dev", "test"}


def test_duplicate_id_across_splits_raises(tmp_path: Path) -> None:
    rows = [_script("a", "feed", "Hook A"), _script("a", "dev", "Hook B")]
    with pytest.raises(ValueError):
        load_curated_scripts(_write(tmp_path, rows))


def test_duplicate_content_across_splits_raises(tmp_path: Path) -> None:
    rows = [_script("a", "feed", "Hook giống nhau"), _script("b", "dev", "Hook giống nhau")]
    with pytest.raises(ValueError):
        load_curated_scripts(_write(tmp_path, rows))


def test_invalid_scenes_skipped(tmp_path: Path) -> None:
    bad = _script("a", "feed")
    bad["scenes"] = []
    rows = [bad, _script("b", "dev", "Hook B")]
    scripts = load_curated_scripts(_write(tmp_path, rows))
    assert [s.id for s in scripts] == ["b"]


def test_non_array_top_level_raises(tmp_path: Path) -> None:
    path = tmp_path / "scripts_raw.json"
    path.write_text('{"not": "array"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_curated_scripts(path)
