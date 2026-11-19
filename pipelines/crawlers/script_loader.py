"""Load and validate structured short-video scripts with disjoint splits.

Enforces that feed / dev / test scripts share neither ids nor content, so the
evaluation sets never leak into the serving corpus.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from pipelines.crawlers.product_loader import Category

Split = Literal["feed", "dev", "test"]


class RawScript(BaseModel):
    """A structured short-video script before LLM enrichment."""

    id: str
    split: Split
    category: Category
    hook: str
    scenes: list[str] = Field(min_length=1)
    voiceover: str
    visual_cue: str

    @field_validator("id", "hook", "voiceover", "visual_cue", mode="before")
    @classmethod
    def _nfc(cls, value: object) -> object:
        if not isinstance(value, str):
            return value  # non-str must fail pydantic validation, not crash NFC
        return unicodedata.normalize("NFC", value).strip()

    @field_validator("scenes", mode="before")
    @classmethod
    def _nfc_scenes(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        return [unicodedata.normalize("NFC", s).strip() if isinstance(s, str) else s for s in value]


def load_curated_scripts(raw_path: Path) -> list[RawScript]:
    """Read raw JSON scripts, validate, and reject cross-split leakage.

    Args:
        raw_path: Path to a JSON array of script objects.

    Returns:
        Valid scripts with NFC-normalized text.

    Raises:
        ValueError: On non-array input, an invalid row, duplicate ids, or
            duplicate content across splits (data leakage).
    """
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("expected a JSON array of scripts")
    scripts: list[RawScript] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for row in rows:
        try:
            script = RawScript.model_validate(row)
        except ValidationError as exc:
            # curated corpus must be clean; a bad row is a generator bug, fail loud
            raise ValueError(f"invalid script row: {row}") from exc
        if script.id in seen_ids:
            raise ValueError(f"duplicate script id across splits: {script.id}")
        text = " ".join([script.hook, *script.scenes, script.voiceover, script.visual_cue])
        key = unicodedata.normalize("NFC", text).casefold()
        if key in seen_text:
            raise ValueError(f"duplicate script content across splits (leakage): {script.id}")
        seen_ids.add(script.id)
        seen_text.add(key)
        scripts.append(script)
    return scripts