"""Load and validate the raw product catalog.

Normalizes Unicode to NFC, drops invalid rows (non-positive price, unknown
category) and de-duplicates by product name.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

Category = Literal["book", "tech", "cosmetics", "health", "perfume", "fashion"]


class RawProduct(BaseModel):
    """A single raw catalog row before LLM enrichment."""

    id: str
    name: str
    category: Category
    price: float = Field(gt=0)
    description: str = ""

    @field_validator("id", "name", "description", mode="before")
    @classmethod
    def _normalize_nfc(cls, value: str) -> str:
        """Normalize Vietnamese text to NFC so hashing/dedupe are stable."""
        return unicodedata.normalize("NFC", value).strip()


def load_and_validate_products(raw_path: Path) -> list[RawProduct]:
    """Read raw JSON and return clean, de-duplicated products.

    Args:
        raw_path: Path to a JSON array of raw product objects.

    Returns:
        Valid products, NFC-normalized, unique by case-folded name.
    """
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    seen: set[str] = set()
    products: list[RawProduct] = []
    for row in rows:
        try:
            product = RawProduct.model_validate(row)
        except ValidationError:
            continue
        key = product.name.casefold()
        if key in seen:
            continue
        seen.add(key)
        products.append(product)
    return products