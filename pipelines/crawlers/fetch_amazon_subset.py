"""Fetch a small public Amazon-Reviews-2023 metadata subset (500 products).

Streams only the prefix of each category metadata file (early stop), filters
rows with a positive price, and maps Amazon categories onto the project's six
categories. Provenance: McAuley-Lab/Amazon-Reviews-2023 (Hugging Face).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

BASE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/meta_categories/"
)
# (project category, source file, optional title keywords to isolate a niche)
SOURCES: list[tuple[str, str, tuple[str, ...] | None]] = [
    ("book", "meta_Books.jsonl", None),
    ("tech", "meta_Electronics.jsonl", None),
    ("cosmetics", "meta_Beauty_and_Personal_Care.jsonl", None),
    ("health", "meta_Health_and_Household.jsonl", None),
    ("perfume", "meta_All_Beauty.jsonl", ("perfume", "fragrance", "eau de", "cologne", "parfum")),
    ("fashion", "meta_Amazon_Fashion.jsonl", None),
]
PER_CATEGORY = 84
TOTAL = 500


def _stream(url: str):
    """Yield lines from a remote file, terminating the download early."""
    proc = subprocess.Popen(
        ["curl", "-sL", "--fail", "--max-time", "120", url], stdout=subprocess.PIPE
    )
    try:
        for line in proc.stdout:
            yield line
    finally:
        proc.kill()


def collect(category: str, filename: str, keywords: tuple[str, ...] | None) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for line in _stream(BASE + filename):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = (rec.get("title") or "").strip()
        price = rec.get("price")
        if not title or not isinstance(price, (int, float)) or price <= 0:
            continue
        if keywords and not any(k in title.lower() for k in keywords):
            continue
        key = title.casefold()
        if key in seen:
            continue
        seen.add(key)
        desc = rec.get("description") or []
        if isinstance(desc, list):
            desc = " ".join(desc)
        out.append(
            {
                "id": rec.get("parent_asin") or rec.get("asin"),
                "name": title,
                "category": category,
                "price": float(price),
                "description": str(desc)[:500],
            }
        )
        if len(out) >= PER_CATEGORY:
            break
    return out


def main() -> None:
    products: list[dict] = []
    for category, filename, keywords in SOURCES:
        rows = collect(category, filename, keywords)
        if not rows:
            raise RuntimeError(f"no rows fetched for {category}")
        print(f"{category}: {len(rows)}")
        products.extend(rows)
    products = products[:TOTAL]
    dest = Path(__file__).resolve().parents[2] / "data" / "raw" / "products_raw.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(products, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(products)} products -> {dest}")


if __name__ == "__main__":
    main()