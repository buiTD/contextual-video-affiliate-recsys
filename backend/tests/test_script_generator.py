"""Unit tests for the script corpus generator (split counts, disjoint ids/text)."""

from collections import Counter

from pipelines.crawlers.script_generator import build_corpus


def test_split_counts_and_disjoint() -> None:
    scripts = build_corpus()
    assert len(scripts) == 90
    assert Counter(s["split"] for s in scripts) == {"feed": 50, "dev": 24, "test": 16}
    ids = [s["id"] for s in scripts]
    texts = [" ".join([s["hook"], *s["scenes"], s["voiceover"], s["visual_cue"]]) for s in scripts]
    assert len(set(ids)) == 90
    assert len(set(texts)) == 90
