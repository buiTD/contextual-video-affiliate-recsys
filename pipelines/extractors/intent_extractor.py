"""Batched Gemini latent-intent extraction with JSONL checkpoint and backoff.

Model is configurable via GEMINI_MODEL (gemini-2.5-flash is retired for new
keys; default gemini-3.8-flash). thinking_budget=0 keeps latency/cost down.
Retries 429/503 with exponential backoff; checkpoint lets a run resume.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

from google import genai
from google.genai import types
from pydantic import BaseModel

from pipelines.extractors.llm_schemas import ProductBatch, ScriptBatch

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
BATCH = 20
MAX_RETRY = 5
RETRY_CODES = {429, 503}


def _client() -> genai.Client:
    return genai.Client(http_options=types.HttpOptions(timeout=60000))


def _call(client: genai.Client, prompt: str, schema: type[BaseModel]):
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=schema,
    )
    for attempt in range(MAX_RETRY):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
        except Exception as exc:
            if getattr(exc, "code", None) in RETRY_CODES and attempt < MAX_RETRY - 1:
                time.sleep(2**attempt)
                continue
            raise


def _load_done(checkpoint: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated tail from a killed run; keep the good lines
            done[rec["id"]] = rec
    return done


def _enrich(
    items: list[dict],
    schema: type[BaseModel],
    prompt_builder: Callable[[list[dict]], str],
    checkpoint: Path,
) -> list[dict]:
    done = _load_done(checkpoint)
    pending = [it for it in items if it["id"] not in done]
    client = _client()
    with checkpoint.open("a", encoding="utf-8") as fh:
        for i in range(0, len(pending), BATCH):
            batch = pending[i : i + BATCH]
            resp = _call(client, prompt_builder(batch), schema)
            if not resp.text:
                raise ValueError("empty model response")
            parsed = schema.model_validate_json(resp.text)
            missing = {it["id"] for it in batch} - {rec.id for rec in parsed.results}
            if missing:
                raise ValueError(f"model omitted ids: {sorted(missing)}")
            for rec in parsed.results:
                row = rec.model_dump()
                done[row["id"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [done[it["id"]] for it in items if it["id"] in done]


def _script_prompt(batch: list[dict]) -> str:
    return (
        "For each short-video script, infer dominant_mood, latent_needs (hidden customer "
        "needs), recommended_categories (category + weight 0..1), bridge_copy (one Vietnamese "
        "sentence linking script to product). Return JSON {results:[{id,...}]}.\nScripts: "
        + json.dumps(batch, ensure_ascii=False)
    )


def _product_prompt(batch: list[dict]) -> str:
    return (
        "For each product, infer target_personas, usage_scenarios, solved_problems. "
        "Return JSON {results:[{id,...}]}.\nProducts: "
        + json.dumps(batch, ensure_ascii=False)
    )


def extract_script_intents(scripts: list[dict], checkpoint: Path) -> list[dict]:
    """Enrich scripts with latent intent; resumes from checkpoint."""
    return _enrich(scripts, ScriptBatch, _script_prompt, checkpoint)


def extract_product_contexts(products: list[dict], checkpoint: Path) -> list[dict]:
    """Enrich products with usage context; resumes from checkpoint."""
    return _enrich(products, ProductBatch, _product_prompt, checkpoint)