"""Batched Gemini latent-intent extraction with JSONL checkpoint and backoff.

Model is configurable via GEMINI_MODEL (gemini-2.5-flash is retired for new
keys; default gemini-3.8-flash). thinking_budget=0 keeps latency/cost down.
Retries 429/503 with exponential backoff; checkpoint lets a run resume.
"""

from __future__ import annotations

import json
import os
import ssl
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import truststore
from google import genai
from google.genai import types
from pydantic import BaseModel

from pipelines.extractors.llm_schemas import ProductBatch, ScriptBatch

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
BATCH = 20
MAX_RETRY = 5
RETRY_CODES = {429, 503, 504}
# env MITM CA is in the OS trust store, not certifi; use system certs (keep verification)
_SSL_CTX = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def _client() -> genai.Client:
    return genai.Client(http_options=types.HttpOptions(timeout=60000))


def _retry(fn: Callable[[], object], code_of: Callable[[Exception], int | None]):
    """One retry/backoff loop for both transports; code None = network-level error."""
    for attempt in range(MAX_RETRY):
        try:
            return fn()
        except Exception as exc:
            code = code_of(exc)
            if (code in RETRY_CODES or code is None) and attempt < MAX_RETRY - 1:
                time.sleep(30 if code == 429 else 2**attempt)
                continue
            raise


def _call(client: genai.Client, prompt: str, schema: type[BaseModel]):
    cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=schema,
    )
    return _retry(
        lambda: client.models.generate_content(model=MODEL, contents=prompt, config=cfg),
        lambda exc: getattr(exc, "code", None),
    )


def _call_http(prompt: str, schema: type[BaseModel]):
    """OpenAI-compatible path (Groq/OpenRouter/etc.); json_schema + pydantic validation."""
    base = os.environ["LLM_BASE_URL"].rstrip("/")
    payload = {
        "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b"),
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "batch", "schema": schema.model_json_schema()},
        },
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}

    def once():
        resp = httpx.post(
            base + "/chat/completions",
            json=payload,
            headers=headers,
            timeout=60,
            trust_env=False,  # env NO_PROXY "[::1]" breaks httpx URL parsing
            verify=_SSL_CTX,
        )
        resp.raise_for_status()
        return SimpleNamespace(text=resp.json()["choices"][0]["message"]["content"])

    return _retry(
        once,
        lambda exc: exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None,
    )


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


def _fetch_results(
    batch: list[dict],
    schema: type[BaseModel],
    prompt_builder: Callable[[list[dict]], str],
    use_http: bool,
    client: genai.Client | None,
) -> list:
    """Fetch one batch; re-request only omitted ids (shrinking batch) before failing."""
    prompt = prompt_builder(batch)
    resp = _call_http(prompt, schema) if use_http else _call(client, prompt, schema)
    if not resp.text:
        raise ValueError("empty model response")
    parsed = schema.model_validate_json(resp.text)
    if not parsed.results:
        raise ValueError("model returned empty results")
    missing = {it["id"] for it in batch} - {rec.id for rec in parsed.results}
    if missing:
        if len(batch) == 1:
            raise ValueError(f"model omitted ids: {sorted(missing)}")
        sub = [it for it in batch if it["id"] in missing]
        return list(parsed.results) + _fetch_results(sub, schema, prompt_builder, use_http, client)
    return list(parsed.results)


def _enrich(
    items: list[dict],
    schema: type[BaseModel],
    prompt_builder: Callable[[list[dict]], str],
    checkpoint: Path,
) -> list[dict]:
    done = _load_done(checkpoint)
    pending = [it for it in items if it["id"] not in done]
    use_http = bool(os.environ.get("LLM_BASE_URL"))
    client = None if use_http else _client()
    with checkpoint.open("a", encoding="utf-8") as fh:
        for i in range(0, len(pending), BATCH):
            batch = pending[i : i + BATCH]
            for rec in _fetch_results(batch, schema, prompt_builder, use_http, client):
                row = rec.model_dump()
                done[row["id"]] = row
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return [done[it["id"]] for it in items if it["id"] in done]


def _script_prompt(batch: list[dict]) -> str:
    return (
        "For each short-video script, infer dominant_mood, latent_needs (hidden customer "
        "needs), recommended_categories (category + weight 0..1), bridge_copy (one Vietnamese "
        "sentence linking script to product). recommended_categories.category MUST be one of: "
        "book, tech, cosmetics, health, perfume, fashion. Return JSON {results:[{id,...}]}."
        "\nScripts: " + json.dumps(batch, ensure_ascii=False)
    )


def _product_prompt(batch: list[dict]) -> str:
    return (
        "For each product, infer target_personas, usage_scenarios, solved_problems. "
        "Return JSON {results:[{id,...}]}.\nProducts: " + json.dumps(batch, ensure_ascii=False)
    )


def extract_script_intents(scripts: list[dict], checkpoint: Path) -> list[dict]:
    """Enrich scripts with latent intent; resumes from checkpoint."""
    return _enrich(scripts, ScriptBatch, _script_prompt, checkpoint)


def extract_product_contexts(products: list[dict], checkpoint: Path) -> list[dict]:
    """Enrich products with usage context; resumes from checkpoint."""
    return _enrich(products, ProductBatch, _product_prompt, checkpoint)
