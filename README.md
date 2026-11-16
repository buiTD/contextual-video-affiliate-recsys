# Contextual Short-Video Affiliate RecSys

[![CI Quality](https://github.com/buiTD/contextual-video-affiliate-recsys/actions/workflows/ci-quality.yml/badge.svg)](https://github.com/buiTD/contextual-video-affiliate-recsys/actions/workflows/ci-quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

A contextual recommender that suggests affiliate products for short-video scripts.
**Zero-LLM at serving** (vector retrieval + calibrated reranking, p95 server < 40ms) with a
**hybrid LLM sandbox** for analyzing new scripts. Built on free tiers: Supabase (pgvector),
Google AI Studio (Gemini), Hugging Face Spaces, Vercel.

> Design philosophy: **statistical honesty**. Every result reports N, ESS, MDE and is labeled
> *Directional* (small sample) or *Confirmatory* (public benchmark). See [`docs/limitations.md`](docs/limitations.md).

## Milestones

| Version | Scope | Claim |
| :--- | :--- | :--- |
| **v1.0** (Day 30) | 500-product zero-LLM MVP, 10–20 real beta users | AI/ML Engineer Intern-ready |
| **v2.0** (Day 50) | TREC pooling, κ, bootstrap CI, user-clustered interleaving, dual-track eval (live + public benchmark) | Research-oriented fresher |
| **v3.0** (Day 75) | 50k catalog (int8/halfvec), SNIPS-LambdaMART, hybrid LinUCB, temporal video grounding (Hit@±2s) | Excellent thesis / Junior AI Engineer |

## Architecture (two-stage)

1. **Offline pipeline** — Gemini extracts latent intent + category weights; `gemini-embedding-001`
   (d=768, asymmetric task types) embeds scripts/products; L2-normalize + versioned mean-centering.
2. **Stage 1 retrieval** — soft candidate generation (20 in-category + 10 global), exact KNN / quantized two-pass.
3. **Stage 2 rerank** — calibrated hybrid score + hierarchical Empirical Bayes dislike penalty + mean-centered MMR;
   v3 upgrades to LightGBM LambdaMART (SNIPS) + hybrid shared-cluster LinUCB exploration.
4. **Evaluation** — offline (golden dev/test + public benchmark) and online (position-CTR, interleaving, OPE with ESS gating).

## Tech stack

FastAPI · Pydantic v2 · NumPy · Supabase (PostgreSQL + pgvector) · Gemini · LightGBM ·
Next.js (App Router) · TypeScript · Tailwind · uv · Ruff · pytest · Gitleaks · GitHub Actions · k6.

## Repository layout

```
pipelines/      offline data: crawlers, extractors, embedder, multimodal, sync_db
backend/app/    FastAPI: core, services (retrieval/reranker/ltr/bandit/profile/cache), routers
evaluation/     metrics, stats, baselines, pooling, offline/online runners, bandit_eval, k6
frontend/       Next.js 3-tab UI (feed, sandbox, dashboard)
data/           raw / processed / evaluation / models  (large artifacts on HF Datasets)
docs/           architecture, benchmark protocol, limitations, whitepaper, post-mortem
```

## Quickstart (backend)

```bash
cd backend
uv sync                      # add --extra ml for LightGBM / baselines
cp ../.env.example ../.env   # fill in Supabase + Gemini keys
uv run uvicorn app.main:app --reload --port 8000
```

Frontend is initialized on Day 19 (`create-next-app`); see [`frontend/README.md`](frontend/README.md).

## License

[MIT](LICENSE).