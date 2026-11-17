# Engineering Guidelines

Coding discipline for everyone working on this repository — human or AI agent.
Adapted from Andrej Karpathy's observations on common LLM coding pitfalls
(inspiration: [multica-ai/andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills));
the wording and project-specific rules below are original to this repository.

## 1. Think before coding
- State assumptions explicitly. If a requirement, metric definition, or data schema
  is ambiguous, ask instead of guessing.
- Surface tradeoffs and present alternatives when a decision is non-obvious.
- Push back when a simpler approach exists; do not silently over-build.
- Stop and name what is unclear rather than running with a hidden interpretation.

## 2. Simplicity first
- Write the minimum code that solves the stated problem; nothing speculative.
- No abstractions for single-use code, no unrequested "flexibility" or "configurability".
- No error handling for impossible scenarios; trust internal invariants.
- Prefer clear NumPy vectorization over loops, but do not prematurely optimize.
- Respect the $0 free-tier constraint: no heavy infra (Kafka, Spark, Kubernetes, paid services).
- If 200 lines could be 50, rewrite it.

## 3. Surgical changes
- Touch only what the task requires; every changed line should trace to the request.
- Match existing style (Ruff, full type hints, Google-style docstrings).
- Do not refactor or "improve" adjacent, unrelated code.
- Remove only the imports/variables/functions YOUR change orphaned; leave pre-existing
  dead code alone (mention it instead of deleting).

## 4. Goal-driven execution
- Turn tasks into verifiable goals: write the failing test first, then make it pass.
- For core math (metrics, reranker, user profile, stats, interleaving, bandit):
  red tests before implementation.
- State a short plan with a verify step per item for multi-step work.
- Before committing run: `ruff check`, `ruff format --check`, `pytest` (>=1 test),
  `gitleaks`. Keep `main` CI green; fix forward promptly.

## Project hard rules
- Package management: `uv` only. Input/output validation: Pydantic v2.
- Conventional Commits; see [docs/commit-convention.md](docs/commit-convention.md).
- Never commit secrets, `.env`, or large artifacts (`*.parquet`, model weights) —
  host datasets on Hugging Face Datasets.
- Statistical honesty: report N / ESS / MDE and label results Directional vs
  Confirmatory; never claim "unbiased" after clipping/SNIPS, or "regret converged"
  when ESS < 30. See [docs/limitations.md](docs/limitations.md).