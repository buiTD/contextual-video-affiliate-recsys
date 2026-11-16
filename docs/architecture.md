# Architecture

> Cập nhật Ngày 29 (v1.0) và mở rộng qua v2.0/v3.0. Sơ đồ Mermaid.js.

## TODO
- [ ] Sơ đồ tổng thể: Offline Pipeline → Supabase (pgvector) → FastAPI Zero-LLM Serving → Next.js.
- [ ] Luồng Hybrid Sandbox: SHA-256 → Semantic Cache → Gemini (Miss) → Graceful Degradation.
- [ ] Luồng Evaluation: Golden dev/test + Public Benchmark + Online Interleaving/OPE.
- [ ] Architecture Decision Records (ADR) cho các trade-off chính.

```mermaid
flowchart LR
  subgraph Offline
    A[Gemini intent extractor] --> B[gemini-embedding-001]
    B --> C[(Supabase pgvector)]
  end
  subgraph Serving
    C --> D[Stage 1: soft retrieval]
    D --> E[Stage 2: hybrid rerank + MMR]
    E --> F[Feed API zero-LLM]
  end
  F --> G[Next.js UI]
  G -->|feedback| H[(impressions / interaction_logs)]
```