# Technical Whitepaper

> Biên soạn Ngày 73 (v3.0). Cấu trúc bài báo kỹ thuật thực nghiệm.

## Mục lục (khung)
1. Abstract
2. Introduction & Contributions (kèm claim split Directional/Confirmatory)
3. System Architecture (v1.0 → v2.0 → v3.0)
4. Mathematical Formulation (Mục 4 của master plan)
5. Experimental Setup (datasets, baselines, metrics, giả định PBM)
6. Results & Discussion (Dual-Track Ablation, Interleaving, OPE + ESS, HNSW recall, Hit@±2s)
7. **Limitations** (bắt buộc — đồng bộ với docs/limitations.md)
8. Reproducibility (seeds, pinned versions, one-command eval)

## TODO
- [ ] Bảng ablution kèm 95% CI + Holm–Bonferroni.
- [ ] Phân tích độ nhạy propensity (0.5×, 2×).
- [ ] Thảo luận construct validity của benchmark.