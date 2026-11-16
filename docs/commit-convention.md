# Quy ước commit

```text
<type>(<scope>): <summary>
```

- `type`: `feat` (chức năng), `fix` (sửa lỗi), `docs` (tài liệu), `refactor` (đổi cấu trúc), `test` (kiểm thử), `chore` (bảo trì), `build` (Docker/build), `ci` (pipeline), `perf` (hiệu năng), `revert` (hoàn tác).
- `scope` tùy chọn, theo khối monorepo:
  - Hệ thống: `repo`, `db`, `auth`, `api`, `telemetry`, `cache`, `deploy`, `ops`, `ci`.
  - RecSys: `recsys` (retrieval/reranker/profile), `ltr`, `bandit`, `eval`, `scale`, `multimodal`.
  - Dữ liệu: `pipeline`. Giao diện: `ui`.
  - Tài liệu & chiến dịch: `docs` (kèm `arch`, `release`, `paper`, `career`), `beta`.
- `summary` ngắn, tiếng Anh, bắt đầu chữ thường, không chấm cuối; tối đa 72 ký tự.

```text
feat(recsys): implement min-max calibrated hybrid scorer with bayesian smoothing
fix(bandit): weight design matrix consistently with debiased reward
test(eval): add user-level cluster bootstrap for interleaving
perf(db): tune hnsw iterative scan for 50k halfvec catalog
docs(release): finalize v1.0.0 readme and tag milestone
```

Mỗi commit chứa một thay đổi logic. Trước khi commit:
- `uv run ruff check`, `uv run pytest`, `gitleaks detect` xanh 100%; đọc diff và chạy `git diff --check`.
- Không đưa secret, `.env`, file sinh ra vào commit. Dữ liệu lớn (`*.parquet`, model `*.txt`/`*.npy`) đẩy lên Hugging Face Datasets, không commit vào git.
- File toán lõi: áp dụng test-first (viết test trước, code sau) và review kỹ trước khi commit.
- Summary trung thực thống kê: không ghi `unbiased` khi đã clip/SNIPS, không `regret converged` khi ESS < 30, không `production-scale` khi mới demo.

Breaking change: thêm `!` sau type/scope và footer `BREAKING CHANGE:` giải thích cách nâng cấp.

Milestone: gắn tag annotated `v1.0.0` / `v2.0.0` / `v3.0.0` khi phát hành.

PR ghi ngắn thay đổi và cách kiểm tra. Cập nhật tài liệu trong cùng PR khi đổi API, công thức, schema hoặc cách chạy; đổi claim thực nghiệm phải kèm `docs/limitations.md`.