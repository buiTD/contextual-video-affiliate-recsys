# Public Benchmark Data

Thư mục này chứa dữ liệu benchmark công khai. **Phần raw lớn được git-ignore** — tải về máy/local
hoặc host trên Hugging Face Datasets, không commit vào git.

## Datasets (xem docs/benchmark_protocol.md)
- **KuaiRec** — bandit/personalization (dense fully-observed).
- **Amazon Reviews 2023 (McAuley) subset N=2000** — LTR ranking + strong baseline.

## Quy ước
- `raw/` — dữ liệu gốc tải về (git-ignored).
- File processed nhỏ (samples, splits) có thể commit nếu < vài MB.
- Ghi rõ license + nguồn + ngày tải trong `benchmark_protocol.md`.
- Spike 2 ngày (N31–32) trước khi tích hợp đầy đủ.