# Public Benchmark Protocol (Construct Validity)

> Bức tường chịu lực cho mọi claim Confirmatory. SPIKE 2 ngày (N31–32) trước khi tích hợp đầy đủ.
> Nếu spike cho thấy không map được schema → hạ claim, không ép benchmark vào task sai.

## Datasets & vai trò

| Dataset | Vai trò | Vì sao |
| :--- | :--- | :--- |
| **KuaiRec** (SIGIR 2022, dense fully-observed user–video) | Bandit policy value (direct, không propensity) + sequential personalization replay | Ma trận dense biết mọi reward → unbiased offline eval không cần IPS; item cluster theo video category |
| **Amazon Reviews 2023 (McAuley) subset N=2000 query** | LTR ranking + strong baseline (BM25 + multilingual cross-encoder) | Có text title/description + category + rating/co-purchase để dựng query→product ranking |

## Định nghĩa task Amazon (ghi rõ rubric)
- **Query** = text ngữ cảnh dựng từ title + description của một sản phẩm nguồn (đóng vai "kịch bản").
- **Candidates** = sản phẩm cùng category.
- **Relevance** = co-purchase graph hoặc rating ≥ ngưỡng (ghi ngưỡng cụ thể khi spike).
- Đây là task query→product ranking GENERIC, **không phải** contextual affiliate bridging.

## Ngôn ngữ
- Embedding: `gemini-embedding-001` (multilingual) cho cả live VN + benchmark EN.
- Cross-encoder baseline PHẢI multilingual, ví dụ `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Embedding ứng viên: `intfloat/multilingual-e5-small` hoặc `BAAI/bge-m3`.

## Claim split (bắt buộc ghi mọi báo cáo)
- **Confirmatory** = generic ranking/bandit machinery trên benchmark (đủ power).
- **Directional** = contextual bridging (live N=40) + temporal grounding (N=25), kèm MDE.
- Không gộp hai loại vào một claim "hệ thống đã được chứng minh".

## Fairness
- Baseline cross-encoder chạy trên RAW text; hệ thống dùng LLM-enriched.
- Tách bạch "lợi thế representation (LLM enrichment)" khỏi "lợi thế ranking algorithm" bằng
  ablation raw-vs-enriched; không gộp vào claim "đánh bại baseline".

## Spike checklist (N31–32)
- [ ] Tải KuaiRec + Amazon subset, xác nhận license/quyền sử dụng.
- [ ] Thử map vào schema; chạy thử BM25 + cross-encoder trên 50 query.
- [ ] Xác nhận benchmark exercise được thành phần nào; ghi claim split tương ứng.
- [ ] Ước lượng ESS khả dĩ cho OPE; nếu < 30 → gắn nhãn Inconclusive.