# Labeling Rubric (Golden Set + Temporal)

> Viết TRƯỚC khi gán nhãn và trước khi chạy hệ thống (tránh confirmation bias).

## Relevance分级 cho Golden Set (rel ∈ {0, 1, 2})
- **2** — Sản phẩm giải quyết đúng nhu cầu ngầm của kịch bản, đúng phân khúc giá/ngữ cảnh.
- **1** — Liên quan một phần (đúng ngành nhưng lệch ngữ cảnh/giá, hoặc thay thế hợp lý).
- **0** — Không liên quan / sai nhu cầu.

## Quy trình
- TREC Blind Pooling: gộp Top-10 của nhiều biến thể thuật toán, ẩn nguồn gốc trước khi gán nhãn.
- Gán nhãn độc lập lần 2 trên tập con → Quadratic Weighted Cohen's κ.
- Kịch bản golden phải DISJOINT với 50 kịch bản Feed.

## Temporal Grounding (25 video)
- **2 người gán nhãn độc lập** khoảng [t_true_start, t_true_end] cho mỗi (video, product).
- Báo cáo κ cho nhãn temporal; bất đồng → thảo luận达成共识 hoặc loại ca đó.
- Nguồn video công khai/hợp lệ thuộc 6 ngành (ghi nguồn tại benchmark_protocol.md).

## TODO
- [ ] Chốt ngưỡng rating/co-purchase cho benchmark relevance (sau spike).
- [ ] Thêm ví dụ minh họa cho mỗi mức rel.