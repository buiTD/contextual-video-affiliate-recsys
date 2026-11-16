# Known Limitations (công khai — bắt buộc)

> Nguyên tắc: mọi tuyên bố phải tương xứng cỡ mẫu. File này liệt kê các giới hạn đã biết để
> không claim vượt bằng chứng. Cập nhật mỗi khi cắt giảm scope hoặc phát hiện giới hạn mới.

1. **Contextual bridging chỉ Directional** (live N=40 kịch bản). Public benchmark không exercise
   được đóng góp latent-intent/bridge-copy; chỉ validate machinery xếp hạng generic.
2. **Temporal grounding N=25 video**, nhãn 2 người (có κ) nhưng vẫn nhỏ → Hit@±2s CI rộng.
3. **Bandit reward sau propensity-weight không bị chặn [0,1]** → ellipsoid LinUCB là xấp xỉ;
   `√(u_shared + u_cluster)` cộng hai phương sai là heuristic (dữ liệu chồng nhau → tương quan).
4. **OPE trên KuaiRec dense là direct eval** (biết mọi reward, không cần propensity). IPS/DR trên
   benchmark chạy với logging policy giả lập ε-greedy → semi-synthetic estimator-validation,
   không phải regret thật.
5. **LTR live là demo-scale** (1200 cặp, regularization mạnh: num_leaves=7, min_child_samples=25).
   Sức mạnh thật chứng minh trên benchmark generic, không suy ra cho production.
6. **Free tier:** Supabase 500MB (1 cột halfvec + parquet offload ~240MB), HF Spaces ngủ 48h,
   Colab ngắt phiên (checkpoint xuyên phiên), Render 512MB (ưu tiên HF 16GB).
7. **Ngôn ngữ:** live tiếng Việt + benchmark tiếng Anh; cross-encoder baseline BẮT BUỘC multilingual
   để không dựng strawman yếu.
8. **φ5 (multi-interest) time-decay phi dừng có chủ đích** → giám sát PSI, không chỉ parity giá trị.
9. **Clipped IPS / SNIPS cố ý chấp nhận bias nhỏ** để giảm variance → KHÔNG gọi là "unbiased".

## Scope cuts (nếu tràn giờ)
- Temporal video engine có thể hạ xuống 5 video demo.
- 50k catalog có thể hạ 10k.
- Mọi sự cắt giảm ghi tại đây, KHÔNG cắt phần thống kê trung thực.