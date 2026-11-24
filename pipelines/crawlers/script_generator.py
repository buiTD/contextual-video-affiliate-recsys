"""Generate a synthetic, curated Vietnamese short-video script corpus (90 scripts).

Composes coherent per-category story arcs x audience angles into 50 feed +
24 dev + 16 test scripts with disjoint ids/content. Synthetic by construction;
needs human QA and LLM enrichment (Day 4+) before being treated as real data.
"""

from __future__ import annotations

import json
from pathlib import Path

# (hook, scene1, scene2, voiceover, visual_cue) — coherent problem->resolution arcs
ARCS: dict[str, list[tuple[str, str, str, str, str]]] = {
    "book": [
        (
            "Một cuốn sách thay đổi cách bạn nghĩ về tiền.",
            "Nhân vật lật trang đầu, ánh mắt sáng lên.",
            "Gấp sách, nhìn ra cửa sổ suy ngẫm.",
            "Đọc không phải để biết nhiều, mà để nghĩ khác.",
            "Bìa sách cận cảnh, ánh sáng ấm.",
        ),
        (
            "Bạn đọc mãi mà không nhớ? Sai phương pháp.",
            "Highlight kín trang nhưng đầu trống rỗng.",
            "Ghi chú ba dòng, nhớ cả chương.",
            "Đọc ít lại, nhớ nhiều hơn.",
            "Sách mở và sổ ghi chú cạnh nhau.",
        ),
        (
            "Sách lịch sử mà cuốn như phim.",
            "Trang sách chuyển thành cảnh chiến trận.",
            "Nhân vật kể chuyện say sưa.",
            "Lịch sử không khô khan, cách kể mới khô.",
            "Sách sử và bản đồ cổ.",
        ),
        (
            "Quà tặng cho người thích đọc.",
            "Gói quà mở ra cuốn sách bìa cứng.",
            "Người nhận mỉm cười ôm sách.",
            "Tặng sách là tặng một thế giới.",
            "Hộp quà và sách bìa cứng.",
        ),
        (
            "Sách kỹ năng cho người mới đi làm.",
            "Nhân vật bối rối trước email đầu tiên.",
            "Đọc xong, tự tin trả lời.",
            "Kỹ năng không tự nhiên mà có.",
            "Sách kỹ năng và laptop.",
        ),
        (
            "Mỗi tối hai mươi phút, một cuốn mỗi tháng.",
            "Đồng hồ chỉ 22h, đèn bàn bật.",
            "Chồng sách dày dần theo tuần.",
            "Thói quen nhỏ, kết quả lớn.",
            "Đèn bàn và chồng sách.",
        ),
    ],
    "tech": [
        (
            "Tai nghe chống ồn cho người hay tập trung.",
            "Ồn ào quán cafe biến mất khi đeo tai nghe.",
            "Nhân vật chìm vào công việc.",
            "Im lặng là công cụ làm việc.",
            "Tai nghe cận cảnh, nền cafe mờ.",
        ),
        (
            "Bàn phím cơ gõ sướng tay cho coder.",
            "Ngón tay lướt phím, âm thanh giòn.",
            "Màn hình code chạy mượt.",
            "Công cụ tốt khiến bạn muốn làm việc.",
            "Bàn phím cơ và màn hình code.",
        ),
        (
            "Sạc dự phòng nhỏ bằng lòng bàn tay.",
            "Điện thoại 1% giữa đường.",
            "Cắm sạc, pin nhảy vọt.",
            "Đừng để hết pin phá ngày của bạn.",
            "Sạc dự phòng và điện thoại.",
        ),
        (
            "Chuột không dây cho cổ tay đỡ mỏi.",
            "Cổ tay đau vì chuột cũ.",
            "Đổi chuột công thái học, thoải mái.",
            "Sức khỏe cổ tay đáng đầu tư.",
            "Chuột công thái học và bàn làm việc.",
        ),
        (
            "Webcam rõ nét cho họp online.",
            "Hình mờ khiến bạn thiếu chuyên nghiệp.",
            "Webcam mới, hình sắc nét.",
            "Ấn tượng đầu tiên qua camera.",
            "Webcam và khung hình họp rõ.",
        ),
        (
            "Đèn màn hình chống mỏi mắt đêm khuya.",
            "Mắt mỏi dưới ánh đèn trắng.",
            "Đèn màn hình dịu, mắt dễ chịu.",
            "Làm đêm không phải trả giá bằng mắt.",
            "Đèn screenbar và màn hình đêm.",
        ),
    ],
    "cosmetics": [
        (
            "Serum vitamin C cho da xỉn màu.",
            "Da xỉn trước gương.",
            "Sau hai tuần, da sáng hơn.",
            "Sáng da không phải phép màu, là đều đặn.",
            "Chai serum và giọt serum trên tay.",
        ),
        (
            "Kem chống nắng không nhờn cho da dầu.",
            "Thoa kem, da bóng nhờn khó chịu.",
            "Kem mới thấm nhanh, lì mịn.",
            "Chống nắng mỗi ngày, kể cả trong nhà.",
            "Tuýp kem chống nắng, nền sáng.",
        ),
        (
            "Son lì màu trầm cho da ngăm.",
            "Thử son sáng màu, lệch tông.",
            "Màu trầm tôn da ngăm.",
            "Đúng màu quan trọng hơn màu hot.",
            "Thỏi son lì và swatch trên tay.",
        ),
        (
            "Toner cấp ẩm cho da khô bong tróc.",
            "Da khô căng sau rửa mặt.",
            "Toner vỗ nhẹ, da dịu tức thì.",
            "Cấp ẩm trước, dưỡng sau.",
            "Chai toner và bông cotton.",
        ),
        (
            "Kem dưỡng ẩm ban đêm phục hồi da.",
            "Da mệt mỏi cuối ngày.",
            "Thoa kem đêm, sáng da căng.",
            "Da phục hồi khi bạn ngủ.",
            "Hũ kem đêm và ánh đèn ngủ.",
        ),
        (
            "Sữa rửa mặt dịu nhẹ cho da nhạy cảm.",
            "Rửa mặt xong da rát đỏ.",
            "Sữa rửa mặt dịu, da sạch không rát.",
            "Sạch nhưng không bào mòn da.",
            "Tuýp sữa rửa mặt và bọt mịn.",
        ),
    ],
    "health": [
        (
            "Bình nước nhắc bạn uống đủ mỗi ngày.",
            "Nhức đầu nhẹ vì thiếu nước.",
            "Bình vạch ml, uống đủ hai lít.",
            "Nước là liều thuốc rẻ nhất.",
            "Bình nước vạch ml và lát chanh.",
        ),
        (
            "Vòng đếm bước cho người ngồi nhiều.",
            "Ngồi tám tiếng, lưng mỏi.",
            "Đứng dậy đi bộ, vòng đếm bước.",
            "Vận động nhỏ phá thế ngồi lâu.",
            "Vòng đeo tay và dáng đi bộ.",
        ),
        (
            "Gối công thái học cho giấc ngủ sâu.",
            "Trằn trọc vì gối quá cao.",
            "Gối mới đỡ cổ, ngủ liền mạch.",
            "Ngủ ngon bắt đầu từ cổ.",
            "Gối công thái học và giường tối.",
        ),
        (
            "Vitamin tổng hợp cho người ăn thất thường.",
            "Bữa ăn vội, thiếu chất.",
            "Viên vitamin bù phần thiếu.",
            "Ăn đủ trước, viên uống chỉ bù đắp.",
            "Lọ vitamin và bữa ăn cân bằng.",
        ),
        (
            "Thảm yoga tại nhà cho người mới.",
            "Tập sàn cứng, đau đầu gối.",
            "Thảm êm, tư thế vững.",
            "Bắt đầu nhẹ, đều mới thắng.",
            "Thảm yoga và tư thế cây cầu.",
        ),
        (
            "Máy khuếch tán tinh dầu thư giãn cuối ngày.",
            "Căng thẳng sau giờ làm.",
            "Hương tinh dầu lan tỏa, thở sâu.",
            "Thư giãn cũng là một kỹ năng.",
            "Máy khuếch tán và hơi sương.",
        ),
    ],
    "perfume": [
        (
            "Nước hoa gỗ ấm cho buổi hẹn tối.",
            "Xịt nhẹ cổ tay, hương gỗ lan.",
            "Người đối diện nghiêng lại gần.",
            "Mùi hương là chữ ký vô hình.",
            "Chai nước hoa gỗ và ánh nến.",
        ),
        (
            "Hương citrus tươi cho ngày hè.",
            "Nắng gắt, người uể oải.",
            "Xịt citrus, tinh thần bừng tỉnh.",
            "Mùi hương đổi cả tâm trạng.",
            "Chai citrus và lát cam.",
        ),
        (
            "Nước hoa hoa cỏ nhẹ cho công sở.",
            "Hương nồng gây khó chịu phòng kín.",
            "Hương hoa cỏ thoảng, dễ chịu.",
            "Đi làm nên thơm vừa đủ.",
            "Chai hoa cỏ và bàn làm việc.",
        ),
        (
            "Mùi hương vanilla ngọt cho mùa đông.",
            "Gió lạnh, người co ro.",
            "Hương vanilla ấm như ôm.",
            "Mùa đông cần một mùi ấm.",
            "Chai vanilla và khăn len.",
        ),
        (
            "Nước hoa unisex cho cặp đôi.",
            "Hai người cùng một mùi hương.",
            "Áo khoác thoảng hương chung.",
            "Mùi hương chung là kỷ niệm.",
            "Hai chai giống nhau và áo khoác.",
        ),
        (
            "Nước hoa chính hãng cho người mới chơi.",
            "Sợ mua phải hàng fake.",
            "Mua chính hãng, mã kiểm tra được.",
            "Thơm phải thật, đừng thơm giả.",
            "Chai nước hoa và tem chính hãng.",
        ),
    ],
    "fashion": [
        (
            "Áo thun cotton form rộng cho ngày thường.",
            "Áo cũ nhăn, mất form.",
            "Áo cotton mới, đứng form thoải mái.",
            "Mặc đẹp không cần đắt.",
            "Áo thun cotton gấp gọn.",
        ),
        (
            "Quần jean ống đứng tôn dáng.",
            "Quần cũ bó, khó vận động.",
            "Ống đứng vừa vặn, dễ đi.",
            "Tôn dáng bắt đầu từ vừa vặn.",
            "Quần jean ống đứng và giày trắng.",
        ),
        (
            "Áo khoác gió nhẹ cho mùa mưa.",
            "Mưa bất chợt, áo ướt.",
            "Áo khoác gió trượt nước.",
            "Nhẹ mà chắn được mưa.",
            "Áo khoác gió và giọt nước trượt.",
        ),
        (
            "Váy hoa nhí cho buổi dạo phố.",
            "Phối đồ rối, không ra chất.",
            "Váy hoa nhí và sandals, gọn gàng.",
            "Một món đúng, cả set sáng.",
            "Váy hoa nhí và sandals.",
        ),
        (
            "Túi tote canvas đựng cả thế giới.",
            "Túi nhỏ không đựng nổi laptop.",
            "Tote canvas đựng laptop và sổ.",
            "Đựng được nhiều, vẫn gọn.",
            "Túi tote canvas và laptop.",
        ),
        (
            "Giày sneakers trắng dễ phối đồ.",
            "Giày màu khó phối.",
            "Sneakers trắng cân mọi outfit.",
            "Trắng là màu dễ tính nhất.",
            "Sneakers trắng và nhiều outfit.",
        ),
    ],
}
# (hook prefix, voiceover suffix) — audience angle, keeps arcs coherent
ANGLES: list[tuple[str, str]] = [
    ("Người mới: ", " Bắt đầu từ cái nhỏ nhất."),
    ("Quà tặng: ", " Gói lại là thành món quà tinh tế."),
    ("Tiết kiệm: ", " Chọn bản vừa túi tiền, vẫn đủ chất."),
]
# per-category (feed, dev, test) counts; totals 50 / 24 / 16 = 90
COUNTS: dict[str, tuple[int, int, int]] = {
    "book": (9, 4, 3),
    "tech": (9, 4, 3),
    "cosmetics": (8, 4, 3),
    "health": (8, 4, 3),
    "perfume": (8, 4, 2),
    "fashion": (8, 4, 2),
}


def build_corpus() -> list[dict]:
    scripts: list[dict] = []
    for category, arcs in ARCS.items():
        feed_n, dev_n, test_n = COUNTS[category]
        combos = [(arc, angle) for arc in arcs for angle in ANGLES]
        idx = 0
        for split, n in (("feed", feed_n), ("dev", dev_n), ("test", test_n)):
            for _ in range(n):
                arc, (prefix, suffix) = combos[idx]
                hook, s1, s2, vo, visual = arc
                scripts.append(
                    {
                        "id": f"{category}-{idx:02d}",
                        "split": split,
                        "category": category,
                        "hook": prefix + hook,
                        "scenes": [s1, s2],
                        "voiceover": vo + suffix,
                        "visual_cue": visual,
                    }
                )
                idx += 1
    return scripts


def main() -> None:
    scripts = build_corpus()
    dest = Path(__file__).resolve().parents[2] / "data" / "raw" / "scripts_raw.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(scripts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(scripts)} scripts -> {dest}")


if __name__ == "__main__":
    main()
