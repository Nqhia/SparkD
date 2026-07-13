# SPARK DETECTION — REVIEW TỔNG HỢP

*Đánh giá thuật toán phát hiện tia lửa trong `spark_detector.py`. Hướng dẫn cài đặt & chạy: xem [README.md](README.md).*

---

## 1. Pipeline hiện tại làm gì, dùng những gì

Hệ thống phát hiện tia lửa từ **camera cố định** bằng **classical computer vision** —
không cần training, không cần GPU. Toàn bộ tham số nằm trong 1 dict `CONFIG` ở đầu file.

**Công nghệ sử dụng:** OpenCV (`cv2`) cho xử lý ảnh, NumPy cho số học, SciPy
(`linear_sum_assignment` — Hungarian matching, tùy chọn, thiếu thì fallback greedy).

**Luồng xử lý mỗi frame:**

```
Frame BGR
  │
  ├─ Grayscale + ROI mask (nếu có)
  │
  ├─ Warm-up: vài giây đầu chỉ học nền, chưa detect
  │
  ├─ Background subtraction (MOG2/KNN) ──► foreground mask
  │       (kèm guard: nếu >35% ROI đổi → coi là đổi sáng toàn cảnh, bỏ frame)
  │
  ├─ Brightness gate (ngưỡng sáng thích nghi theo nền) ──► bright mask
  ├─ Color gate (HSV: gần trắng HOẶC hue ấm)            ──► color mask
  │       combined = foreground AND bright AND color
  │
  ├─ Dilate → findContours → lọc diện tích [min_area, max_area]
  │
  ├─ Protect-learning: thay vùng spark bằng nền trước khi cho model học
  │
  ├─ Temporal tracker (centroid + dự đoán vận tốc, Hungarian/greedy)
  │       confirm nếu: tồn tại đủ lâu VÀ có di chuyển VÀ chưa quá lâu (không phải đèn tĩnh)
  │
  └─ Alert (log mọi track confirmed; cooldown chỉ gộp notification)
```

**Nguyên tắc thiết kế:** xếp chồng nhiều lớp lọc yếu (foreground → sáng → màu → diện
tích → hành vi thời gian) thành một bộ lọc mạnh, thay vì tìm một thuật toán hoàn hảo.

---

## 2. Điểm mạnh

- Kiến trúc nhiều tầng hợp lý cho một POC không cần data; mỗi tầng loại một loại nhiễu.
- Đã xử lý các cạm bẫy kinh điển của MOG2: học nền lúc khởi động (warm-up), không cho
  model nuốt spark thành nền (protect-learning), chống đổi sáng toàn cảnh, gộp báo động
  trùng (cooldown).
- Tham số thời gian tính bằng **giây** → không vỡ khi đổi FPS.
- Tracker có dự đoán vận tốc → theo được spark bay nhanh; nhãn debug cho biết track đang
  thiếu điều kiện nào để confirm.

---

## 3. Hạn chế tồn đọng

Xếp theo mức độ nên lo. Với một hệ **an toàn**, nhóm "bỏ sót" đáng sợ hơn "báo giả".

### 3.1. Báo giả (false positive)
- **Ánh sáng hắt / phản chiếu di động** — điểm sáng loé trượt trên bề mặt kim loại, hồ
  quang phản chiếu... vừa sáng vừa di chuyển → lọt. Đây là nguồn báo giả lớn nhất.
- **Người mặc đồ trắng đi lại** — trắng phá được **cả** brightness gate lẫn color gate
  (hai tầng vốn sinh ra để lọc người). Tuyến phòng thủ cuối chỉ còn `max_area`, mà cái
  đó phụ thuộc người trông to bao nhiêu pixel: người ở xa trên CCTV góc rộng, hoặc mũ
  bảo hộ/vạch phản quang trắng nhỏ → nhỏ hơn `max_area` → **lọt → báo giả**. `max_duration`
  có cắt nhưng chỉ sau ~2s (đã kịp bắn 1 alert), và mỗi lần đi lại lại tạo track mới.
- **Đèn beacon cảnh báo quay** (cam/đỏ) — hue ấm + sáng + quay (di chuyển) → qua mọi gate.
- **Côn trùng/chim ban đêm dưới đèn IR**, **artifact nén H.264**, **rung camera** (vi phạm
  giả định fixed-camera).

### 3.2. Bỏ sót spark thật (false negative) — nguy hiểm hơn
- **`max_area` nhỏ** → chùm spark lớn / spark gần camera bị loại oan.
- **`min_travel_px` bắt buộc di chuyển** → spark bắn thẳng về phía camera (gần như đứng
  yên trên ảnh 2D) bị coi là đèn tĩnh → loại.
- **Motion blur ở FPS thấp** (CCTV 12–15fps) → spark nhanh thành vệt mờ, độ sáng phân tán
  → không đạt ngưỡng sáng.
- **`min_duration` ≈ 1 frame** → spark chớp 1 frame khó tách khỏi nhiễu 1 frame.
- **Ngược sáng ban ngày** → ngưỡng sáng thích nghi bị đẩy lên trần → sót spark yếu.

### 3.3. Vận hành / kỹ thuật (chưa cần cho giai đoạn này)
- RTSP đọc đồng bộ, chưa có thread riêng / auto-reconnect; đồng hồ dùng `frame_idx/fps`
  nên lệch khi drop frame; đầu ra mới là print + JPG (chưa webhook/MQTT).

### 3.4. Vấn đề gốc: đánh đổi không đo được
Nhóm "báo giả" và "bỏ sót" **kéo ngược nhau** — siết ngưỡng để bớt báo giả thì tăng bỏ
sót và ngược lại. Hiện **không có ground truth** nên không biết đang đứng ở đâu trên đường
đánh đổi. Đây là hạn chế lớn nhất: **không đo được thì không tối ưu được.**

---

## 4. Điểm mấu chốt: tia lửa điện vs tia lửa cơ khí

Nếu mục tiêu gồm **cả hồ quang/chập điện (arc)**, cần lưu ý pipeline hiện tại được thiết
kế cho **tia lửa cơ khí** (hàn/cắt/mài) — hai loại có chữ ký gần như **ngược nhau**:

| Đặc trưng | Tia lửa **điện** (arc) | Tia lửa **cơ khí** | Tham số hiện tại giả định |
|---|---|---|---|
| Chuyển động | Chớp **tại chỗ**, gần đứng yên | Bắn ra, bay ballistic | `min_travel` → **bắt buộc di chuyển** |
| Tồn tại | Nhấp nháy lặp tại 1 điểm | Xuất hiện rồi tắt khi rơi | `max_duration` → đứng lâu = "đèn tĩnh" → loại |
| Màu | Trắng–xanh/tím | Cam/vàng (ấm) | `warm_hue_max` → giả định **màu ấm** |

→ Ba tham số lõi (`min_travel`, `max_duration→static`, `warm_hue`) đang **chủ động loại
bỏ đúng tia lửa điện**. Muốn bắt cả hai loại thì phải **đảo lại logic**, không chỉ tune nhẹ.

---

## 5. Hướng cải thiện (khi cần nâng cấp)

**Điểm chung của cả hai loại spark = một "flash":** bùng sáng đột ngột, lõi bão hòa (~255),
cục bộ, nổi trên nền ổn định. Đây là tín hiệu mạnh hơn "đủ sáng" đơn thuần, và chính là thứ
phân biệt spark thật với áo trắng / phản chiếu / beacon (những cái đó sáng **phẳng, ổn định**).

Đề xuất kiến trúc:
- **Đổi từ chuỗi cổng AND cứng → chấm điểm bằng chứng.** Mỗi ứng viên tính điểm từ:
  độ dốc bùng sáng (flash onset), lõi bão hòa, profile đỉnh nhọn + quầng (bloom), màu (nới
  cho cả cam **và** trắng-xanh). Confirm khi tổng điểm ≥ **một** ngưỡng → chỉ còn một núm
  để trượt trên đường precision–recall (thay vì nhiều van cứng, mỗi van một chỗ dễ sót).
- **Hai nhánh hành vi (OR, không AND):** nhánh **cơ khí** (bay ballistic + tắt dần + theo
  chùm) và nhánh **arc** (gần đứng yên + nhấp nháy lặp tại chỗ + lõi trắng-xanh). Khớp một
  nhánh là đủ.
- **Burst/cluster logic:** spark thật thường là **chùm nhiều hạt**; yêu cầu ≥N đốm trong
  cùng cửa sổ không–thời gian loại gần hết báo giả từ một đốm lẻ (người áo trắng, phản chiếu).

**Giới hạn cứng:** muốn *đảm bảo* độ chính xác cao nhất thì bắt buộc **có data thật để đo**
precision/recall. Không có data thì chỉ thiết kế đúng theo vật lý và kiểm trên **benchmark
tổng hợp** (mô phỏng cả hai chữ ký + các ca nhiễu khó) — synthetic bắt được lỗi logic nhưng
**recall thật vẫn là ẩn số** cho tới khi có clip thật. Khi đã có ít data, hướng cho độ chính
xác cao nhất là **classical đề xuất ứng viên + một classifier nhỏ xác minh patch**.

---

## 6. Tham số chính (tham khảo khi tune)

Tất cả nằm trong `CONFIG` đầu `spark_detector.py`. Thứ tự tune đề xuất:

1. **Ngưỡng sáng** (`adaptive_floor` / `min_absolute_brightness`) — ảnh hưởng nhiều nhất
   đến báo giả.
2. **`bg_var_threshold`** — độ nhạy background model với chuyển động nền.
3. **Tầng temporal** (`min_duration_s`, `min_travel_px`, `max_duration_s`).
4. **`max_area`** — tăng nếu spark chùm lớn bị loại oan; giảm nếu blob lớn (đèn) lọt qua.

Camera grayscale 1 kênh: đặt `use_color_gate: False` (color gate cần ảnh màu). Video ngắn
để test: giảm `bg_history` xuống 100–150 để MOG2 hội tụ kịp.
