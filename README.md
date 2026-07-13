# SparkD — Phát hiện tia lửa (Spark Detection) không cần training

Hệ thống phát hiện **tia lửa** từ camera cố định bằng **classical computer vision** —
không cần dữ liệu huấn luyện, không cần GPU. Kết hợp nhiều tầng lọc yếu (background
subtraction + ngưỡng sáng + màu + diện tích + theo dõi thời gian) thành một bộ lọc mạnh.

> 📄 Review tổng hợp thuật toán, hạn chế tồn đọng và hướng cải thiện: xem **[SparkDet.md](SparkDet.md)**.

---

## 1. Yêu cầu & Cài đặt

- **Python** 3.9+
- **OpenCV**, **NumPy** (bắt buộc), **SciPy** (tùy chọn — cho Hungarian matching; thiếu sẽ
  tự dùng greedy fallback).

```bash
# (khuyến nghị) tạo virtualenv riêng
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# cài phụ thuộc
pip install -r requirements.txt
```

Kiểm tra nhanh môi trường:
```bash
python -c "import cv2, numpy; print('OpenCV', cv2.__version__, '| NumPy', numpy.__version__)"
python -c "import scipy; print('SciPy OK')"   # nếu lỗi vẫn chạy được (greedy fallback)
```

---

## 2. Bắt đầu nhanh (Quickstart)

> ⚠️ **Repo chỉ chứa code + tài liệu, KHÔNG kèm file video/ảnh** (đã `.gitignore`). Trỏ
> `--source` tới video thực tế của bạn. Trong các ví dụ, `video.mp4` chỉ là **tên minh
> họa** — thay bằng đường dẫn file thật.

Chạy trên video thực tế, xem cửa sổ debug (chậm 2× để quan sát kỹ):

```bash
python spark_detector.py --source video.mp4 --show --speed 0.5
```

Nếu không có màn hình (server/headless), bỏ `--show` và xuất video kết quả:
```bash
python spark_detector.py --source video.mp4 --save out_annotated.mp4
```

Kết thúc, chương trình in tổng số cảnh báo và mốc thời gian từng cái. Snapshot mỗi lần báo
động lưu trong thư mục `alerts/`.

---

## 3. Cách dùng (CLI)

```bash
python spark_detector.py --source <NGUỒN> [tùy chọn]
```

`--source` nhận **1 trong 3 loại**, code tự nhận diện:

| Nguồn | Ví dụ | Chế độ xử lý |
|---|---|---|
| File video | `--source duong_dan/video.mp4` | Full pipeline (bg subtraction + tracker) |
| Webcam / index | `--source 0` | Full pipeline, realtime từ webcam |
| Luồng RTSP | `--source "rtsp://user:pass@ip:554/Streaming/Channels/101"` | Full pipeline |
| **Thư mục ảnh** | `--source ./images/` | Chỉ brightness + color + area (không temporal) |

### Các cờ (flags)

| Cờ | Mặc định | Ý nghĩa |
|---|---|---|
| `--source` | *(bắt buộc)* | Video / index webcam / URL RTSP / thư mục ảnh |
| `--show` | tắt | Hiện cửa sổ debug (`spark` + `combined_mask`) |
| `--save <path>` | — | Lưu video kết quả đã vẽ annotation (mp4) |
| `--roi "x1,y1 x2,y2 ..."` | — | Khóa vùng giám sát (đa giác ≥ 3 điểm) |
| `--brightness <int>` | — | Ghi đè `min_absolute_brightness` (0–255) |
| `--speed <float>` | 1.0 | Tốc độ phát khi `--show`: `0.25` = chậm 4×, `2` = nhanh 2× (không ảnh hưởng kết quả detect) |

### Phím tắt khi `--show`
- **SPACE** — pause / tiếp tục
- Phím bất kỳ (khi đang pause) — tua đúng **1 frame**
- **q** — thoát

---

## 4. Ví dụ thường dùng

```bash
# (video.mp4 dưới đây là tên minh họa — thay bằng đường dẫn video thực tế của bạn)

# 1) Xem debug chậm 2× trên video
python spark_detector.py --source video.mp4 --show --speed 0.5

# 2) Khóa vùng ROI (chỉ giám sát khu vực máy cắt) + lưu kết quả
python spark_detector.py --source video.mp4 \
    --roi "100,50 500,50 500,400 100,400" --save out_annotated.mp4

# 3) Webcam realtime, ngưỡng sáng chặt hơn
python spark_detector.py --source 0 --show --brightness 220

# 4) Luồng RTSP Hikvision
python spark_detector.py --source "rtsp://admin:pass@192.168.1.64:554/Streaming/Channels/101" --show

# 5) Test nhanh trên thư mục ảnh tĩnh (chỉ hiệu chỉnh ngưỡng sơ bộ)
python spark_detector.py --source ./spark_images/ --show
```

---

## 5. Đọc kết quả annotation

Trên khung hình khi `--show` / `--save`:

- **Ô đỏ `#id SPARK`** — track đã được xác nhận là tia lửa → **cảnh báo**.
- **Ô vàng `#id? s=… d=…`** — ứng viên chưa đủ điều kiện confirm; `s` = span thời gian (giây),
  `d` = quãng đường di chuyển (px). Nhìn là biết đang thiếu điều kiện nào.
- **Ô xám `#id static`** — nguồn sáng tĩnh (tồn tại quá lâu) → bị loại.
- **Viền đỏ + `!! SPARK ALERT !!`** — có ít nhất 1 spark confirmed trong frame.
- **`WARM-UP...`** — đang học nền, chưa detect (vài giây đầu).
- **`GLOBAL LIGHT CHANGE - skip`** — phát hiện đổi sáng toàn cảnh, bỏ frame.
- Góc dưới: `t=<giây>` và `thr=<ngưỡng sáng hiện tại>`.

Log console mỗi lần báo động:
```
[ALERT] t=8.1s frame=121 track#3 bbox=(318, 227, 6, 6)
```

---

## 6. Tinh chỉnh (tuning)

Mọi tham số nằm trong dict `CONFIG` ở đầu [spark_detector.py](spark_detector.py). Thứ tự
tune đề xuất (chi tiết trong [SparkDet.md](SparkDet.md) mục 6):

1. **Ngưỡng sáng** — `adaptive_floor` (hoặc `--brightness` cho mode `fixed`): giảm false positive.
2. **`bg_var_threshold`** — độ nhạy background model với chuyển động nền.
3. **Tầng temporal** — `min_duration_s`, `min_travel_px`, `max_duration_s`.
4. **`max_area`** — **tăng** nếu spark là chùm lớn bị loại oan; giảm nếu blob lớn (đèn) lọt qua.

Camera grayscale 1 kênh: đặt `use_color_gate: False` (color gate cần ảnh màu).

---

## 7. Cấu trúc dự án

```
SparkD/
├── spark_detector.py     # Toàn bộ pipeline + CLI (CONFIG ở đầu file)
├── SparkDet.md           # Review tổng hợp thuật toán + hạn chế + hướng cải thiện
├── README.md             # File này — cài đặt & hướng dẫn chạy
├── requirements.txt
└── alerts/               # Snapshot mỗi lần báo động (tự tạo, đã .gitignore)
```

---

## 8. Giới hạn đã biết

- Chưa production-ready cho RTSP dài hạn: đọc frame đồng bộ, chưa có thread đọc riêng /
  auto-reconnect / đồng hồ thời gian thực (hiện dùng `frame_idx / fps`).
- Không phân biệt hình dạng — nguồn sáng nhấp nháy di chuyển đúng khoảng thời gian vẫn có thể lọt.
- Color gate vô tác dụng trên luồng grayscale 1 kênh.

Xem review đầy đủ (báo giả / bỏ sót / vận hành, và phân biệt tia lửa điện vs cơ khí) ở
[SparkDet.md](SparkDet.md) mục 3–4.
