# -*- coding: utf-8 -*-
"""
Cấu hình mặc định cho spark_detector.py — toàn bộ tham số tune tại đây.

Khi nhúng vào project khác: copy cả 2 file (spark_detector.py + spark_config.py),
tạo config riêng bằng cách copy dict này rồi ghi đè theo từng camera:

    from spark_config import CONFIG
    cfg = dict(CONFIG)
    cfg["exclude_rects"] = [(0, 40, 640, 80)]   # vd: che OSD timestamp
    detector = SparkDetector(cfg, fps=25.0)

Các giá trị mặc định đã kiểm chứng trên camera Hikvision 1080p thật
(soak test 10 phút: 0 crash, 27fps, FP chỉ còn từ glare kính — loại nốt
bằng exclude_rects theo cảnh; recall lửa thật + giả lập 12/12).
"""

CONFIG = {
    # --- Nguồn ---
    "fps_fallback": 15.0,          # dùng khi video không khai báo FPS

    # --- ROI ---
    "roi_polygon": None,           # 1 vùng [(x, y), ...] hoặc nhiều vùng
                                   # [[(x, y), ...], ...]; None = toàn khung hình
    "exclude_rects": [],           # [(x, y, w, h), ...] vùng LOẠI TRỪ. QUAN TRỌNG với
                                   # camera thật: khoanh chữ OSD (timestamp nhảy số =
                                   # nguồn báo giả vô tận) và bề mặt kính/bóng
    "process_scale": "auto",       # "auto" = thu nhỏ sao cho chiều rộng xử lý ≤960px
                                   # (nhanh + tự lọc nhiễu nén 1px); hoặc số 0.5, 1.0...
                                   # Tọa độ config + output luôn theo ảnh GỐC

    # --- Background subtraction ---
    "bg_method": "MOG2",           # "MOG2" | "KNN"
    "bg_history": 500,
    "bg_var_threshold": 16,
    "bg_detect_shadows": False,
    "bg_protect_sparks": True,     # fix #1: không cho model học vùng spark

    # --- Brightness gate ---
    "brightness_mode": "adaptive",       # "adaptive" | "fixed"
    "min_absolute_brightness": 200,      # dùng khi "fixed", và là trần khởi tạo cho adaptive
    "adaptive_offset": 40,               # ngưỡng = p99(nền) + offset
    "adaptive_floor": 160,               # không cho ngưỡng tụt dưới mức này
    "adaptive_ceil": 250,
    "adaptive_update_every_s": 2.0,

    # --- Color gate (fix #3) ---
    "use_color_gate": True,
    "warm_hue_max": 45,            # H (OpenCV 0-180): 0-45 = đỏ/cam/vàng
    "white_sat_max": 60,           # S thấp = gần trắng → pass bất kể hue

    # --- Contour / area ---
    "min_area": 10,                # theo px ảnh GỐC. 10 đã kiểm chứng trên 1080p:
                                   # lọc hết nhiễu nén H.264 (blob 3x3px) mà vẫn bắt
                                   # được lửa thật; camera <720p có thể giảm còn 4-6
    "max_area": 400,
    "dilate_kernel_size": 3,
    "dilate_iterations": 1,

    # --- Temporal tracker (fix #2: tính bằng giây) ---
    "min_duration_s": 0.25,        # khoảng detection đầu→cuối >= mức này → xác nhận.
                                   # 0.25 kiểm chứng trên camera thật: chặn nhiễu lóe
                                   # ngắn, vẫn bắt lửa thật với độ trễ ~0.3s.
                                   # Giảm về 0.06-0.1 nếu cần bắt spark cực nhanh
    "max_duration_s": 2.0,         # tồn tại > mức này → coi là nguồn sáng tĩnh (None = tắt)
    "min_travel_px": 5,            # spark phải DI CHUYỂN; đèn bật đứng yên bị loại (0 = tắt)
    "track_base_match_dist_px": 25,
    "track_init_match_dist_px": 70,  # cổng match khi track mới 1 detection (vận tốc
                                     # chưa biết) — spark nhanh cần cổng rộng để không
                                     # bị vỡ thành track mới mỗi frame
    "track_velocity_gain": 1.5,    # fix #8: cổng match = base + gain * |v| (px/frame)
    "track_max_missed_frames": 2,

    # --- Global illumination guard (fix #5) ---
    "global_change_ratio": 0.35,   # >35% ROI là foreground → coi là đổi sáng toàn cảnh
    "global_adapt_lr": 0.05,       # learning rate tăng tốc để model thích nghi lại

    # --- Warm-up (fix #6) ---
    "warmup_s": 3.0,

    # --- Cảnh báo ---
    "alert_cooldown_s": 5.0,       # fix #9: gộp các báo động sát nhau
    "save_alert_snapshots": True,
    "snapshot_dir": "alerts",
}
