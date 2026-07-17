# -*- coding: utf-8 -*-
"""
Spark Detector — pipeline classical CV, không cần training (fixed camera).

Phiên bản cải tiến từ thiết kế trong SparkDet.md, đã xử lý các vấn đề:
  #1  MOG2 học spark thành nền      → protect-learning: vùng spark được thay bằng
                                      ảnh nền trước khi cho model học
  #2  n_max_frames phụ thuộc FPS    → tham số thời gian tính bằng GIÂY, tự quy đổi
  #3  Mất thông tin màu             → color gate HSV (spark = trắng hoặc hue ấm)
  #4  Ngưỡng sáng cứng              → adaptive threshold từ ảnh nền của MOG2
  #5  AE camera / đổi sáng toàn cảnh→ global-change guard: bỏ qua frame, cho model
                                      thích nghi nhanh (không thay được việc tắt AE
                                      trên camera — vẫn nên tắt)
  #6  Cold start                    → warm-up window, không báo động khi model chưa hội tụ
  #7  Tracker swap ID               → Hungarian matching (scipy) / fallback greedy
  #8  Spark tốc độ cao              → dự đoán vị trí theo vận tốc + cổng match nới theo tốc độ
  #9  Báo động kép (phản chiếu)     → alert cooldown toàn cục

Thiết kế để NHÚNG vào project khác: phần lõi là class SparkDetector — nhận
frame BGR qua process_frame(frame, t=...), trả về dict kết quả; không tự quản
lý luồng video (project chủ tự lo capture/reconnect). Với luồng live, truyền
t = thời gian thật (giây); với video file có thể bỏ trống. process_frame có
guard chống crash: frame None/rỗng, frame grayscale, đổi độ phân giải giữa chừng.

Cách dùng standalone (test):
  python spark_detector.py --source video.mp4 --show
  python spark_detector.py --source anh_folder/ --show          (test ảnh tĩnh)
  python spark_detector.py --source video.mp4 --save out.mp4
  python spark_detector.py --source video.mp4 --roi "100,50 500,50 500,400 100,400"
"""

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# CONFIG mặc định tách riêng tại spark_config.py (nhúng vào project khác thì
# copy cả 2 file). Import kiểu kép để chạy được cả dạng script lẫn trong package.
try:
    from spark_config import CONFIG
except ImportError:
    from .spark_config import CONFIG


# =====================================================================
# Tracker
# =====================================================================
@dataclass
class Track:
    track_id: int
    centroid: tuple
    bbox: tuple                    # (x, y, w, h)
    first_t: float                 # thời điểm detection ĐẦU TIÊN (giây, theo video)
    last_t: float                  # thời điểm detection GẦN NHẤT (không tính frame missed)
    first_centroid: tuple = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    max_travel: float = 0.0        # quãng đường xa nhất tính từ điểm xuất hiện
    hits: int = 1                  # số detection thật đã khớp
    missed: int = 0
    confirmed: bool = False
    static_light: bool = False     # tồn tại quá max_duration_s
    alerted: bool = False

    def __post_init__(self):
        if self.first_centroid is None:
            self.first_centroid = self.centroid

    def span_s(self) -> float:
        """Khoảng thời gian giữa detection đầu và cuối — nhiễu 1 frame có span = 0."""
        return self.last_t - self.first_t

    def predict(self) -> np.ndarray:
        return np.array(self.centroid, dtype=float) + self.velocity


class CentroidTracker:
    """Tracker centroid với dự đoán vận tốc + Hungarian matching."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tracks: dict[int, Track] = {}
        self._next_id = 0

    def update(self, detections: list, now: float) -> list:
        """detections: [(cx, cy, bbox), ...] → trả về list Track đang sống."""
        cfg = self.cfg
        track_ids = list(self.tracks.keys())

        if detections and track_ids:
            # Ma trận chi phí: khoảng cách detection ↔ vị trí DỰ ĐOÁN của track
            cost = np.zeros((len(track_ids), len(detections)), dtype=float)
            gates = np.zeros(len(track_ids), dtype=float)
            for i, tid in enumerate(track_ids):
                tr = self.tracks[tid]
                pred = tr.predict()
                if tr.hits == 1:
                    # Track mới: vận tốc chưa ước lượng được → cổng rộng hơn
                    gates[i] = cfg["track_init_match_dist_px"]
                else:
                    gates[i] = (cfg["track_base_match_dist_px"]
                                + cfg["track_velocity_gain"] * float(np.linalg.norm(tr.velocity)))
                for j, (cx, cy, _) in enumerate(detections):
                    cost[i, j] = np.hypot(pred[0] - cx, pred[1] - cy)

            if HAS_SCIPY:
                rows, cols = linear_sum_assignment(cost)
                pairs = list(zip(rows, cols))
            else:
                # Greedy fallback: khớp cặp gần nhất trước
                pairs = []
                used_r, used_c = set(), set()
                order = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
                for r, c in order:
                    if r not in used_r and c not in used_c:
                        pairs.append((r, c))
                        used_r.add(r)
                        used_c.add(c)

            matched_dets = set()
            matched_tracks = set()
            for r, c in pairs:
                if cost[r, c] <= gates[r]:
                    tid = track_ids[r]
                    self._update_track(self.tracks[tid], detections[c], now)
                    matched_tracks.add(tid)
                    matched_dets.add(c)
        else:
            matched_dets, matched_tracks = set(), set()

        # Track không khớp → tăng missed, xóa nếu quá hạn
        for tid in track_ids:
            if tid not in matched_tracks:
                tr = self.tracks[tid]
                tr.missed += 1
                if tr.missed > cfg["track_max_missed_frames"]:
                    del self.tracks[tid]

        # Detection không khớp → track mới
        for j, (cx, cy, bbox) in enumerate(detections):
            if j not in matched_dets:
                self.tracks[self._next_id] = Track(
                    track_id=self._next_id, centroid=(cx, cy),
                    bbox=bbox, first_t=now, last_t=now)
                self._next_id += 1

        # Cập nhật trạng thái confirm / static-light
        for tr in self.tracks.values():
            # Confirm: đủ span detection thật + có di chuyển (đèn tĩnh đứng yên bị loại)
            if (not tr.static_light
                    and tr.span_s() >= cfg["min_duration_s"]
                    and tr.max_travel >= cfg["min_travel_px"]):
                tr.confirmed = True
            if (cfg["max_duration_s"] is not None
                    and (now - tr.first_t) > cfg["max_duration_s"]):
                tr.static_light = True
                tr.confirmed = False

        return list(self.tracks.values())

    def _update_track(self, tr: Track, det: tuple, now: float):
        cx, cy, bbox = det
        old = np.array(tr.centroid, dtype=float)
        new = np.array([cx, cy], dtype=float)
        # EMA vận tốc để mượt (px/frame)
        tr.velocity = 0.6 * tr.velocity + 0.4 * (new - old)
        tr.centroid = (cx, cy)
        tr.bbox = bbox
        tr.last_t = now
        tr.hits += 1
        tr.missed = 0
        travel = float(np.hypot(cx - tr.first_centroid[0], cy - tr.first_centroid[1]))
        tr.max_travel = max(tr.max_travel, travel)


# =====================================================================
# Detector
# =====================================================================
class SparkDetector:
    def __init__(self, cfg: dict, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.frame_idx = 0
        self.roi_mask = None
        self.brightness_thr = float(cfg["min_absolute_brightness"])
        self._last_adaptive_update = -1e9
        self._last_alert_t = -1e9

        # process_scale "auto" cần biết kích thước frame → khởi tạo lười ở
        # frame đầu tiên (_ensure_scaled_cfg). Tham số pixel trong config khai
        # theo ảnh GỐC, được quy đổi 1 lần sang không gian ảnh đã thu nhỏ.
        self.scale = None
        self.cfg_p = None              # config trong không gian xử lý (đã scale)
        self.tracker = None
        self.alerts = []               # log các lần báo động

        self._frame_shape = None       # để phát hiện đổi độ phân giải giữa chừng
        self.bg = self._make_bg()

        k = cfg["dilate_kernel_size"]
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    def _make_bg(self):
        cfg = self.cfg
        if cfg["bg_method"].upper() == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                history=cfg["bg_history"],
                dist2Threshold=cfg["bg_var_threshold"] * 25.0,
                detectShadows=cfg["bg_detect_shadows"])
        return cv2.createBackgroundSubtractorMOG2(
            history=cfg["bg_history"],
            varThreshold=cfg["bg_var_threshold"],
            detectShadows=cfg["bg_detect_shadows"])

    def _reset_pipeline(self):
        """Đổi độ phân giải giữa chừng (camera đổi profile stream...) →
        mask/model/tracker cũ đều vô nghĩa, khởi tạo lại thay vì crash."""
        self.bg = self._make_bg()
        self.roi_mask = None
        self.cfg_p = None
        self.tracker = None
        self.scale = None

    # -----------------------------------------------------------------
    def _ensure_scaled_cfg(self, frame_shape):
        """Khởi tạo scale + config đã quy đổi ở frame đầu tiên."""
        if self.cfg_p is not None:
            return
        cfg = self.cfg
        s_raw = cfg.get("process_scale", "auto")
        if s_raw == "auto":
            s = min(1.0, 960.0 / frame_shape[1])
        else:
            s = float(s_raw)
        self.scale = s
        cfg_p = dict(cfg)
        cfg_p["min_area"] = cfg["min_area"] * s * s
        cfg_p["max_area"] = cfg["max_area"] * s * s
        cfg_p["track_base_match_dist_px"] = cfg["track_base_match_dist_px"] * s
        cfg_p["track_init_match_dist_px"] = cfg["track_init_match_dist_px"] * s
        cfg_p["min_travel_px"] = cfg["min_travel_px"] * s
        if cfg.get("roi_polygon"):
            # Chấp nhận 1 polygon [(x,y),...] hoặc nhiều [[(x,y),...], ...] —
            # chuẩn hóa nội bộ thành list-of-polygons
            roi = cfg["roi_polygon"]
            polys = roi if isinstance(roi[0][0], (list, tuple)) else [roi]
            cfg_p["roi_polygon"] = [[(int(x * s), int(y * s)) for x, y in poly]
                                    for poly in polys]
        cfg_p["exclude_rects"] = [tuple(int(v * s) for v in r)
                                  for r in cfg.get("exclude_rects", [])]
        self.cfg_p = cfg_p
        self.tracker = CentroidTracker(cfg_p)

    def _build_roi_mask(self, shape):
        """Mask = union các polygon ROI (hoặc toàn khung) TRỪ exclude_rects."""
        if self.cfg_p.get("roi_polygon"):
            mask = np.zeros(shape[:2], dtype=np.uint8)
            for poly in self.cfg_p["roi_polygon"]:   # đã chuẩn hóa nested
                pts = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        else:
            mask = np.full(shape[:2], 255, dtype=np.uint8)
        for (x, y, w, h) in self.cfg_p.get("exclude_rects", []):
            mask[y:y + h, x:x + w] = 0
        return mask

    def _color_mask(self, frame_bgr):
        """Spark = gần trắng (S thấp) HOẶC hue ấm (đỏ/cam/vàng)."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)
        near_white = (s <= self.cfg["white_sat_max"])
        warm = (h <= self.cfg["warm_hue_max"])
        return ((near_white | warm).astype(np.uint8)) * 255

    def _update_adaptive_threshold(self, now: float):
        cfg = self.cfg
        if cfg["brightness_mode"] != "adaptive":
            return
        if now - self._last_adaptive_update < cfg["adaptive_update_every_s"]:
            return
        bg_img = self.bg.getBackgroundImage()
        if bg_img is None:
            return
        if bg_img.ndim == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        pixels = bg_img[self.roi_mask > 0] if self.roi_mask is not None else bg_img.ravel()
        if pixels.size == 0:
            return
        p99 = float(np.percentile(pixels, 99))
        self.brightness_thr = float(np.clip(
            p99 + cfg["adaptive_offset"], cfg["adaptive_floor"], cfg["adaptive_ceil"]))
        self._last_adaptive_update = now

    # -----------------------------------------------------------------
    def process_frame(self, frame_bgr, t=None):
        """Trả về dict: tracks, confirmed, alert (bool), debug masks, trạng thái.

        t: timestamp (giây) — bắt buộc truyền với luồng live (RTSP) vì frame có
        thể bị drop; mặc định None = tính từ frame_idx/fps (đúng cho file video).
        """
        cfg = self.cfg
        now = t if t is not None else self.frame_idx / self.fps
        self.frame_idx += 1

        # ---- Guards đầu vào: nhúng vào project khác thì không được crash ----
        if frame_bgr is None or frame_bgr.size == 0:
            return self._result(now, status="bad_frame")
        if frame_bgr.ndim == 2:                      # nguồn grayscale
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.shape[2] == 4:                # nguồn BGRA
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
        if (self._frame_shape is not None
                and frame_bgr.shape[:2] != self._frame_shape):
            print(f"[spark_detector] Độ phân giải đổi "
                  f"{self._frame_shape} → {frame_bgr.shape[:2]}, reset pipeline")
            self._reset_pipeline()
        self._frame_shape = frame_bgr.shape[:2]
        self._ensure_scaled_cfg(frame_bgr.shape)

        # Thu nhỏ ảnh xử lý (tọa độ output sẽ được quy về ảnh gốc khi vẽ/log)
        if self.scale != 1.0:
            frame_bgr = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale,
                                   interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        need_mask = cfg.get("roi_polygon") or cfg.get("exclude_rects")
        if need_mask and self.roi_mask is None:
            self.roi_mask = self._build_roi_mask(gray.shape)
        if self.roi_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=self.roi_mask)
        roi_area = (cv2.countNonZero(self.roi_mask)
                    if self.roi_mask is not None else gray.size)

        # ---- Warm-up (fix #6): chỉ cho model học, không phát hiện ----
        if now < cfg["warmup_s"]:
            self.bg.apply(gray)
            return self._result(now, status="warmup")

        # ---- [3] Background subtraction (lr=0: chưa học vội) ----
        fg_mask = self.bg.apply(gray, learningRate=0)
        if cfg["bg_detect_shadows"]:
            fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)[1]

        # ---- Global-change guard (fix #5) ----
        fg_ratio = cv2.countNonZero(fg_mask) / max(roi_area, 1)
        if fg_ratio > cfg["global_change_ratio"]:
            # Đổi sáng toàn cảnh: bỏ frame này, ép model thích nghi nhanh
            self.bg.apply(gray, learningRate=cfg["global_adapt_lr"])
            return self._result(now, status="global_change", fg_mask=fg_mask)

        # ---- [4] Brightness gate (adaptive, fix #4) ----
        self._update_adaptive_threshold(now)
        _, bright_mask = cv2.threshold(
            gray, self.brightness_thr, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(fg_mask, bright_mask)

        # ---- Color gate (fix #3) ----
        if cfg["use_color_gate"]:
            combined = cv2.bitwise_and(combined, self._color_mask(frame_bgr))

        # ---- [5] Dilate + contour + area filter ----
        combined = cv2.dilate(combined, self.dilate_kernel,
                              iterations=cfg["dilate_iterations"])
        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.cfg_p["min_area"] <= area <= self.cfg_p["max_area"]:
                x, y, w, h = cv2.boundingRect(c)
                detections.append((x + w / 2.0, y + h / 2.0, (x, y, w, h)))

        # ---- Protect-learning (fix #1): không cho model học vùng spark ----
        if cfg["bg_protect_sparks"] and detections:
            bg_img = self.bg.getBackgroundImage()
            if bg_img is not None:
                if bg_img.ndim == 3:
                    bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
                protect = cv2.dilate(combined, self.dilate_kernel, iterations=3)
                learn = gray.copy()
                learn[protect > 0] = bg_img[protect > 0]
                self.bg.apply(learn)
            else:
                self.bg.apply(gray)
        else:
            self.bg.apply(gray)

        # ---- [6] Temporal tracker ----
        tracks = self.tracker.update(detections, now)
        confirmed = [t for t in tracks if t.confirmed]

        # ---- Cảnh báo ----
        # Mọi track confirmed đều được LOG; cooldown (fix #9) chỉ gộp notification
        # để phản chiếu/chùm spark không tạo chuỗi báo động dồn dập.
        alert = False
        for t in confirmed:
            if not t.alerted:
                t.alerted = True
                notified = now - self._last_alert_t >= cfg["alert_cooldown_s"]
                # bbox trong log quy về tọa độ ảnh GỐC
                bbox_orig = tuple(int(round(v / self.scale)) for v in t.bbox)
                self.alerts.append({
                    "time_s": round(now, 2), "frame": self.frame_idx - 1,
                    "track_id": t.track_id, "bbox": bbox_orig,
                    "notified": notified})
                if notified:
                    alert = True
                    self._last_alert_t = now

        return self._result(now, status="ok", tracks=tracks,
                            confirmed=confirmed, alert=alert,
                            fg_mask=fg_mask, combined=combined)

    def _result(self, now, status, tracks=None, confirmed=None,
                alert=False, fg_mask=None, combined=None):
        return {
            "time_s": now, "status": status,
            "tracks": tracks or [], "confirmed": confirmed or [],
            "alert": alert, "fg_mask": fg_mask, "combined_mask": combined,
            "brightness_thr": self.brightness_thr,
            # track coords đang ở không gian đã scale (1.0 nếu chưa khởi tạo)
            "scale": self.scale if self.scale else 1.0,
        }


# =====================================================================
# Vẽ kết quả
# =====================================================================
def annotate(frame, result, cfg):
    out = frame.copy()
    inv = 1.0 / result.get("scale", 1.0)   # track coords → tọa độ ảnh gốc
    if cfg["roi_polygon"]:
        roi = cfg["roi_polygon"]
        polys = roi if isinstance(roi[0][0], (list, tuple)) else [roi]
        for poly in polys:
            cv2.polylines(out, [np.array(poly, dtype=np.int32)],
                          True, (255, 200, 0), 1)
    for (ex, ey, ew, eh) in cfg.get("exclude_rects", []):
        cv2.rectangle(out, (ex, ey), (ex + ew, ey + eh), (80, 80, 80), 1)

    for t in result["tracks"]:
        x, y, w, h = (int(round(v * inv)) for v in t.bbox)
        if t.static_light:
            color, label = (128, 128, 128), f"#{t.track_id} static"
        elif t.confirmed:
            color, label = (0, 0, 255), f"#{t.track_id} SPARK"
        else:
            # Nhãn debug: s = span detection (giây), d = quãng di chuyển (px)
            # → nhìn là biết thiếu điều kiện nào để confirm
            color = (0, 255, 255)
            label = f"#{t.track_id}? s={t.span_s():.2f} d={t.max_travel * inv:.0f}"
        cv2.rectangle(out, (x - 2, y - 2), (x + w + 2, y + h + 2), color, 2)
        cv2.putText(out, label, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    status = result["status"]
    if status == "warmup":
        cv2.putText(out, "WARM-UP...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    elif status == "global_change":
        cv2.putText(out, "GLOBAL LIGHT CHANGE - skip", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    elif result["confirmed"]:
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1),
                      (0, 0, 255), 6)
        cv2.putText(out, "!! SPARK ALERT !!", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

    cv2.putText(out, f"t={result['time_s']:.1f}s thr={result['brightness_thr']:.0f}",
                (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# =====================================================================
# Chạy trên video / stream
# =====================================================================
def run_on_stream(source, cfg, show=False, save_path=None, speed=1.0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Không mở được nguồn: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = cfg["fps_fallback"]
        print(f"[i] Không đọc được FPS, dùng fallback = {fps}")
    print(f"[i] FPS = {fps:.1f} | scipy Hungarian: {HAS_SCIPY} | "
          f"brightness mode: {cfg['brightness_mode']}")

    detector = SparkDetector(cfg, fps)
    writer = None
    if cfg["save_alert_snapshots"]:
        os.makedirs(cfg["snapshot_dir"], exist_ok=True)

    # Tốc độ phát khi --show: speed=0.25 → chậm 4 lần (không ảnh hưởng detect)
    delay_ms = max(1, int(round(1000.0 / (fps * speed))))
    paused = False
    if show:
        print(f"[i] Phát {speed}x (delay {delay_ms}ms/frame) | "
              f"SPACE = pause/tiếp, phím khác khi pause = tua 1 frame, q = thoát")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = detector.process_frame(frame)
        vis = annotate(frame, result, cfg)

        if result["alert"]:
            info = detector.alerts[-1]
            print(f"[ALERT] t={info['time_s']}s frame={info['frame']} "
                  f"track#{info['track_id']} bbox={info['bbox']}")
            if cfg["save_alert_snapshots"]:
                path = os.path.join(cfg["snapshot_dir"],
                                    f"spark_t{info['time_s']:.1f}s.jpg")
                cv2.imwrite(path, vis)

        if save_path:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(save_path, fourcc, fps,
                                         (frame.shape[1], frame.shape[0]))
            writer.write(vis)

        if show:
            cv2.imshow("spark", vis)
            if result["combined_mask"] is not None:
                cv2.imshow("combined_mask", result["combined_mask"])
            key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused
            # Khi đang pause, bất kỳ phím nào khác sẽ tua đúng 1 frame

    cap.release()
    if writer:
        writer.release()
        print(f"[i] Đã lưu video: {save_path}")
    cv2.destroyAllWindows()

    print(f"\n=== Tổng kết: {len(detector.alerts)} cảnh báo ===")
    for a in detector.alerts:
        print(f"  t={a['time_s']}s  frame={a['frame']}  track#{a['track_id']}")
    return detector.alerts


# =====================================================================
# Chạy trên thư mục ảnh tĩnh (chỉ test tầng brightness + color + area)
# =====================================================================
def run_on_images(image_dir, cfg, show=False):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = sorted(sum((glob.glob(os.path.join(image_dir, e)) for e in exts), []))
    if not paths:
        sys.exit(f"Không tìm thấy ảnh trong: {image_dir}")

    print(f"[i] {len(paths)} ảnh — chế độ ảnh tĩnh: chỉ kiểm tra "
          f"brightness + color + area (không có bg-subtraction/tracker)")
    thr = cfg["min_absolute_brightness"]
    k = cfg["dilate_kernel_size"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    stats = {"total": 0, "detected": 0}
    out_dir = os.path.join(image_dir, "annotated")
    os.makedirs(out_dir, exist_ok=True)

    for path in paths:
        frame = cv2.imread(path)
        if frame is None:
            continue
        stats["total"] += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)

        if cfg["use_color_gate"]:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, _ = cv2.split(hsv)
            cmask = (((s <= cfg["white_sat_max"]) | (h <= cfg["warm_hue_max"]))
                     .astype(np.uint8)) * 255
            mask = cv2.bitwise_and(mask, cmask)

        mask = cv2.dilate(mask, kernel, iterations=cfg["dilate_iterations"])
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if cfg["min_area"] <= area <= cfg["max_area"]:
                boxes.append(cv2.boundingRect(c))

        vis = frame.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(vis, (x - 2, y - 2), (x + w + 2, y + h + 2),
                          (0, 0, 255), 2)
        if boxes:
            stats["detected"] += 1
            cv2.putText(vis, f"SPARK x{len(boxes)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(path)), vis)

        if show:
            cv2.imshow("spark-static", vis)
            if cv2.waitKey(0) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    pct = 100 * stats["detected"] / max(stats["total"], 1)
    print(f"\nTổng ảnh: {stats['total']} | Có spark candidate: "
          f"{stats['detected']} ({pct:.1f}%)")
    print(f"Ảnh đã vẽ bbox lưu tại: {out_dir}")
    print("Lưu ý: kết quả ảnh tĩnh chỉ để hiệu chỉnh ngưỡng sơ bộ — "
          "chưa qua tầng temporal, tầng lọc quan trọng nhất.")


# =====================================================================
def parse_roi(text):
    # "100,50 500,50 500,400 100,400" → [(100,50), ...]
    pts = []
    for tok in text.split():
        x, y = tok.split(",")
        pts.append((int(x), int(y)))
    if len(pts) < 3:
        sys.exit("ROI cần ít nhất 3 điểm")
    return pts


def main():
    ap = argparse.ArgumentParser(description="Spark detector (no training)")
    ap.add_argument("--source", required=True,
                    help="đường dẫn video / index webcam / thư mục ảnh")
    ap.add_argument("--show", action="store_true", help="hiển thị cửa sổ debug")
    ap.add_argument("--save", default=None, help="lưu video kết quả (mp4)")
    ap.add_argument("--roi", default=None,
                    help='polygon ROI: "x1,y1 x2,y2 x3,y3 ..."')
    ap.add_argument("--brightness", type=int, default=None,
                    help="ghi đè min_absolute_brightness")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="tốc độ phát khi --show: 0.25 = chậm 4x, 2 = nhanh 2x")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.roi:
        cfg["roi_polygon"] = parse_roi(args.roi)
    if args.brightness is not None:
        cfg["min_absolute_brightness"] = args.brightness

    if os.path.isdir(args.source):
        run_on_images(args.source, cfg, show=args.show)
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        run_on_stream(source, cfg, show=args.show, save_path=args.save,
                      speed=args.speed)


if __name__ == "__main__":
    main()
