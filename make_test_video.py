# -*- coding: utf-8 -*-
"""Tạo video test tổng hợp cho spark_detector.py — 4 tình huống:
  1. Spark thật: đốm sáng trắng-cam nhỏ, bay chéo xuống, tồn tại ~0.6s
  2. Người đi qua: khối xám lớn di chuyển ngang (phải bị brightness gate loại)
  3. Đèn bật tĩnh: đốm sáng đứng yên >2s (phải bị loại vì static light)
  4. Nhiễu sensor: chấm sáng đúng 1 frame (phải bị loại vì < min_duration)
Kỳ vọng: đúng 1 cảnh báo duy nhất (spark thật ở giây ~8).
"""
import cv2
import numpy as np

FPS = 15
W, H = 640, 480
DURATION_S = 16
N = FPS * DURATION_S

rng = np.random.default_rng(42)

# Nền xưởng: gradient tối + vài hình khối tĩnh
base = np.zeros((H, W, 3), dtype=np.uint8)
for y in range(H):
    v = 40 + int(30 * y / H)
    base[y, :] = (v, v, v + 5)
cv2.rectangle(base, (400, 250), (600, 460), (70, 65, 60), -1)   # máy
cv2.rectangle(base, (50, 300), (180, 460), (55, 60, 65), -1)    # bàn

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("test_input.mp4", fourcc, FPS, (W, H))

for i in range(N):
    t = i / FPS
    frame = base.copy()
    # Nhiễu sensor nhẹ toàn khung
    noise = rng.normal(0, 3, (H, W, 3))
    frame = np.clip(frame.astype(np.int16) + noise.astype(np.int16), 0, 255).astype(np.uint8)

    # --- 2. Người đi qua (giây 3-6): khối xám 40x100 đi ngang ---
    if 3.0 <= t <= 6.0:
        px = int(80 + (t - 3.0) / 3.0 * 400)
        cv2.rectangle(frame, (px, 200), (px + 40, 300), (120, 120, 120), -1)
        cv2.circle(frame, (px + 20, 185), 15, (110, 115, 120), -1)

    # --- 3. Đèn bật tĩnh (giây 5 → hết): đốm sáng trắng đứng yên ---
    if t >= 5.0:
        cv2.circle(frame, (520, 100), 6, (250, 250, 250), -1)

    # --- 1. Spark thật (giây 8.0-8.6): đốm trắng-cam bay chéo ---
    if 8.0 <= t <= 8.6:
        f = (t - 8.0) / 0.6
        sx = int(300 + 60 * f)
        sy = int(200 + 90 * f)
        # lõi trắng + viền cam (BGR)
        cv2.circle(frame, (sx, sy), 3, (255, 255, 255), -1)
        cv2.circle(frame, (sx + 1, sy + 1), 5, (60, 180, 255), 1)

    # --- 4. Nhiễu 1 frame (giây 11): chấm sáng chớp đúng 1 frame ---
    if i == int(11.0 * FPS):
        cv2.circle(frame, (150, 120), 3, (255, 255, 255), -1)

    out.write(frame)

out.release()
print(f"Đã tạo test_input.mp4: {N} frame, {FPS} fps, {DURATION_S}s")
print("Kỳ vọng: 1 cảnh báo duy nhất tại t≈8.1s (spark thật)")
