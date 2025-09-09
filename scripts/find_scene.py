import cv2
import numpy as np
import os

# ========================
# 설정값
# ========================
VIDEO_PATH = "examples/Golden - KPop Demon Hunters.mp4"
LOG_PATH = "results/Golden - KPop Demon Hunters.log"

WINDOW_SEC = 2.5      # 앞뒤 구간 길이(초)
THRESHOLD = 15.0      # 평균값 차이 임계치 (0~255 범위)
# 값이 낮으면 민감하게 잡고, 크면 급격한 변화만 잡음

# ========================
# 1. 비디오 읽기
# ========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

window_size = int(WINDOW_SEC * fps)  # 프레임 단위로 변환

frame_means = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    frame_means.append(mean_val)

cap.release()
frame_means = np.array(frame_means)

# ========================
# 2. 구간 평균값 차이 계산
# ========================
candidates = []
for i in range(window_size, total_frames - window_size):
    prev_mean = frame_means[i - window_size : i].mean()
    next_mean = frame_means[i : i + window_size].mean()
    diff = abs(next_mean - prev_mean)

    if diff > THRESHOLD:
        candidates.append((i, diff))

# ========================
# 3. 후보 중 가장 큰 변화량을 장면전환점으로 확정
#    (중복 구간 방지 위해 최소 간격 설정)
# ========================
scene_changes = []
min_gap = int(1.0 * fps)  # 최소 1초 간격

last_change = -min_gap
for idx, diff in candidates:
    if idx - last_change < min_gap:
        # 기존 후보와 겹치면 더 큰 변화량 선택
        if diff > scene_changes[-1][1]:
            scene_changes[-1] = (idx, diff)
    else:
        scene_changes.append((idx, diff))
        last_change = idx

# ========================
# 4. 로그 파일 저장
# ========================
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "w") as f:
    for frame_idx, diff in scene_changes:
        time_sec = frame_idx / fps
        f.write(f"{time_sec:.3f} sec (frame {frame_idx}, diff={diff:.2f})\n")

print(f"[완료] {len(scene_changes)}개의 장면 전환점을 {LOG_PATH}에 기록했습니다.")
