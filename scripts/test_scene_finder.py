#!/usr/bin/env python3
"""
scene_detect.py

Usage:
    python scene_detect.py input_video.mp4 output_log.txt

Description:
    - Detects abrupt cuts and gradual fades/dissolves.
    - Logs timestamps to output_log.txt.
    - Adjustable thresholds in the PARAMETERS section below.
"""

import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import math
import datetime

# ----------------------------
# PARAMETERS (튜닝 가능한 값)
# ----------------------------
RESIZE_WIDTH = 320          # 속도 위해 프레임 리사이즈(원본이 클 경우)
HIST_BINS = [32, 32, 32]    # HSV 히스토그램 빈 (H, S, V)
CUT_THRESHOLD = 0.6         # 히스토그램 비교(상관도)에서 '컷'으로 판단할 임계값 (작을수록 민감: 0..1)
SSIM_CUT_THRESHOLD = 0.35   # SSIM 값이 이보다 작으면 급격한 전환(컷)으로 간주
FADE_WINDOW = 30            # 페이드 감지에 사용할 프레임 길이(약 1~2초 권장, FPS에 따라 조절)
FADE_SUM_DIFF_THRESHOLD = 8.0   # 페이드 후보: 윈도우 내 히스토그램 차이 합(경험적 값)
FADE_MAX_SINGLE_DIFF = 0.9  # 페이드 여부 판단 시 윈도우 내 단일 프레임이 컷 수준이면 페이드가 아님
MIN_FADE_LENGTH_FRAMES = 8  # 페이드로 인정할 최소 연속 프레임 수
# ----------------------------

def timestamp_from_frame(frame_idx, fps):
    seconds = frame_idx / fps
    td = datetime.timedelta(seconds=seconds)
    # format hh:mm:ss.mmm
    total_seconds = seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def compute_hsv_hist(frame, bins=HIST_BINS):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256])
    hist = np.concatenate([h.flatten(), s.flatten(), v.flatten()])
    # 정규화
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist

def hist_compare_chi2(h1, h2):
    # OpenCV의 CHISQR 유사 방식. 값이 작을수록 비슷함.
    # 우리는 "유사도"처럼 작을수록 같음 -> 1/(1+chi)로 0..1 유사도 변환
    eps = 1e-10
    chi = 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))
    sim = 1.0 / (1.0 + chi)   # 1에 가까우면 비슷
    return sim

def compute_ssim_gray(a, b):
    # a, b : grayscale images
    s = ssim(a, b)
    return s

def detect_scenes(video_path, out_log_path):
    print(f"[DEBUG] Opening video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    resize_w = RESIZE_WIDTH

    prev_hist = None
    prev_gray = None

    # For fade detection: store recent diffs
    hist_diffs = []   # store (frame_idx, hist_sim) where hist_sim is similarity measure
    ssim_vals = []

    scenes = []  # list of detected events: dicts with type, frame_idx, timestamp, optional end_frame

    pbar = tqdm(total=frame_count, desc="Processing frames", unit="fr")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # resize preserving aspect
        h, w = frame.shape[:2]
        if w > resize_w:
            scale = resize_w / float(w)
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_hist = compute_hsv_hist(frame)

        if prev_hist is None:
            prev_hist = curr_hist
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)
            continue

        # histogram similarity (1 == identical, smaller -> different)
        hist_sim = hist_compare_chi2(prev_hist, curr_hist)  # 0..1
        # SSIM (1 == identical)
        ssim_val = compute_ssim_gray(prev_gray, gray)

        # store diffs (we'll use 1 - sim as "difference")
        hist_diffs.append((frame_idx, 1.0 - hist_sim))
        ssim_vals.append((frame_idx, 1.0 - ssim_val))

        # 1) Abrupt cut detection (immediate)
        is_cut_hist = hist_sim < CUT_THRESHOLD
        is_cut_ssim = ssim_val < SSIM_CUT_THRESHOLD
        if is_cut_hist or is_cut_ssim:
            # Log abrupt cut at this frame index (transition between frame_idx-1 -> frame_idx)
            scenes.append({
                "type": "cut",
                "frame": frame_idx,
                "timestamp": timestamp_from_frame(frame_idx, fps)
            })
            # clear recent diffs because abrupt cut resets fade accumulation
            hist_diffs.clear()
            ssim_vals.clear()
            # Move on
            prev_hist = curr_hist
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)
            continue

        # 2) Gradual fade/dissolve detection
        # Consider last FADE_WINDOW frames of histogram "difference" (1 - sim)
        if len(hist_diffs) >= FADE_WINDOW:
            window = hist_diffs[-FADE_WINDOW:]
            frames_in_window = [f for f, d in window]
            diffs = np.array([d for f, d in window])
            sum_diffs = diffs.sum()
            max_single = diffs.max()
            # Heuristic:
            # - sum of diffs in window is sufficiently large (i.e., overall change)
            # - but no single diff is huge (otherwise it'd be a cut, but that was tested)
            # - differences are somewhat continuous (not just one spike)
            if sum_diffs >= FADE_SUM_DIFF_THRESHOLD and max_single <= FADE_MAX_SINGLE_DIFF:
                # Also ensure at least MIN_FADE_LENGTH_FRAMES of non-trivial diffs
                nonzero_count = int((diffs > 0.02).sum())
                if nonzero_count >= MIN_FADE_LENGTH_FRAMES:
                    # We detected a fade spanning roughly the window. Find start frame:
                    # Determine first frame in window where cumulative diff from window start crosses small threshold
                    cumulative = np.cumsum(diffs)
                    start_offset = np.searchsorted(cumulative, 0.05 * cumulative[-1])
                    fade_start_frame = window[start_offset][0]
                    fade_end_frame = window[-1][0]
                    # Avoid duplicate detections: if last detected scene was very close, skip
                    if len(scenes) == 0 or (scenes[-1]["type"] != "fade" or (fade_start_frame - scenes[-1].get("frame", -999)) > int(0.5 * fps)):
                        scenes.append({
                            "type": "fade",
                            "start_frame": int(fade_start_frame),
                            "end_frame": int(fade_end_frame),
                            "start_time": timestamp_from_frame(int(fade_start_frame), fps),
                            "end_time": timestamp_from_frame(int(fade_end_frame), fps)
                        })
                        # clear diffs to avoid re-detecting same fade
                        hist_diffs.clear()
                        ssim_vals.clear()

        prev_hist = curr_hist
        prev_gray = gray
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    # Write log file
    with open(out_log_path, "w", encoding="utf-8") as f:
        f.write(f"# Scene detection log for: {video_path}\n")
        f.write(f"# FPS (detected): {fps}\n")
        f.write(f"# Total frames processed: {frame_idx}\n\n")
        for ev in scenes:
            if ev["type"] == "cut":
                f.write(f"CUT\tframe:{ev['frame']}\ttime:{ev['timestamp']}\n")
            elif ev["type"] == "fade":
                f.write(f"FADE\tstart_frame:{ev['start_frame']}\tend_frame:{ev['end_frame']}\tstart_time:{ev['start_time']}\tend_time:{ev['end_time']}\n")

    print(f"Detection finished. {len(scenes)} events logged to {out_log_path}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scene_detect.py input_video.mp4 output_log.txt")
        sys.exit(1)
    input_video = sys.argv[1]
    out_log = sys.argv[2]
    detect_scenes(input_video, out_log)
