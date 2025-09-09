#!/usr/bin/env python3
"""
scene_detect_ms.py

Usage:
    python scene_detect_ms.py input_video.mp4 output_log.txt

Description:
    - Detects abrupt cuts and gradual fades/dissolves.
    - Logs timestamps (millisecond precision) to output_log.txt.
    - Cuts must be at least 1 second apart to be recorded.
"""

import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# ----------------------------
# PARAMETERS
# ----------------------------
RESIZE_WIDTH = 320          # 속도 위해 프레임 리사이즈
HIST_BINS = [32, 32, 32]    # HSV 히스토그램 bin 수
CUT_THRESHOLD = 0.6         # 히스토그램 유사도 컷 임계값
SSIM_CUT_THRESHOLD = 0.35   # SSIM 컷 임계값
FADE_WINDOW = 30            # 페이드 탐지 윈도우 크기 (프레임 수)
FADE_SUM_DIFF_THRESHOLD = 8.0
FADE_MAX_SINGLE_DIFF = 0.9
MIN_FADE_LENGTH_FRAMES = 8
MIN_SCENE_LENGTH_SEC = 1.0  # 컷 사이 최소 1초
# ----------------------------


def timestamp_from_frame(frame_idx, fps, milliseconds=True):
    seconds = frame_idx / fps
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    if milliseconds:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}"


def compute_hsv_hist(frame, bins=HIST_BINS):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0], None, [bins[0]], [0, 180])
    s = cv2.calcHist([hsv], [1], None, [bins[1]], [0, 256])
    v = cv2.calcHist([hsv], [2], None, [bins[2]], [0, 256])
    hist = np.concatenate([h.flatten(), s.flatten(), v.flatten()])
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


def hist_compare_chi2(h1, h2):
    eps = 1e-10
    chi = 0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps))
    sim = 1.0 / (1.0 + chi)
    return sim


def compute_ssim_gray(a, b):
    return ssim(a, b)


def detect_scenes(video_path, out_log_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    resize_w = RESIZE_WIDTH

    prev_hist = None
    prev_gray = None
    hist_diffs = []
    ssim_vals = []
    scenes = []

    last_cut_time = -9999.0  # 마지막 기록된 컷 시간(초)

    pbar = tqdm(total=frame_count, desc="Processing frames", unit="fr")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > resize_w:
            scale = resize_w / float(w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_hist = compute_hsv_hist(frame)

        if prev_hist is None:
            prev_hist = curr_hist
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)
            continue

        hist_sim = hist_compare_chi2(prev_hist, curr_hist)
        ssim_val = compute_ssim_gray(prev_gray, gray)

        hist_diffs.append((frame_idx, 1.0 - hist_sim))
        ssim_vals.append((frame_idx, 1.0 - ssim_val))

        # -------- Abrupt cut detection --------
        is_cut_hist = hist_sim < CUT_THRESHOLD
        is_cut_ssim = ssim_val < SSIM_CUT_THRESHOLD
        if is_cut_hist or is_cut_ssim:
            current_time_sec = frame_idx / fps
            if current_time_sec - last_cut_time >= MIN_SCENE_LENGTH_SEC:
                scenes.append({
                    "type": "cut",
                    "frame": frame_idx,
                    "timestamp": timestamp_from_frame(frame_idx, fps, milliseconds=True)
                })
                last_cut_time = current_time_sec

            hist_diffs.clear()
            ssim_vals.clear()
            prev_hist = curr_hist
            prev_gray = gray
            frame_idx += 1
            pbar.update(1)
            continue

        # -------- Fade detection --------
        if len(hist_diffs) >= FADE_WINDOW:
            window = hist_diffs[-FADE_WINDOW:]
            diffs = np.array([d for _, d in window])
            sum_diffs = diffs.sum()
            max_single = diffs.max()
            if sum_diffs >= FADE_SUM_DIFF_THRESHOLD and max_single <= FADE_MAX_SINGLE_DIFF:
                nonzero_count = int((diffs > 0.02).sum())
                if nonzero_count >= MIN_FADE_LENGTH_FRAMES:
                    cumulative = np.cumsum(diffs)
                    start_offset = np.searchsorted(cumulative, 0.05 * cumulative[-1])
                    fade_start_frame = window[start_offset][0]
                    fade_end_frame = window[-1][0]
                    if len(scenes) == 0 or (scenes[-1]["type"] != "fade" or
                                            (fade_start_frame - scenes[-1].get("frame", -999)) > int(0.5 * fps)):
                        scenes.append({
                            "type": "fade",
                            "start_frame": int(fade_start_frame),
                            "end_frame": int(fade_end_frame),
                            "start_time": timestamp_from_frame(int(fade_start_frame), fps, milliseconds=True),
                            "end_time": timestamp_from_frame(int(fade_end_frame), fps, milliseconds=True)
                        })
                        hist_diffs.clear()
                        ssim_vals.clear()

        prev_hist = curr_hist
        prev_gray = gray
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    with open(out_log_path, "w", encoding="utf-8") as f:
        f.write(f"# Scene detection log for: {video_path}\n")
        f.write(f"# FPS (detected): {fps}\n")
        f.write(f"# Total frames processed: {frame_idx}\n\n")
        for ev in scenes:
            if ev["type"] == "cut":
                f.write(f"CUT\tframe:{ev['frame']}\ttime:{ev['timestamp']}\n")
            elif ev["type"] == "fade":
                f.write(f"FADE\tstart_frame:{ev['start_frame']}\tend_frame:{ev['end_frame']}\t"
                        f"start_time:{ev['start_time']}\tend_time:{ev['end_time']}\n")

    print(f"Detection finished. {len(scenes)} events logged to {out_log_path}.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scene_detect_ms.py input_video.mp4 output_log.txt")
        sys.exit(1)
    input_video = sys.argv[1]
    out_log = sys.argv[2]
    detect_scenes(input_video, out_log)
