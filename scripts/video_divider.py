import cv2
import os
from datetime import datetime

def timestamp_to_seconds(ts):
    # HH:MM:SS -> 초 단위
    h, m, s = map(int, ts.strip().split(":"))
    return h*3600 + m*60 + s

def split_scenes(video_path, log_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # log.txt에서 장면 타임스탬프 읽기
    scene_times = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CUT"):
                parts = line.strip().split("\t")
                time_str = parts[2].split(":")[1:]  # ["00","00","14"]
                timestamp = ":".join(time_str)
                scene_times.append(timestamp_to_seconds(timestamp))

    if not scene_times:
        print("장면 정보 없음.")
        return

    # 마지막 장면 끝 = 영상 끝
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    scene_times.append(int(duration))  # 마지막 장면 끝

    print(f"총 {len(scene_times)-1} 장면 생성 예정, 영상 길이: {duration:.2f}s")

    # 장면별로 영상 저장
    for i in range(len(scene_times)-1):
        start_sec = scene_times[i]
        end_sec = scene_times[i+1]
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        scene_number = i + 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        out_file = os.path.join(output_dir, f"scene_{scene_number:02d}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_file, fourcc, fps, (width, height))

        for f in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            # 왼쪽 하단에 장면번호 넣기
            cv2.putText(frame, f"Scene {scene_number}", (20, height-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            out.write(frame)

        out.release()
        print(f"[완료] Scene {scene_number}: {start_sec}s ~ {end_sec}s -> {out_file}")

    cap.release()
    print("모든 장면 저장 완료!")

if __name__ == "__main__":
    video_path = r".\results\Golden - KPop Demon Hunters\1080p.mp4"
    log_path = r".\log2.txt"
    output_dir = r".\scene_clips"
    split_scenes(video_path, log_path, output_dir)
