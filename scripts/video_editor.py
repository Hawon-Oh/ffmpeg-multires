import cv2
import os


def timestamp_to_seconds(ts):
    """HH:MM:SS -> seconds"""
    h, m, s = map(int, ts.strip().split(":"))
    return h*3600 + m*60 + s

def label_scenes_on_video(video_path, log_path, output_path):
    # log.txt 읽기
    scene_times = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("CUT"):
                parts = line.strip().split("\t")
                
                time_str = parts[2].split("time:")[1]  # "00:00:03" 그대로 가져오기
                timestamp = ":".join(time_str)
                scene_times.append(timestamp_to_seconds(time_str))

    if not scene_times:
        print("장면 정보 없음.")
        return

    # 마지막 장면 끝 = 영상 끝
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    scene_times.append(int(duration))  # 마지막 장면 끝

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 현재 장면 추적
    current_scene_idx = 0
    next_scene_time = scene_times[current_scene_idx + 1]

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        current_time_sec = frame_idx / fps

        # 다음 장면 시간 도달하면 장면 번호 증가
        if current_time_sec >= next_scene_time:
            current_scene_idx += 1
            next_scene_time = scene_times[current_scene_idx + 1] if current_scene_idx + 1 < len(scene_times) else duration

        scene_label = f"Scene {current_scene_idx + 1}"

        # 왼쪽 하단 텍스트
        cv2.putText(frame, scene_label, (20, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)

        if frame_idx % 500 == 0:
            print(f"Processing frame {frame_idx}/{total_frames} - {scene_label}")

    cap.release()
    out.release()
    print(f"완료! 장면 레이블 적용 영상: {output_path}")

if __name__ == "__main__":
    video_path = r".\results\Golden - KPop Demon Hunters\720p.mp4"
    log_path = r".\log2.txt"
    output_path = r".\results\Golden - KPop Demon Hunters\720p_labeled.mp4"

    label_scenes_on_video(video_path, log_path, output_path)
