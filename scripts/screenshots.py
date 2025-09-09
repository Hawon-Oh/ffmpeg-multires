import os
import subprocess

def extract_screenshots(video_path, log_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("CUT"):
            parts = line.strip().split("\t")
            frame_info = parts[1]  # e.g., frame:341
            time_info = parts[2]   # e.g., time:00:00:14

            frame_num = frame_info.split(":")[1]
            timestamp = time_info.split(":")[1:]  # ["00","00","14"]
            timestamp_str = ":".join(timestamp)   # "00:00:14"

            output_file = os.path.join(output_dir, f"frame_{frame_num}.jpg")

            # FFmpeg 명령어: 특정 시간에서 한 장 추출
            cmd = [
                "ffmpeg",
                "-ss", timestamp_str,
                "-i", video_path,
                "-frames:v", "1",
                "-q:v", "2",  # 품질 (1 최고 ~ 31 최저)
                "-y",         # 덮어쓰기
                output_file
            ]

            print("Extracting:", timestamp_str, "->", output_file)
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("완료! 스크린샷이", output_dir, "에 저장되었습니다.")

if __name__ == "__main__":
    # 예시 실행
    video_path = r".\results\Golden - KPop Demon Hunters\360p.mp4"
    log_path = r".\log2.txt"
    output_dir = r".\screenshots2"

    extract_screenshots(video_path, log_path, output_dir)
