#!/bin/bash

# =======================================
# video frame extraction for AI analysis
# =======================================

# 입력 영상과 출력 디렉토리 설정
INPUT="$1"
if [ -z "$INPUT" ]; then
    echo "Usage: $0 <video_file> [output_dir]"
    exit 1
fi

BASENAME=$(basename "$INPUT" .mp4)
OUTPUT_DIR="${2:-./results}/$BASENAME/screenshots"

mkdir -p "$OUTPUT_DIR"

# -----------------------------
# 1. 기본 fps 설정 (1초에 30프레임)
FPS=10

# -----------------------------
# 2. 장면 전환 감지 임계값
# scene=0.2 정도가 일반적으로 적당, 필요시 조정
SCENE_THRESHOLD=0.05

# -----------------------------
# 3. ffmpeg 실행
ffmpeg -i "$INPUT" \
-vf "fps=$FPS,select='gt(scene\,$SCENE_THRESHOLD)'" \
-vsync vfr \
"$OUTPUT_DIR/%08d.jpg"

# -----------------------------
echo "Frames extracted to: $OUTPUT_DIR"
