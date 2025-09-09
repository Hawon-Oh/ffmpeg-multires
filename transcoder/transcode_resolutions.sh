#!/bin/bash
# 사용법: bash transcoder/transcode_resolutions.sh path/to/input.mp4 [output_directory]


# video 파일과 출력 디렉토리 설정
# 출력 디렉토리 default는 ./results
INPUT="$1"
BASENAME=$(basename "$INPUT" .mp4)
OUTPUT_DIR="./${2:-./results}/$BASENAME"

mkdir -p "$OUTPUT_DIR"


# 원본 영상 세로 해상도 확인
ORIGINAL_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
    -of csv=p=0 "$INPUT")

# ORIGINAL_HEIGHT=480 # 테스트용 하드코딩

echo "Original height: $ORIGINAL_HEIGHT"



# 변환할 해상도 목록
RESOLUTIONS=(1080 720 480 360)

for RES in "${RESOLUTIONS[@]}"; do
    # 원본보다 큰 해상도는 스킵
    if [ $RES -le $ORIGINAL_HEIGHT ]; then
        OUTPUT="$OUTPUT_DIR/${RES}p.mp4"
        echo "Converting $INPUT to ${RES}p..."
        ffmpeg -i "$INPUT" \
            -vf scale=-2:$RES \
            -c:v libx264 -crf 23 -preset fast \
            "$OUTPUT"
    else
        echo "Skipping ${RES}p (larger than original)"
    fi
done



echo "All MP4 conversions done!"