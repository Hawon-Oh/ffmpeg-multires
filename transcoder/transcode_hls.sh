#!/bin/bash
# 사용법: bash transcoder/transcode_hls.sh path/to/input.mp4 [output_directory]

# video 파일과 출력 디렉토리 설정
# 출력 디렉토리 default는 ./results
INPUT="$1"
BASENAME=$(basename "$INPUT" .mp4)
OUTPUT_DIR="./${2:-./results}/$BASENAME"



mkdir -p "$OUTPUT_DIR"

# 원본 영상 세로 해상도 확인
ORIGINAL_HEIGHT=$(ffprobe -v error -select_streams v:0 -show_entries stream=height \
    -of csv=p=0 "$INPUT")
echo "Original height: $ORIGINAL_HEIGHT"


# 해상도/비트레이트 설정
declare -A HLS_SETTINGS
HLS_SETTINGS=( ["360"]="800k" ["480"]="1400k" ["720"]="2800k" ["1080"]="5000k" )

MASTER_PLAYLIST="$OUTPUT_DIR/master.m3u8"
> "$MASTER_PLAYLIST"

for RES in "${!HLS_SETTINGS[@]}"; do
    if [ $RES -le $ORIGINAL_HEIGHT ]; then
        BITRATE="${HLS_SETTINGS[$RES]}"
        SEGMENT_PREFIX="$OUTPUT_DIR/${RES}p_"
        PLAYLIST_FILE="$OUTPUT_DIR/${RES}p.m3u8"

        echo "Creating HLS for ${RES}p..."
        ffmpeg -i "$INPUT" \
            -vf scale=-2:$RES \
            -c:v h264 -b:v $BITRATE \
            -c:a aac -ar 48000 -ac 2 -b:a 128k \
            -f hls -hls_time 6 -hls_playlist_type vod \
            -hls_segment_filename "${SEGMENT_PREFIX}%03d.ts" \
            "$PLAYLIST_FILE"

        echo "#EXT-X-STREAM-INF:BANDWIDTH=$BITRATE,RESOLUTION=${RES}p" >> "$MASTER_PLAYLIST"
        echo "$(basename $PLAYLIST_FILE)" >> "$MASTER_PLAYLIST"
    else
        echo "Skipping ${RES}p (larger than original)"
    fi
done



echo "HLS conversion done! Master playlist: $MASTER_PLAYLIST"