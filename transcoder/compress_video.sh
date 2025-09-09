#!/bin/bash
# compress.sh
# Usage: bash compress.sh input.mp4 output.mp4

INPUT="$1"
OUTPUT="$2"

# ffmpeg로 H.264 압축 (화질-용량 균형 CRF=23)
# ffmpeg -y -i "$INPUT" -c:v libx264 -crf 23 -preset fast -c:a copy "$OUTPUT"
ffmpeg -y -i "$INPUT" -c:v libx264 -pix_fmt yuv420p -crf 23 -preset fast -c:a copy "$OUTPUT"