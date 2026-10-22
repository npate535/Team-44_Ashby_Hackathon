#!/bin/zsh
# Render, then master the audio to -14 LUFS / -1 dBTP (two-pass loudnorm) without re-encoding the video.
set -e
cd "$(dirname "$0")/.."
npx remotion render src/index.ts Launch out/sepia-launch-raw.mp4 --codec=h264 --crf=16 --audio-codec=aac --audio-bitrate=320k --concurrency=10 --log=error
OUT=../../outputs/launch-video/Sepia_Launch.mp4
mkdir -p "$(dirname $OUT)"
M=$(ffmpeg -hide_banner -i out/sepia-launch-raw.mp4 -af loudnorm=I=-14:TP=-1.0:LRA=11:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p' |
  python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"measured_I={d['input_i']}:measured_TP={d['input_tp']}:measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:offset={d['target_offset']}\")")
ffmpeg -y -v error -i out/sepia-launch-raw.mp4 -c:v copy -af "loudnorm=I=-14:TP=-1.0:LRA=11:${M}:linear=true,aresample=48000" -c:a aac -b:a 320k -movflags +faststart $OUT
ffmpeg -hide_banner -i $OUT -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+(I|Peak):" | tail -2
echo "wrote $OUT"
