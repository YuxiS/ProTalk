#!/usr/bin/env bash
set -euo pipefail
if (( $# != 3 )); then
  echo 'Usage: bash scripts/postprocess.sh SILENT_VIDEO AUDIO OUTPUT_DIRECTORY' >&2
  exit 2
fi
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
video="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
audio="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
mkdir -p "$3"
output="$(cd "$3" && pwd)"
wav2lip_root="${WAV2LIP_ROOT:-$repo_root/Wav2Lip}"
if [[ ! -f "$wav2lip_root/inference.py" || ! -f "${WAV2LIP_CHECKPOINT:-}" ]]; then
  echo 'Set WAV2LIP_ROOT and WAV2LIP_CHECKPOINT to an installed Wav2Lip checkout and checkpoint.' >&2
  exit 1
fi
# Use Wav2Lip's environment; its script resolves helper paths from its own directory.
cd "$wav2lip_root"
"${WAV2LIP_PYTHON:-python}" inference.py --checkpoint_path "$WAV2LIP_CHECKPOINT" \
  --face "$video" --audio "$audio" --outfile "$output/lip_synced.mp4"
# The repository already includes a GFPGAN video entry point; use its separate environment.
cd "$repo_root"
mkdir -p "$output/gfpgan_input" "$output/restored"
cp "$output/lip_synced.mp4" "$output/gfpgan_input/lip_synced.mp4"
"${GFPGAN_PYTHON:-python}" inference_gfpgan.py -i "$output/gfpgan_input" -o "$output/restored"
ffmpeg -y -i "$output/restored/lip_synced.mp4" -i "$audio" \
  -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest "$output/result.mp4"
