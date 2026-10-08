#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
  echo 'Usage: bash generate.sh REFERENCE_IMAGE DRIVING_AUDIO [reference.py options...]' >&2
  exit 2
fi
reference_image="$1"
driving_audio="$2"
shift 2
# Resolve user inputs before changing the working directory.
reference_image="$(cd "$(dirname "$reference_image")" && pwd)/$(basename "$reference_image")"
driving_audio="$(cd "$(dirname "$driving_audio")" && pwd)/$(basename "$driving_audio")"
cd "$(dirname "$0")"
exec "${PYTHON:-python}" reference.py --ref_img "$reference_image" --driven_audio "$driving_audio" "$@"
