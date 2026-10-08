#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/train.yaml}"
for stage in expression vqvae sampler; do
  "${PYTHON:-python}" -m protalk train --config "$config" --stage "$stage"
done
