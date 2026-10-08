#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" -m torch.distributed.run --nproc_per_node="${NPROC_PER_NODE:-1}" --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-12365}" --module protalk.training.run --stage vqvae --config "${TRAIN_CONFIG:-configs/train.yaml}" "$@"
