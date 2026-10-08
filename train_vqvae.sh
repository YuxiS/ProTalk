#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE:-1}" \
  --master_port="${MASTER_PORT:-12365}" \
  --module "train_script.train_vqvae" --hparams "${HPARAMS:-hparams.yaml}" --distributed_run true  "$@"
