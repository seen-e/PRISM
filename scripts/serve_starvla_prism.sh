#!/usr/bin/env bash
set -euo pipefail

: "${CKPT_PATH:?Set CKPT_PATH to the trained StarVLA checkpoint}"

PRISM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${PRISM_ROOT}/starvla}
PORT=${PORT:-10093}
GPU_ID=${GPU_ID:-0}
cd "${STARVLA_DIR}"
export PYTHONPATH="${PRISM_ROOT}:${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python deployment/model_server/server_policy.py \
  --ckpt_path "${CKPT_PATH}" \
  --port "${PORT}" \
  --use_bf16 \
  "$@"
