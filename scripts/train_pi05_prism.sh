#!/usr/bin/env bash
set -euo pipefail

: "${PRISM_REPO_ID:?Set PRISM_REPO_ID to the LeRobot repository id}"
: "${PRISM_PPS_PATH:?Set PRISM_PPS_PATH to pps_weights.npy}"

PRISM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OPENPI_DIR=${OPENPI_DIR:-${PRISM_ROOT}/openpi}
EXP_NAME=${EXP_NAME:-prism_run}
PRISM_LOSS_ACTION_DIM=${PRISM_LOSS_ACTION_DIM:-14}
PRISM_COMPUTE_NORM_STATS=${PRISM_COMPUTE_NORM_STATS:-1}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}
OVERWRITE=${OVERWRITE:-0}

export PRISM_REPO_ID PRISM_PPS_PATH PRISM_LOSS_ACTION_DIM XLA_PYTHON_CLIENT_MEM_FRACTION
cd "${OPENPI_DIR}"

if [[ "${PRISM_COMPUTE_NORM_STATS}" == "1" ]]; then
  uv run scripts/compute_norm_stats.py --config-name pi05_prism
fi

TRAIN_CMD=(uv run scripts/train.py pi05_prism --exp-name "${EXP_NAME}")
if [[ "${OVERWRITE}" == "1" ]]; then
  TRAIN_CMD+=(--overwrite)
fi
"${TRAIN_CMD[@]}" "$@"
