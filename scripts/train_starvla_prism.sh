#!/usr/bin/env bash
set -euo pipefail

: "${BASE_VLM:?Set BASE_VLM to the Qwen3-VL-4B model directory}"
: "${DATA_ROOT:?Set DATA_ROOT to the StarVLA LeRobot data root}"
: "${DATA_MIX:?Set DATA_MIX to a registered StarVLA dataset mixture}"
: "${PPS_ROOT:?Set PPS_ROOT to the directory containing <dataset_name>.npz}"
: "${PRETRAINED_CHECKPOINT:?Set PRETRAINED_CHECKPOINT to the StarVLA QwenPI_v3 initialization checkpoint}"

PRISM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${PRISM_ROOT}/starvla}
RUN_ROOT=${RUN_ROOT:-${STARVLA_DIR}/results/PRISM}
RUN_ID=${RUN_ID:-prism_qwenpi_v3}
NUM_PROCESSES=${NUM_PROCESSES:-1}
ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-${STARVLA_DIR}/starVLA/config/deepseeds/deepspeed_zero2.yaml}

cd "${STARVLA_DIR}"
export PYTHONPATH="${PRISM_ROOT}:${STARVLA_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
accelerate launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${PRISM_ROOT}/configs/starvla_prism_qwenpi_v3.yaml" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.pps_root "${PPS_ROOT}" \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --run_root_dir "${RUN_ROOT}" \
  --run_id "${RUN_ID}" \
  "$@"
