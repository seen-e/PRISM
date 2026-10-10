#!/usr/bin/env bash
set -euo pipefail

PRISM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STARVLA_DIR=${PRISM_ROOT}/starvla

python -m pip install -r "${STARVLA_DIR}/requirements.txt"
python -m pip install -e "${PRISM_ROOT}[data]" -e "${STARVLA_DIR}"
echo "Installed the bundled StarVLA + PRISM code."
