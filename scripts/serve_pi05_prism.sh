#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to an OpenPI checkpoint step directory}"

PRISM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OPENPI_DIR=${OPENPI_DIR:-${PRISM_ROOT}/openpi}
cd "${OPENPI_DIR}"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_prism \
  --policy.dir="${CHECKPOINT_DIR}" \
  "$@"
