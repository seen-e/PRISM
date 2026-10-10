#!/usr/bin/env bash
set -euo pipefail

PRISM_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
OPENPI_DIR=${PRISM_ROOT}/openpi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun." >&2
  exit 1
fi

cd "${OPENPI_DIR}"
uv sync --frozen
uv pip install -e "${PRISM_ROOT}[data]"
echo "Installed the bundled pi0.5 + PRISM code in ${OPENPI_DIR}/.venv."
