#!/usr/bin/env bash
# Train the canonical ablation from any working directory.
#
#   bash model/train.sh            # full CUDA training
#   bash model/train.sh --smoke    # short validation run
#
# Override the defaults with PYTHON_BIN and DEVICE. The package must first be
# installed with `uv sync --locked` or `bash model/setup.sh` on a CUDA pod.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
DEFAULT_PYTHON="$MODEL_DIR/.venv-ablation/bin/python"
if [[ ! -x "$DEFAULT_PYTHON" ]]; then
    DEFAULT_PYTHON="python"
fi
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
DEVICE="${DEVICE:-cuda}"

exec "$PYTHON_BIN" -m train --all --device "$DEVICE" "$@"
