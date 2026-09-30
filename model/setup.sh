#!/usr/bin/env bash
# Build the exact RunPod training environment without replacing the image's
# CUDA-enabled PyTorch/torchvision installation.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$MODEL_DIR/.." && pwd)"
SYSTEM_PYTHON="${SYSTEM_PYTHON:-/usr/local/bin/python}"
VENV_DIR="${ABLATION_VENV:-$MODEL_DIR/.venv-ablation}"
EXPECTED_TORCH_VERSION="${EXPECTED_TORCH_VERSION:-2.8.0}"
EXPECTED_TORCHVISION_VERSION="${EXPECTED_TORCHVISION_VERSION:-0.23.0}"
EXPECTED_CUDA_VERSION="${EXPECTED_CUDA_VERSION:-12.8}"
UV_VERSION="${UV_VERSION:-0.5.18}"

if ! command -v rsync >/dev/null 2>&1; then
    apt-get update
    apt-get install --yes rsync
fi

verify_cuda_stack() {
    "$1" - "$EXPECTED_TORCH_VERSION" "$EXPECTED_TORCHVISION_VERSION" \
        "$EXPECTED_CUDA_VERSION" <<'PY'
import sys

import torch
import torchvision

expected_torch, expected_torchvision, expected_cuda = sys.argv[1:]
actual_torch = torch.__version__.split("+", 1)[0]
actual_torchvision = torchvision.__version__.split("+", 1)[0]
assert actual_torch == expected_torch, (actual_torch, expected_torch)
assert actual_torchvision == expected_torchvision, (
    actual_torchvision,
    expected_torchvision,
)
assert torch.version.cuda == expected_cuda, (torch.version.cuda, expected_cuda)
assert torch.cuda.is_available(), "system PyTorch cannot see CUDA"
print(f"Using torch {torch.__version__} / torchvision {torchvision.__version__}")
print(f"CUDA {torch.version.cuda}: {torch.cuda.get_device_name(0)}")
PY
}

if ! verify_cuda_stack "$SYSTEM_PYTHON"; then
    echo "Expected the RunPod CUDA stack: torch $EXPECTED_TORCH_VERSION, torchvision $EXPECTED_TORCHVISION_VERSION, CUDA $EXPECTED_CUDA_VERSION." >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$installer"
    UV_INSTALL_DIR=/usr/local/bin sh "$installer"
fi

# --system-site-packages exposes the pod image's torch 2.8/CUDA 12.8 build.
# Exporting from uv.lock pins every other dependency. Pruning torch and
# torchvision is intentional: resolving either would download a second,
# multi-GB CUDA stack and could break the image's driver-compatible build.
# Pin the venv to the same interpreter whose CUDA stack was verified above.
# Without --python, uv honors the repository's .python-version and may download
# a different patch release; that interpreter cannot see the system torch even
# with --system-site-packages.
uv venv --python "$SYSTEM_PYTHON" --no-python-downloads \
    --system-site-packages "$VENV_DIR"
locked_requirements="$(mktemp)"
trap 'rm -f "${installer:-}" "$locked_requirements"' EXIT
uv export --directory "$REPO_DIR" --locked --no-dev --no-emit-project \
    --no-hashes --prune torch --prune torchvision \
    --output-file "$locked_requirements"
uv pip install --python "$VENV_DIR/bin/python" --no-deps \
    --requirements "$locked_requirements"
uv pip install --python "$VENV_DIR/bin/python" --no-deps --editable "$REPO_DIR"

verify_cuda_stack "$VENV_DIR/bin/python"
"$VENV_DIR/bin/python" - <<'PY'
import ablation
import fvcore
import segmentation_models_pytorch
import timm
import torch

probe = torch.ones(1, device="cuda")
print(f"Environment ready; CUDA probe={probe.item():.0f}")
PY
