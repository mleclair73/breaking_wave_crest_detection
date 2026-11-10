#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python_bin="${PYTHON_BIN:-python}"
device="${DEVICE:-cuda}"
batch_size="${INFERENCE_BATCH_SIZE:-32}"
metric_workers="${METRIC_WORKERS:-8}"

"${python_bin}" -m ablation.verify
"${python_bin}" -m train --all --device "${device}"
"${python_bin}" -m ablation.infer --device "${device}" --batch-size "${batch_size}"
"${python_bin}" -m ablation.evaluate --workers "${metric_workers}"
"${python_bin}" -m ablation.visualize_metrics --examples 4
"${python_bin}" -m ablation.costs
"${python_bin}" -m ablation.operating_points --workers "${metric_workers}"
"${python_bin}" -m ablation.report
