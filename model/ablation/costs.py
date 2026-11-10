#!/usr/bin/env python3
"""Measure parameters and conventional 2x-MAC FLOPs for table models."""

from __future__ import annotations

import argparse
import csv

import torch

from .config import load_all_configs, resolve_path
from models import build_model


def run_costs() -> list[dict]:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError as error:
        raise ImportError("Cost measurement requires fvcore") from error
    manifest, entries = load_all_configs()
    rows = []
    for entry, config in entries:
        if not entry.get("table"):
            continue
        build_config = dict(config)
        build_config["encoder_pretrained"] = False
        torch.manual_seed(int(config["seed"]))
        model = build_model(build_config).eval()
        shape = (
            1,
            int(config["in_channels"]),
            int(config["image_size"]),
            int(config["image_size"]),
        )
        sample = torch.zeros(shape)
        analysis = FlopCountAnalysis(model, sample)
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        macs = int(analysis.total())
        rows.append({
            "model_id": config["id"],
            "model": config["label"],
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "macs": macs,
            "flops": 2 * macs,
            "input": "1x3x224x224",
        })
        print(
            f"{config['label']}: params={rows[-1]['parameters']:,} "
            f"MACs={macs / 1e9:.3f}G FLOPs={2 * macs / 1e9:.3f}G",
            flush=True,
        )
    output = resolve_path(manifest["output_root"]) / "metrics"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_costs()


if __name__ == "__main__":
    main()
