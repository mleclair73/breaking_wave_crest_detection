#!/usr/bin/env python
"""OOD variant of figure 13 — per-image attention alignment (axis view) and
native-boundary PCA for four out-of-distribution ``ood_test`` frames.

This is a thin driver: it imports the figure-13 module unchanged and reuses its
exact machinery (same model, hooks, PCA and alignment computations). Only the
inputs (four ``ood_test`` images) and the output names (prefixed per image)
differ. The two requested figures per image are:

  <stem>_pca_native_boundary.{pdf,png}
  <stem>_attention_alignment_axis.{pdf,png}

The correct-class alignment view is intentionally skipped. Outputs land in this
folder (figures/ml-exploration/, pdfs under pdf/).

Run:
    python ml-exploration/13_attention_viz_ood.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import torch

from exploration_common import (
    DEFAULT_OOD_IMAGES,
    add_runtime_arguments,
    import_figure,
    load_runtime,
    require_file,
)

# Module name starts with a digit, so import it through the shared helper.
viz = import_figure("13_attention_viz")

OOD_STEMS = ("ood_test_004", "ood_test_014", "ood_test_019", "ood_test_017")


def _rename(output_dir: Path, old_stem: str, new_stem: str) -> None:
    """figure_pca_native_boundary hardcodes its output name; rename the png and
    its pdf sidecar to the per-image stem."""
    for folder, ext in ((output_dir, "png"), (output_dir / "pdf", "pdf")):
        src = folder / f"{old_stem}.{ext}"
        if src.exists():
            src.replace(folder / f"{new_stem}.{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser)
    parser.add_argument(
        "--ood-dir", type=Path, default=DEFAULT_OOD_IMAGES,
        help="directory containing OOD PNG inputs",
    )
    parser.add_argument(
        "--stems", nargs="+", default=list(OOD_STEMS),
        help="image stems to render (without .png)",
    )
    args = parser.parse_args()

    viz.apply_style()
    runtime = load_runtime(args)
    viz.OUT_DIR = runtime.output_dir
    ood_dir = args.ood_dir.expanduser().resolve()

    for stem in args.stems:
        path = require_file(ood_dir / f"{stem}.png", "OOD image")
        print(f"\n=== {stem} ===")
        print(f"Loading image: {path}")
        img_tensor, img_raw, img_hw = viz.preprocess_image(path)
        img_display = viz.denormalize(img_raw)

        # Single forward pass: hooks record MSCA activations while the same
        # logits give the prediction mask (see figure 13 for the rationale).
        hook = viz.AttentionHook()
        hook.register_hooks(runtime.model)
        with torch.no_grad():
            logits = runtime.model(img_tensor.to(runtime.device))
        fg_mask = viz.mask_from_logits(logits, img_hw)
        print(f"  Breaking: {fg_mask.sum()} / {fg_mask.size} px "
              f"({100 * fg_mask.mean():.1f}%)")
        acts = hook.activations

        # (1) native-boundary PCA (fixed name → rename per image)
        print("  Figure: native-boundary PCA...")
        viz.figure_pca_native_boundary(
            runtime.model, acts, img_display, img_hw, fg_mask
        )
        _rename(
            runtime.output_dir,
            "13_pca_native_boundary",
            f"{stem}_pca_native_boundary",
        )

        # (2) attention alignment — axis view only (skip correct-class)
        print("  Figure: attention alignment (axis)...")
        stage_contributions = viz._gather_attention_contributions(acts)
        per_stage = viz._attention_class_alignment_axis(stage_contributions,
                                                        fg_mask)
        axis_maps = {si: amap for si, (amap, _fg) in per_stage.items()}
        viz._plot_alignment_row(
            axis_maps,
            viz._combine_maps(list(axis_maps.values()), img_hw),
            img_display, img_hw, fg_mask,
            f"{stem}_attention_alignment_axis.png",
            pos_label="breaking", neg_label="non-breaking",
        )
        hook.remove_hooks()

    print("\nDone.")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
