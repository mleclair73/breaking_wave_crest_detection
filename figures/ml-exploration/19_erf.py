"""Receptive field of a single breaking crest, with/without MSCA attention.

What this shows
---------------
For one predicted breaking crest, the map |∂(Σ_crest logit_brk)/∂ input|
answers a concrete question: *which input pixels does the model actually
consult when it calls this crest breaking?* Repeating it with each encoder
stage's attention zeroed shows how much of that footprint each attention
branch supplies.

How to read it (timestack axes: cross-shore × time):
- Mass tight on the crest (inside the contour) = purely local evidence.
- Mass along the crest's diagonal trace *before* the target = the model is
  using the wave's propagation history (temporal context).
- Mass spread across the surfzone away from the trace = scene-level
  context (lighting, surfzone extent) feeding the decision.
- The panels are **cumulative sweeps**: mechanisms are removed one after
  another, so each panel differs from its left neighbour by exactly one
  more mechanism off. Forward row: fine → coarse → decoder
  (S0, +S1, +S2, +S3, +Ham). Backward row: decoder → coarse → fine
  (Ham, +S3, +S2, +S1, +S0). Both rows start from the full model and end
  at the same everything-off condition, so the two paths bracket where
  the context lives: if the footprint collapses early in the backward
  row, the global mechanisms (Hamburger, deep attention) carry it; if it
  survives until the fine stages go, the context is assembled early. The
  Hamburger couples every spatial position (it factorises the whole
  feature map), so the "+Ham" step is where image-wide coupling is
  removed.

Method
------
This is the segmentation-region form of the Effective Receptive Field
(Luo et al. 2016, "Understanding the Effective Receptive Field in Deep
Convolutional Neural Networks", NeurIPS, arXiv:1701.04128): instead of a
single output unit, the score is the summed breaking logit over the crest
region (the same region-sum target as SEG-GRAD-CAM, Vinogradova et al.
2020, arXiv:2002.11434 — but differentiated back to the *input*, not to a
feature layer). Anchoring on a full crest keeps the map in image
coordinates and interpretable; single-pixel seeds needed cross-image
alignment and were dominated by upsampling/strip-kernel artefacts.

- Crest = connected component of the prediction nearest the image centre
  (shared helper with 20_seg_grad_cam.py).
- ERF map = |∂ score/∂x| summed over RGB channels — natively input-sized,
  so every panel renders at the same matplotlib size as the input panel.
- Attention zeroing: forward hooks replace each MSCAN block's
  SpatialAttention output with zeros (from 15_attention_comparison.py);
  the MLP/residual path is untouched.
- Hamburger bypass: a forward hook replaces the whole module's output
  with ReLU(input) (from the source 16_hamburger_viz.py bypass ablation)
  — the coarse decoder features pass through, but the NMF's global
  reconstruction (and with it all image-wide coupling) is removed.
- Compactness metric: high-contribution area ratio (RepLKNet, Ding et al.
  2022, arXiv:2203.06717) — the smallest pixel fraction (descending ERF
  order) holding AREA_T of the total gradient mass.
- Display is log10 of the max-normalised map (ERF intensity spans orders
  of magnitude; Luo et al. 2016), crest contoured in cyan.

Outputs
-------
  19_erf.{pdf,png}
      2 × 6 grid of cumulative sweeps. Row 1 (fine → coarse): Full |
      S0 off | S0-1 off | S0-2 off | S0-3 off | S0-3+Ham off. Row 2
      (decoder → fine): Full | Ham off | Ham+S3 | Ham+S3-2 | Ham+S3-1 |
      Ham+S3-0. The target crest is contoured cyan in every panel and the
      area@50% metric is printed in each panel corner.

Run with the `dunex_pytorch` pyenv env (gradients required):
    python 19_erf.py
"""

import argparse

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from exploration_common import (
    ML_DIR,
    add_runtime_arguments,
    import_figure,
    load_runtime,
    require_file,
)
from common.figure_style import FONTSIZE_LABEL, FONTSIZE_LEGEND, PAGE_W, apply_style, savefig

viz13 = import_figure("13_attention_viz")
gradcam = import_figure("20_seg_grad_cam")   # central_component

OUT_DIR = ML_DIR

# ── Config ───────────────────────────────────────────────────────────────────
AREA_T = 0.5            # high-contribution area threshold (RepLKNet t=50%)
LOG_FLOOR = 1e-4        # display floor before log-scaling

# Cumulative sweeps: (label, encoder stages zeroed, ham mode). Each entry
# adds one more mechanism to the previous one, so panel-to-panel changes
# are attributable to the newly removed mechanism.
# ham mode: None = intact, "bypass" = module output replaced by
# ReLU(input) — coarse features kept, global NMF coupling removed.
FORWARD_SWEEP = [                       # fine → coarse → decoder
    ("Full",         (),           None),
    ("S0 off",       (0,),         None),
    ("S0-1 off",     (0, 1),       None),
    ("S0-2 off",     (0, 1, 2),    None),
    ("S0-3 off",     (0, 1, 2, 3), None),
    ("S0-3+Ham off", (0, 1, 2, 3), "bypass"),
]
BACKWARD_SWEEP = [                      # decoder → coarse → fine
    ("Full",         (),           None),
    ("Ham off",      (),           "bypass"),
    ("Ham+S3 off",   (3,),         "bypass"),
    ("Ham+S3-2 off", (2, 3),       "bypass"),
    ("Ham+S3-1 off", (1, 2, 3),    "bypass"),
    ("Ham+S3-0 off", (0, 1, 2, 3), "bypass"),
]


def register_zero_attention_hooks(model, stages):
    """Zero the SpatialAttention output of every block in `stages`.

    Ported from 15_attention_comparison.py: the hook replaces the module's
    output with zeros, so the block reduces to its MLP/residual path. The
    checkpoint is untouched; remove the handles to restore behaviour.
    """
    hooks = []
    for stage_idx in stages:
        block = getattr(model.backbone, f"block{stage_idx + 1}")
        for blk in block:
            def _zero(module, inp, out):
                return torch.zeros_like(out)
            hooks.append(blk.attn.register_forward_hook(_zero))
    return hooks


def register_ham_bypass_hook(model):
    """Replace the Hamburger module's output with ReLU(input).

    Ported from the source 16_hamburger_viz.py bypass ablation: the coarse
    decoder features flow on, but the NMF reconstruction — and with it the
    module's global spatial coupling — is removed. Gradients pass through
    the ReLU normally.
    """
    ham = model.decode_head.hamburger
    return ham.register_forward_hook(
        lambda _m, inp, _o: F.relu(inp[0], inplace=False))


def crest_erf(model, device, img_tensor, region, stages, ham_mode=None):
    """|∂(Σ_region logit_brk)/∂ input| for one ablation condition → (H, W).

    autograd.grad (not backward) keeps the parameters' .grad buffers
    untouched; the region mask is moved to the logits grid if the model
    padded the input to a stride multiple.
    """
    import cv2
    hooks = register_zero_attention_hooks(model, stages)
    if ham_mode == "bypass":
        hooks.append(register_ham_bypass_hook(model))
    x = img_tensor.clone().to(device).requires_grad_(True)
    with torch.enable_grad():
        logits = model(x)
        lh, lw = logits.shape[2], logits.shape[3]
        reg = region
        if reg.shape != (lh, lw):
            reg = cv2.resize(reg.astype(np.uint8), (lw, lh),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        score = logits[0, 1][torch.from_numpy(reg).to(logits.device)].sum()
        grad, = torch.autograd.grad(score, x)
    for h in hooks:
        h.remove()
    return grad.abs().sum(dim=1)[0].detach().cpu().numpy()   # input-sized


def area_ratio(erf, t=AREA_T):
    """High-contribution area ratio (RepLKNet): smallest pixel fraction
    (descending ERF order) containing fraction t of the total mass."""
    flat = np.sort(erf.flatten())[::-1]
    csum = np.cumsum(flat)
    if csum[-1] <= 0:
        return np.nan
    n = np.searchsorted(csum, t * csum[-1]) + 1
    return n / flat.size


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser, image=True)
    args = parser.parse_args()

    apply_style()
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir
    image_path = require_file(args.image, "image")

    print(f"Loading image: {image_path}")
    img_tensor, img_raw, img_hw = viz13.preprocess_image(image_path)

    # Target crest: central connected component of the (size-guarded)
    # prediction — the same target as the Seg-Grad-CAM single-crest row.
    pred = viz13.get_prediction_mask(
        runtime.model, img_tensor, runtime.device, img_hw
    )
    crest = gradcam.central_component(pred)
    print(f"  crest: {crest.sum():,} px of {pred.sum():,} breaking px")

    # ── Compute each unique condition once (Full and all-off are shared
    # between the two sweeps → 10 gradient passes) ──
    cache = {}
    for label, stages, ham_mode in FORWARD_SWEEP + BACKWARD_SWEEP:
        key = (stages, ham_mode)
        if key in cache:
            continue
        print(f"  [{label}] ...")
        g = crest_erf(
            runtime.model, runtime.device, img_tensor, crest, stages, ham_mode
        )
        cache[key] = (g, area_ratio(g))
        print(f"      area ratio (t={AREA_T:.0%}): {cache[key][1]:.3f}")

    # ── 2 × 6: one sweep per row, one more mechanism off per column ──
    # constrained layout spaces titles/labels automatically (no overlap)
    n_cols = len(FORWARD_SWEEP)
    fig, axes = plt.subplots(2, n_cols, figsize=(PAGE_W, PAGE_W * 0.40),
                             layout="constrained")
    row_specs = [("Fine → coarse", FORWARD_SWEEP),
                 ("Decoder → fine", BACKWARD_SWEEP)]
    for row, (row_name, sweep) in enumerate(row_specs):
        for col, (label, stages, ham_mode) in enumerate(sweep):
            ax = axes[row, col]
            g, ar = cache[(stages, ham_mode)]
            disp = np.log10(np.maximum(g / (g.max() + 1e-12), LOG_FLOOR))
            ax.imshow(disp, cmap="inferno", vmin=np.log10(LOG_FLOOR),
                      vmax=0, interpolation="nearest")
            ax.contour(crest.astype(float), levels=[0.5], colors="cyan",
                       linewidths=0.3, alpha=0.9)
            ax.text(0.03, 0.03, f"{100 * ar:.1f}%", transform=ax.transAxes,
                    color="white", fontsize=FONTSIZE_LEGEND,
                    va="bottom", ha="left")
            ax.set_title(label, fontsize=FONTSIZE_LABEL, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(row_name, fontsize=FONTSIZE_LABEL,
                                fontweight="bold")

    fig.suptitle(
        "Crest effective receptive field under cumulative ablation",
        fontsize=FONTSIZE_LABEL, fontweight="bold")
    fig.supxlabel(
        f"log-normalised gradient; cyan: target crest; inset: area@{AREA_T:.0%}",
        fontsize=FONTSIZE_LEGEND, style="italic", color="0.4")
    savefig(fig, OUT_DIR / "19_erf.png", facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
