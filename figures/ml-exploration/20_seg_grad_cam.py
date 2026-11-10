"""Seg-Grad-CAM — class-discriminative evidence maps per encoder stage.

What this shows
---------------
The Δ-residual maps (13_attention_viz.py) show where attention *changes*
the features; Grad-CAM answers the complementary question: *which spatial
evidence supports classifying a region as breaking?* For each encoder stage
we map the gradient-weighted activations that increase the breaking logits
in a target region — first for all predicted breaking, then for one
individual crest, which tests whether the evidence is local to that crest
or drawn from the wider surfzone.

Method — Grad-CAM for segmentation, element-wise (HiResCAM) variant
-------------------------------------------------------------------
- Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep
  Networks via Gradient-based Localization", ICCV. arXiv:1610.02391.
- Vinogradova, Dibrov & Myers (2020), "Towards Interpretable Semantic
  Segmentation via Gradient-weighted Class Activation Mapping
  (SEG-GRAD-CAM)", AAAI (SA). arXiv:2002.11434.
- Draelos & Carin (2020), "Use HiResCAM instead of Grad-CAM for faithful
  explanations of convolutional neural networks". arXiv:2011.08891.

For a target region M and class c (breaking), the scalar score is the sum
of class-c logits over M:  y = Σ_{(i,j)∈M} logit_c(i, j),  and gradients
are taken w.r.t. each encoder stage's feature map A^k (the last MSCAN
block's SpatialAttention output — the layer analysed by the other appendix
figures).

Why element-wise weighting: classic Grad-CAM averages the gradient over
space (α_k = mean_ij ∂y/∂A^k_ij) and paints Σ_k α_k A^k — the CAM then
lights up *everywhere channel k is active*, not where the gradient was.
In this binary task the breaking-selective channels are active along
every crest in the image, so a small target region (one crest) yields a
diluted, near-zero CAM: exactly the "no signal" failure mode. HiResCAM
keeps the gradient's location by multiplying element-wise before the
channel sum:

    L = Σ_k  ∂y/∂A^k ⊙ A^k

which Draelos & Carin prove reflects the model's actual computation for
the final conv layer, and which preserves the crest-local evidence here.
Set CAM_STYLE = "gradcam" to reproduce the classic behaviour.

Display — both renderings are produced
--------------------------------------
1. Signed CAM on a symmetric log scale (20_seg_grad_cam): red = cells
   whose activation pushes the target's breaking score up (supporting
   evidence), blue = cells that push it down (suppressive evidence —
   e.g. competing crests or foam the model discounts). SymLogNorm has a
   linear band around zero (LINTHRESH), so true zeros render as neutral
   white, small values stay visible, and nothing is clipped.
2. ReLU + LogNorm (20_seg_grad_cam_relu): the classic positive-evidence
   rendering, inferno on a log scale. Note why most of it sits at the
   floor: the ReLU zeroes every suppressive cell, and cells outside the
   target's gradient cone are *exactly* zero — log scaling has no
   representation for 0, so all of those pin to the clip floor. Useful
   as the conventional view; the signed version carries more
   information.

Target regions
--------------
Row 1: every predicted breaking pixel (global evidence for the class).
Row 2: the single connected breaking component nearest the image centre
       (evidence for one crest) — locality here means the model's evidence
       is crest-specific rather than a generic surfzone response.

Outputs
-------
  20_seg_grad_cam.{pdf,png}       — signed, RdBu_r + SymLogNorm
  20_seg_grad_cam_relu.{pdf,png}  — ReLU'd, inferno + LogNorm (clipped)
      Each: 2 rows (targets) × 6 cols: input with the target region
      tinted transparent blue | stage 0-3 CAMs | combined (mean of
      per-stage CAMs). CAMs are normalised per panel to max |.| = 1,
      NN-upscaled to the exact input-image size so every panel renders at
      the same matplotlib size, with the target region contoured.

Run with the `dunex_pytorch` pyenv env (needs gradients):
    python 20_seg_grad_cam.py
"""

import argparse

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

from exploration_common import (
    ML_DIR,
    add_runtime_arguments,
    import_figure,
    load_runtime,
    require_file,
)
from common.figure_style import FONTSIZE_LABEL, FONTSIZE_LEGEND, PAGE_W, apply_style, savefig

viz13 = import_figure("13_attention_viz")

OUT_DIR = ML_DIR
NUM_STAGES = viz13.NUM_STAGES

MIN_COMPONENT_PX = 30    # ignore tiny specks when picking the single crest
CAM_STYLE = "hirescam"   # "hirescam" (element-wise, faithful) | "gradcam"
LINTHRESH = 0.03         # SymLogNorm linear band half-width (of max |CAM|)


class StageOutputGrabber:
    """Keep the *live* (graph-attached) SpatialAttention output of the last
    block per stage during one grad-enabled forward pass.

    Unlike viz13.AttentionHook (which detaches to CPU for feature
    analysis), Grad-CAM needs tensors that remain part of the autograd
    graph so we can take ∂score/∂A^k afterwards.
    """

    def __init__(self, model):
        self.acts = {}
        self.hooks = []
        for si in range(NUM_STAGES):
            block = getattr(model.backbone, f"block{si + 1}")
            last = len(block) - 1

            def _make(si):
                def fn(_m, _i, out):
                    self.acts[si] = out
                return fn
            self.hooks.append(block[last].attn.register_forward_hook(_make(si)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def central_component(pred):
    """Connected breaking component whose centroid is nearest the centre.

    Components smaller than MIN_COMPONENT_PX are ignored. Returns a bool
    mask (all-False if there is no breaking at all).
    """
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8), connectivity=8)
    H, W = pred.shape
    best, best_d = None, np.inf
    for lab in range(1, n):                      # label 0 = background
        if stats[lab, cv2.CC_STAT_AREA] < MIN_COMPONENT_PX:
            continue
        cx, cy = centroids[lab]
        d = (cy - H / 2) ** 2 + (cx - W / 2) ** 2
        if d < best_d:
            best, best_d = lab, d
    return labels == best if best is not None else np.zeros_like(pred, bool)


def seg_grad_cam(model, device, img_tensor, region):
    """Per-stage Seg-Grad-CAM maps for one target region.

    A fresh forward pass per target keeps the graph small; the score is
    y = Σ_{M} logit_brk and gradients are taken with autograd.grad so the
    parameters' .grad buffers are never touched.
    """
    grabber = StageOutputGrabber(model)
    x = img_tensor.to(device)
    with torch.enable_grad():
        logits = model(x)
        # The region is defined on the (size-guarded) input grid; if the
        # model padded to a stride multiple, move it to the logits grid.
        lh, lw = logits.shape[2], logits.shape[3]
        if region.shape != (lh, lw):
            region = cv2.resize(region.astype(np.uint8), (lw, lh),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
        region_t = torch.from_numpy(region).to(logits.device)
        score = logits[0, 1][region_t].sum()
        acts = [grabber.acts[si] for si in range(NUM_STAGES)]
        grads = torch.autograd.grad(score, acts)
    grabber.remove()

    cams = []
    for act, grad in zip(acts, grads):
        if CAM_STYLE == "hirescam":
            # Element-wise grad ⊙ activation keeps the evidence where the
            # gradient actually was (Draelos & Carin 2020) — essential for
            # small target regions in this sparse binary task.
            cam = (grad * act).sum(dim=1)[0]                 # (H', W')
        else:
            # Classic Grad-CAM: spatially averaged channel weights
            alpha = grad.mean(dim=(2, 3), keepdim=True)      # (1, C, 1, 1)
            cam = (alpha * act).sum(dim=1)[0]                # (H', W')
        # Signed, normalised to max |CAM| = 1 (see docstring: no ReLU —
        # suppressive evidence is informative and true zeros should render
        # neutral, not clip)
        cam = cam.detach().cpu().numpy()
        cams.append(cam / (np.abs(cam).max() + 1e-12))
    return cams


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser, image=True)
    parser.add_argument(
        "--mode", choices=("signed", "relu", "both"), default="both",
        help="CAM rendering to write (default: both)",
    )
    args = parser.parse_args()

    apply_style()
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir
    image_path = require_file(args.image, "image")

    print(f"Loading image: {image_path}")
    img_tensor, img_raw, img_hw = viz13.preprocess_image(image_path)
    img_display = viz13.denormalize(img_raw)

    # Regions are defined on the model's own output grid; the prediction
    # mask is size-guarded so overlays align with the input exactly.
    pred = viz13.get_prediction_mask(
        runtime.model, img_tensor, runtime.device, img_hw
    )
    crest = central_component(pred)
    targets = [
        ("All breaking", pred),
        ("Single crest", crest),
    ]
    print(f"  breaking px: {pred.sum():,} | central crest px: {crest.sum():,}")

    # Compute the CAMs once per target; render both views from the cache
    results = []
    for label, region in targets:
        print(f"Seg-Grad-CAM: {label}...")
        results.append(
            (
                label,
                region,
                seg_grad_cam(
                    runtime.model, runtime.device, img_tensor, region
                ),
            )
        )

    outputs = {
        "signed": "20_seg_grad_cam.png",
        "relu": "20_seg_grad_cam_relu.png",
    }
    modes = outputs if args.mode == "both" else (args.mode,)
    for mode in modes:
        out_name = outputs[mode]
        print(f"Rendering {mode} view → {out_name}")
        render_cam_figure(mode, out_name, results, img_display, img_hw)
    print("Done.")


def render_cam_figure(mode, out_name, results, img_display, img_hw):
    """One 2×6 CAM figure in either display mode (see module docstring).

    mode = "signed": RdBu_r + SymLogNorm — red supports / blue suppresses
    the target's breaking score, white = no influence, nothing clipped.
    mode = "relu":   inferno + LogNorm on ReLU'd CAMs — the conventional
    positive-evidence view; zeros pin to the LOG floor by construction.
    """
    viz_size = (img_hw[1], img_hw[0])
    fig, axes = plt.subplots(2, 2 + NUM_STAGES,
                             figsize=(PAGE_W, PAGE_W * 0.40), squeeze=False)

    if mode == "signed":
        norm = SymLogNorm(linthresh=LINTHRESH, vmin=-1.0, vmax=1.0)
        cmap = "RdBu_r"
        theme_color = "lime"     # visible on the diverging map

        def _view(cam):
            return cam
    else:
        norm = LogNorm(vmin=1e-3, vmax=1.0)
        cmap = "inferno"
        theme_color = "cyan"

        def _view(cam):
            pos = np.maximum(cam, 0)
            return np.clip(pos / (pos.max() + 1e-12), 1e-3, 1.0)

    dilation_kernel = np.ones((3, 3), np.uint8)
    im = None

    for row, (label, region, cams) in enumerate(results):

        # Configure styling properties based on row target context
        if label == "Single crest":
            contour_mask = cv2.dilate(region.astype(np.uint8), dilation_kernel, iterations=1).astype(float)
            contour_width = 0.6
            contour_alpha = 1.0
        else:
            contour_mask = region.astype(float)
            contour_width = 0.4
            contour_alpha = 0.8

        # Col 0: input with the target region as a *transparent blue* tint —
        # opaque fills block the underlying crest structure the CAM is
        # supposed to be compared against.
        REGION_TINT = np.array([0.25, 0.45, 1.0], dtype=np.float32)
        REGION_ALPHA = 0.4
        overlay = img_display.copy()
        overlay[region] = (overlay[region] * (1 - REGION_ALPHA)
                           + REGION_TINT * REGION_ALPHA)
        axes[row, 0].imshow(overlay)
        
        # Row labels on the LHS
        axes[row, 0].set_ylabel(label, fontsize=FONTSIZE_LABEL, fontweight="bold")
        if row == 0:
            axes[row, 0].set_title("Input + target", fontsize=FONTSIZE_LABEL,
                                   fontweight="bold")

        # Cols 1-4: per-stage CAMs through the mode's view/norm/cmap
        cam_ups = [cv2.resize(c, viz_size, interpolation=cv2.INTER_NEAREST)
                   for c in cams]
        for si, cam_up in enumerate(cam_ups):
            ax = axes[row, 1 + si]
            im = ax.imshow(_view(cam_up), cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.contour(contour_mask, levels=[0.5], colors=theme_color,
                       linewidths=contour_width, alpha=contour_alpha)
            if row == 0:
                H, W = cams[si].shape
                ds = viz13._downsample_factor(img_hw[0], H)
                ax.set_title(f"Stage {si} (1/{ds})",
                             fontsize=FONTSIZE_LABEL, fontweight="bold")

        # Combined panel (col 5): mean of the signed per-stage CAMs,
        # renormalised to max |.| = 1, then passed through the same view
        combined = np.mean(cam_ups, axis=0)
        combined = combined / (np.abs(combined).max() + 1e-12)

        ax = axes[row, 1 + NUM_STAGES]
        ax.imshow(_view(combined), cmap=cmap, norm=norm,
                  interpolation="nearest")
        ax.contour(contour_mask, levels=[0.5], colors=theme_color,
                   linewidths=contour_width, alpha=contour_alpha)
        if row == 0:
            ax.set_title("Combined", fontsize=FONTSIZE_LABEL,
                         fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    # Grid compression to clear room for the shared sidebar colorbar on the RHS
    fig.subplots_adjust(bottom=0.07, top=0.90, left=0.03, right=0.92,
                        wspace=0.05, hspace=0.06)

    cbar_ax = fig.add_axes([0.94, 0.07, 0.015, 0.83])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=FONTSIZE_LEGEND)

    footers = {
        "signed": "Signed Seg-HiResCAM: red supports, blue suppresses; "
                  "lime marks the target boundary",
        "relu": "Positive Seg-HiResCAM (log scale); cyan marks the target "
                "boundary",
    }
    fig.text(0.48, 0.01,
             footers[mode],
             ha="center", va="bottom", fontsize=FONTSIZE_LEGEND,
             style="italic", color="0.4")

    savefig(fig, OUT_DIR / out_name, facecolor="white", edgecolor="none")
    plt.close(fig)


if __name__ == "__main__":
    main()
