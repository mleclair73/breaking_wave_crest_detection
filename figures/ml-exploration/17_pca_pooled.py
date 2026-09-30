"""Appendix figures — class-conditioned PCA with bases pooled across images.

Variant of 13_attention_viz.py where the PCA bases are fit on encoder
features pooled from a broader selection of images, then a few sample
images are projected through those shared bases. A basis fit on one image
can encode that image's idiosyncrasies; pooling gives stable components, and
because every sample shares the same basis and colour range, the same colour
means the same feature direction across samples.

(The Hamburger in/out PCA variants were dropped: shared-basis PCA colours
barely move across the NMF, so the rendering carries no signal — the
module's effect is quantified statistically in 16_hamburger_viz.py
instead.)

Outputs (one per SHOW image, tagged by date + alongshore id):

  17_pca_pooled_boundary_<tag>.{pdf,png}
      Boundary-class native PCA (rows: all / breaking / boundary /
      background; columns: encoder stages 0-3), pooled bases.

  17_pc_exemplars_stage{0..3}.{pdf,png}
      For each principal component of a pooled stage basis, the fit images
      that express it most strongly (rows = PCs, cols = top images). An
      image's score for component p is a robust top percentile
      (EXEMPLAR_PCT) of its cells' projections on p. Panels show the PC
      value painted on the scored cells over the dimmed image.

      Conditioning (EXEMPLAR_CLASS): the task is a binary breaking /
      background segmentation, so with the unconditioned "all" basis
      (default) PC1 is essentially the class axis and the exemplars show
      where and how strongly that axis is expressed. Setting "fg" instead
      scores breaking cells through the class-conditioned basis — there
      the fg/bg axis has been removed, so exemplars rank images by
      *intra-breaking* appearance modes (useful for asking what kinds of
      breaking the encoder distinguishes, not which images break most).

Pooling: N_FIT images evenly spaced through the sorted image directory; per
image and class, up to MAX_CELLS_PER_IMAGE cells are sampled (fixed seed).
Cell class labels use each image's own prediction mask, as in the
single-image scripts.

Rendering: every panel (the input included) is drawn from an array at the
exact input-image dimensions — feature-resolution products are NN-upscaled
first — so matplotlib renders all panels at identical size, and prediction
frames are size-guarded against model padding (viz13._match_size).

Run with the `dunex_pytorch` pyenv env:
    python 17_pca_pooled.py
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from exploration_common import (
    ML_DIR,
    PRODUCTION_THRESHOLD,
    add_runtime_arguments,
    import_figure,
    load_runtime,
    require_file,
)
from common.figure_style import FONTSIZE_LABEL, FONTSIZE_LEGEND, PAGE_W, apply_style, savefig

# Reuse the single-image machinery (hooks, masks, helpers, constants).
viz13 = import_figure("13_attention_viz")

IMAGE_DIR = viz13.IMAGE_DIR
OUT_DIR = ML_DIR

NUM_STAGES = viz13.NUM_STAGES
BG_GRAY = viz13.BG_GRAY
PRED_TINT = viz13.PRED_TINT
BOUNDARY_TINT = viz13.BOUNDARY_TINT

# ── Image selection ──────────────────────────────────────────────────────────
N_FIT = 24                  # images pooled into the PCA bases
MAX_CELLS_PER_IMAGE = 10_000   # per class per stage per image (fixed seed)
RNG_SEED = 0

N_EXTRA_SHOW = 2            # additional samples drawn from the fit selection

CLASSES = ("all", "fg", "boundary", "bg")


def _select_fit_images(n_images=N_FIT):
    """N_FIT images evenly spaced through the sorted directory listing."""
    all_imgs = sorted(IMAGE_DIR.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"no images in {IMAGE_DIR}")
    if n_images <= 0:
        raise ValueError("fit image count must be positive")
    idx = np.unique(np.linspace(0, len(all_imgs) - 1, n_images).astype(int))
    return [all_imgs[i] for i in idx]


def _tag(path):
    """'20210919T163000Z_seg..._a0748.png' → '20210919_a0748'."""
    parts = Path(path).stem.split("_")
    return f"{parts[0][:8]}_{parts[-1]}"


# ---------------------------------------------------------------------------
# Forward pass — one hooked pass per image collects everything needed
# ---------------------------------------------------------------------------

def run_image(model, device, image_path):
    """Hooked forward pass for one image.

    Returns dict with: img_display, img_hw, fg_mask, stage feats (list of 4
    (1, C, H, W) tensors), resolutions.
    """
    img_tensor, img_raw, img_hw = viz13.preprocess_image(image_path)

    att_hook = viz13.AttentionHook()
    att_hook.register_hooks(model)

    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    # Output frame must be exactly the input-image size (padding guard)
    prob = viz13._match_size(prob, img_hw)

    att_hook.remove_hooks()

    feats, resolutions = viz13._gather_stage_feats(model, att_hook.activations)
    return {
        "img_display": viz13.denormalize(img_raw),
        "img_hw": img_hw,
        "fg_mask": prob > PRODUCTION_THRESHOLD,
        "feats": feats,
        "resolutions": resolutions,
    }


def _class_cell_masks(fg_mask, feat_hw):
    """{'all', 'fg', 'boundary', 'bg'} → flat bool masks for one stage."""
    fg_cells, boundary, bg_cells = viz13._fg_bg_boundary_cells(fg_mask, feat_hw)
    n = feat_hw[0] * feat_hw[1]
    return {
        "all": np.ones(n, dtype=bool),
        "fg": fg_cells.flatten(),
        "boundary": boundary.flatten(),
        "bg": bg_cells.flatten(),
    }


# ---------------------------------------------------------------------------
# Pooled PCA bases
# ---------------------------------------------------------------------------

class PooledBasis:
    """PCA basis + per-component colour range, fit on pooled features."""

    def __init__(self):
        self.pca = None
        self.lo = None
        self.hi = None

    def fit(self, features):
        if features.shape[0] < 4:
            return self
        self.pca = PCA(n_components=min(3, features.shape[1]))
        proj = self.pca.fit_transform(features)
        self.lo = proj.min(axis=0)
        self.hi = proj.max(axis=0)
        return self

    def rgb(self, feat_flat):
        """Project (N, C) features; normalise with the pooled range."""
        if self.pca is None:
            return None
        proj = self.pca.transform(feat_flat)
        proj = (proj - self.lo) / (self.hi - self.lo + 1e-8)
        return np.clip(proj[:, :3], 0, 1)


def fit_pooled_bases(model, device, fit_images):
    """Fit encoder-stage (per class) PCA bases on pooled cells."""
    rng = np.random.default_rng(RNG_SEED)
    stage_pool = {(si, cls): [] for si in range(NUM_STAGES) for cls in CLASSES}

    for k, path in enumerate(fit_images):
        print(f"  [{k + 1}/{len(fit_images)}] {path.name}")
        res = run_image(model, device, path)

        for si, feat in enumerate(res["feats"]):
            if feat is None:
                continue
            C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
            feat_flat = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()
            masks = _class_cell_masks(res["fg_mask"], (H, W))
            for cls in CLASSES:
                cells = feat_flat[masks[cls]]
                if cells.shape[0] > MAX_CELLS_PER_IMAGE:
                    pick = rng.choice(cells.shape[0], MAX_CELLS_PER_IMAGE,
                                      replace=False)
                    cells = cells[pick]
                if cells.shape[0]:
                    stage_pool[(si, cls)].append(cells)

    print("Fitting pooled PCA bases...")
    bases = {}
    for key, chunks in stage_pool.items():
        feats = (np.concatenate(chunks, axis=0) if chunks
                 else np.empty((0, 1)))
        bases[key] = PooledBasis().fit(feats)
        print(f"  stage {key[0]} {key[1]:>8}: {feats.shape[0]:>8,} cells")
    return bases


# ---------------------------------------------------------------------------
# Figures (pooled-basis renderings of one sample image)
# ---------------------------------------------------------------------------

def _pooled_cells_rgb(feat, cell_mask, basis):
    """Class-conditioned panel through a pooled basis; other cells gray."""
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    feat_flat = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()
    rgb = basis.rgb(feat_flat)
    canvas = np.full((H * W, 3), BG_GRAY, dtype=np.float32)
    mask = cell_mask.flatten()
    if rgb is not None and mask.any():
        canvas[mask] = rgb[mask]
    return canvas.reshape(H, W, 3)


def figure_pooled_boundary(res, bases, tag):
    """Boundary-layout PCA figure for one sample, pooled bases."""
    img_display, img_hw = res["img_display"], res["img_hw"]
    fg_mask, feats, resolutions = res["fg_mask"], res["feats"], res["resolutions"]
    viz_size = (img_hw[1], img_hw[0])

    n_rows, n_cols = 4, 1 + NUM_STAGES
    # constrained layout spaces titles/labels automatically (no overlap)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PAGE_W, PAGE_W * 0.86),
                             layout="constrained")

    row_labels = ["All pixels", "Foreground", "Boundary", "Background"]

    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Input", fontsize=FONTSIZE_LABEL, fontweight="bold")

    pred_fg = img_display.copy()
    pred_fg[fg_mask] = pred_fg[fg_mask] * 0.5 + PRED_TINT * 0.5
    axes[1, 0].imshow(pred_fg)
    axes[1, 0].set_title("Prediction", fontsize=FONTSIZE_LABEL, fontweight="bold")

    bd_overlay = img_display.copy()
    finest = next((i for i, r in enumerate(resolutions) if r[0] > 0), None)
    bd_title = "Boundary"
    if finest is not None:
        _, boundary_f, _ = viz13._fg_bg_boundary_cells(
            fg_mask, resolutions[finest])
        bd_full = cv2.resize(boundary_f.astype(np.uint8), viz_size,
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        bd_overlay[bd_full] = bd_overlay[bd_full] * 0.5 + BOUNDARY_TINT * 0.5
        bd_ds = viz13._downsample_factor(img_hw[0], resolutions[finest][0])
        bd_title = f"Boundary cells (1/{bd_ds})"
    axes[2, 0].imshow(bd_overlay)
    axes[2, 0].set_title(bd_title, fontsize=FONTSIZE_LABEL, fontweight="bold")

    pred_bg = img_display.copy()
    pred_bg[~fg_mask] = pred_bg[~fg_mask] * 0.5 + PRED_TINT * 0.5
    axes[3, 0].imshow(pred_bg)
    axes[3, 0].set_title("Prediction", fontsize=FONTSIZE_LABEL, fontweight="bold")

    for row in range(n_rows):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=FONTSIZE_LABEL,
                                fontweight="bold", rotation=90, labelpad=6)

    for si in range(NUM_STAGES):
        col = 1 + si
        feat = feats[si]
        H, W = resolutions[si]
        ds = viz13._downsample_factor(img_hw[0], H)
        masks = _class_cell_masks(fg_mask, (H, W)) if feat is not None else None
        for row, cls in enumerate(CLASSES):
            ax = axes[row, col]
            if feat is not None:
                panel = _pooled_cells_rgb(
                    feat, masks[cls].reshape(H, W), bases[(si, cls)])
                up = cv2.resize(panel, viz_size,
                                interpolation=cv2.INTER_NEAREST)
                ax.imshow(up)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        if feat is not None:
            axes[0, col].set_title(f"Stage {si} (1/{ds})",
                                   fontsize=FONTSIZE_LABEL, fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    fig.supxlabel(f"PCA bases fit on {N_FIT} pooled images — colours "
                  "comparable across samples",
                  fontsize=FONTSIZE_LEGEND, style="italic", color="0.4")
    savefig(fig, OUT_DIR / f"17_pca_pooled_boundary_{tag}.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PC exemplars — which images express each principal component most strongly
# ---------------------------------------------------------------------------

N_EXEMPLARS = 4          # images shown per principal component
EXEMPLAR_PCT = 99        # robust top-end percentile used as the image score
EXEMPLAR_CLASS = "all"   # basis + cells scored: "all" | "fg" | "bg" | "boundary"


def collect_pc_records(model, device, fit_images, bases):
    """Second pass over the fit images: raw PC projection maps + scores.

    For each image and encoder stage, the EXEMPLAR_CLASS cells' features
    are projected through that class's pooled basis; the image's score for
    component p is the EXEMPLAR_PCT percentile of that projection over
    those cells (a robust "highest activation").
    """
    records = []
    for k, path in enumerate(fit_images):
        print(f"  [{k + 1}/{len(fit_images)}] {path.name}")
        res = run_image(model, device, path)
        rec = {"path": path, "tag": _tag(path), "stages": {}}

        for si, feat in enumerate(res["feats"]):
            basis = bases[(si, EXEMPLAR_CLASS)]
            if feat is None or basis.pca is None:
                continue
            C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
            cells = _class_cell_masks(res["fg_mask"],
                                      (H, W))[EXEMPLAR_CLASS].reshape(H, W)
            if cells.sum() < 4:
                continue
            feat_flat = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()
            proj = basis.pca.transform(feat_flat)          # (N, 3), raw units
            scores = np.percentile(proj[cells.flatten()], EXEMPLAR_PCT,
                                   axis=0)
            rec["stages"][si] = {
                "proj": proj.reshape(H, W, -1).astype(np.float32),
                "cells": cells,
                "scores": scores,
            }

        records.append(rec)
    return records


def _exemplar_panel(ax, path, proj_comp, lo, hi, cell_mask=None):
    """One exemplar: dimmed image with cells painted by the PC value."""
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    H, W = proj_comp.shape
    small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) * 0.35
    v = np.clip((proj_comp - lo) / (hi - lo + 1e-8), 0, 1)
    colors = plt.cm.magma(v)[:, :, :3].astype(np.float32)
    mask = np.ones((H, W), dtype=bool) if cell_mask is None else cell_mask
    small[mask] = colors[mask]
    up = cv2.resize(small, (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    ax.imshow(np.clip(up, 0, 1))
    ax.set_xticks([]); ax.set_yticks([])


def _plot_exemplar_grid(entries, lo, hi, out_name, suptitle, cell_key=None):
    """rows = PCs, cols = top-N_EXEMPLARS records for that PC.

    `entries` is a list of (record, stage_dict) pairs sharing one basis.
    """
    n_pc = len(lo)
    # constrained layout spaces titles/labels automatically (no overlap)
    fig, axes = plt.subplots(n_pc, N_EXEMPLARS,
                             figsize=(PAGE_W, PAGE_W * 0.24 * n_pc),
                             squeeze=False, layout="constrained")
    for p in range(n_pc):
        order = sorted(entries, key=lambda e: e[1]["scores"][p],
                       reverse=True)[:N_EXEMPLARS]
        for col in range(N_EXEMPLARS):
            ax = axes[p, col]
            if col >= len(order):
                ax.axis("off")
                continue
            rec, st = order[col]
            _exemplar_panel(ax, rec["path"], st["proj"][:, :, p],
                            lo[p], hi[p],
                            st.get(cell_key) if cell_key else None)
            ax.set_title(f"{rec['tag']} ({st['scores'][p]:.1f})",
                         fontsize=FONTSIZE_LEGEND)
        axes[p, 0].set_ylabel(f"PC{p + 1}", fontsize=FONTSIZE_LABEL,
                              fontweight="bold")
    fig.suptitle(suptitle, fontsize=FONTSIZE_LABEL, fontweight="bold")
    savefig(fig, OUT_DIR / out_name, facecolor="white", edgecolor="none")
    plt.close(fig)


def figure_pc_exemplars(records, bases):
    """Exemplar grids: one per encoder stage."""
    for si in range(NUM_STAGES):
        basis = bases[(si, EXEMPLAR_CLASS)]
        entries = [(r, r["stages"][si]) for r in records if si in r["stages"]]
        if basis.pca is None or not entries:
            continue
        _plot_exemplar_grid(
            entries, basis.lo, basis.hi,
            f"17_pc_exemplars_stage{si}.png",
            f"Stage {si}: pooled-PCA exemplars ({EXEMPLAR_CLASS} cells)",
            cell_key="cells",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser, image=True)
    parser.add_argument(
        "--fit-images", type=int, default=N_FIT,
        help=f"number of images used to fit PCA bases (default: {N_FIT})",
    )
    parser.add_argument(
        "--extra-show", type=int, default=N_EXTRA_SHOW,
        help=f"additional fitted images to render (default: {N_EXTRA_SHOW})",
    )
    args = parser.parse_args()
    if args.extra_show < 0:
        parser.error("--extra-show must be non-negative")

    apply_style()
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir

    fit_images = _select_fit_images(args.fit_images)
    print(f"Pooling features from {len(fit_images)} images...")
    bases = fit_pooled_bases(runtime.model, runtime.device, fit_images)

    # Show the configured samples plus a few from the fit selection
    show = [require_file(args.image, "image")]
    for p in fit_images:
        if len(show) >= 1 + args.extra_show:
            break
        if p not in show:
            show.append(p)

    for path in show:
        tag = _tag(path)
        print(f"Rendering {path.name} (tag {tag})...")
        res = run_image(runtime.model, runtime.device, path)
        figure_pooled_boundary(res, bases, tag)

    print("Scoring PC activations across the fit images...")
    records = collect_pc_records(
        runtime.model, runtime.device, fit_images, bases
    )
    figure_pc_exemplars(records, bases)

    print("Done.")


if __name__ == "__main__":
    main()
