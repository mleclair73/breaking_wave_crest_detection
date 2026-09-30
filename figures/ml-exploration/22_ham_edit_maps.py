"""Appendix figures — what the Hamburger edit does to the breaking class.

Why these figures (and why not before/after NMF atoms)
------------------------------------------------------
The statistics figures (16, 21) show *that* the Hamburger NMF (Geng et
al. 2021, "Is Attention Better Than Matrix Decomposition?", ICLR,
arXiv:2109.04553) changes the class geometry; these show *what that
accomplishes* on one timestack. A before/after comparison of DFF atoms
(18) would be nearly circular: the module is residual
(out = in + correction) and its correction lies in the span of its own
NMF atoms, so a shared dictionary would return the same atoms with
marginally cleaner coefficients. Instead everything here renders the
features themselves, before vs after, under one *shared* basis so the
two sides are directly comparable, and the earlier 3-PC RGB rendering
(hard to compare channel-by-channel across rows) is replaced by
single-PC views.

Shared machinery
----------------
- Breaking/background labels are the production prediction (P > 0.48)
  mapped to the Ham grid with the "any pixel" rule, as in 16/18/21.
- Class-conditioned PCA (as in DINOv2, Oquab et al. 2024,
  arXiv:2304.07193): one PCA fit on the *pooled* input+output breaking
  cells; both sides are transformed with it, so scores are comparable.
- Before/after encoding matches 16/21: input = filled markers / solid
  outlines, output = open markers / dashed outlines.
- Every map is NN-upscaled to the exact input-image size so matplotlib
  renders all panels identically (repo-wide convention).

Figure 22_ham_edit_feature_space — the edit in feature space
-------------------------------------------------------------
(a) Breaking cells in the shared PC1–PC2 plane: input = filled dots,
    output = open dots, one grey segment per cell connecting its
    before→after position (a displacement field of the edit). Diamonds
    mark the class centroids; the grey × is the background centroid
    (input features) projected into the same plane — the direction the
    breaking cloud moves away from. A visibly stretched output cloud =
    the edit adds descriptive variation to the class; the whole cloud
    shifting away from the × = the separation increase of 16/21(c).
(b, c) Histograms of the breaking cells' PC1 and PC2 scores, input
    (filled, solid) vs output (open, dashed), shared bins per panel.
    Wider output histograms are the variance increase along the class's
    two dominant modes — comparing the panels shows whether the edit is
    concentrated in one direction or spread across the leading modes;
    the corner text gives each side's std.
(d) The same displacement scatter for the *background* class: PCA fit
    on the pooled input+output background cells, drawn at the class's
    own scale — background's variation is invisible at breaking's scale
    (and vice versa), so each class gets its own panel instead of one
    shared view. A background subsample keeps the segments legible.
    Nonlinear joint embeddings (t-SNE / UMAP) were tried here and
    dropped: slow, and the cluster shredding made the before/after
    comparison unreadable — two class-conditioned linear views say it
    directly.

Figure 22_ham_edit_maps — the edit in image space
--------------------------------------------------
Rows = Ham input / Ham output; colours computed once, applied to both.
- Column 1: the timestack (top) and the model's breaking prediction
  (bottom) for context.
- Column 2, "PC1": each breaking cell coloured by its PC1 score
  (background grey). The scores are *z-scored per side* over that
  side's breaking cells before rendering, so the two rows have matched
  colour distributions by construction and compare pure spatial
  pattern — the PCA basis is already shared (one fit on pooled
  input+output cells), and the z-scoring removes the residual
  mean/scale shift the module adds along it. The *amount* of variance
  change deliberately does not appear here; it lives in the companion
  figure's histogram and in the corner number: breaking intraclass
  variance relative to that side's pooled within-class variance RMS_w²
  (scale-invariant, and — unlike a share of total variance — not
  deflated mechanically when the between-class gap grows; same
  normalisation as 16/21 panel b).
- Column 3, "Class-axis projection": each cell's feature projected onto
  the fixed class axis u = (μ_brk − μ_bg)/‖·‖ computed from the *input*
  features, centred on the centroid midpoint (shared symmetric
  diverging scale). Red = breaking side, blue = background side;
  stronger red on crests after the module = breaking cells pushed out
  along the discriminant. Corner number: Fisher-style separation
  ‖μ_brk − μ_bg‖ / RMS_w of that side.
- Column 4, "Ham edit" (the diff): top = ΔPC1 = PC1(out) − PC1(in) per
  breaking cell (diverging, own symmetric scale) — in *which direction*
  along the dominant mode each cell moved; bottom = ‖ΔPC‖, the length
  of each cell's displacement across all N_PCS scores (sequential) —
  *how much* it moved. Qualitative panels: corner labels instead of
  colorbars.

Figure 22_ham_edit_rgb — the original 3-PC RGB overview (kept, with a
caveat)
------------------------------------------------------------------------
Rows = Ham input / Ham output: context | breaking-class PCA as RGB
(top-3 shared-basis PC scores → R,G,B with pooled 2–98 percentile
limits, background grey) | class-axis projection. **Caveat:** although
the PCA basis is shared, the module shifts and stretches the scores
along it, so row-to-row colour differences mix genuine spatial-pattern
change with that mean/scale drift — the PCs (and their colours) are not
directly comparable between rows. Read this figure as a qualitative
overview only; the z-scored PC1 maps (pattern) and the histogram /
scatter / share numbers (magnitude) in the other two figures are the
comparable views.

Outputs
-------
  22_ham_edit_feature_space.{pdf,png}
  22_ham_edit_maps.{pdf,png}
  22_ham_edit_rgb.{pdf,png}

Run with the `dunex_pytorch` pyenv env:
    python 22_ham_edit_maps.py
"""

import argparse

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from exploration_common import (
    ML_DIR,
    PRODUCTION_THRESHOLD,
    add_runtime_arguments,
    import_figure,
    load_runtime,
    require_file,
)
from common.figure_style import (
    CMAP,
    FONTSIZE_LABEL,
    FONTSIZE_LEGEND,
    PAGE_W,
    apply_style,
    panel_label,
    savefig,
    styled_legend,
)

# Single-image Ham script provides paths, preprocessing, the I/O hook,
# and mask downsampling; 21 provides the separation statistic.
ham16 = import_figure("16_hamburger_viz")
ham21 = import_figure("21_ham_variability_pooled")

OUT_DIR = ML_DIR
FG_COL, BG_COL = ham16.FG_COL, ham16.BG_COL

# ── Config ───────────────────────────────────────────────────────────────────
N_PCS = 3               # PCA components kept (PC1 shown; all used for ‖ΔPC‖)
BG_GRAY = 0.85          # grey level for non-breaking cells in the maps
PCT_LO, PCT_HI = 2, 98  # pooled RGB channel limits (22_ham_edit_rgb)
PCT_SYM = 98            # symmetric percentile limit for diverging maps
PROJ_PCT = 99           # symmetric limit percentile for the projection
N_HIST_BINS = 24        # shared bins of the PC1 histogram
MAX_SCATTER = 500       # cap on cells drawn in the breaking scatter
N_BG_SCATTER = 250      # background cells drawn in the background scatter
RNG_SEED = 0


def _upscale(arr, img_hw):
    """Ham-grid map → exact input-image size (NN keeps cells crisp;
    NaNs pass through untouched)."""
    return cv2.resize(arr.astype(np.float32), (img_hw[1], img_hw[0]),
                      interpolation=cv2.INTER_NEAREST)


def _masked_map(values, fg, grid_hw):
    """Per-cell values → Ham-grid map with background cells NaN (drawn
    in the colormap's 'bad' grey)."""
    m = np.full(values.shape[0], np.nan)
    m[fg] = values[fg]
    return m.reshape(*grid_hw)


def _grey_cmap(name):
    cm = plt.get_cmap(name).copy()
    cm.set_bad(str(BG_GRAY))
    return cm


def _pca_rgb(Y, fg, lo, hi, grid_hw):
    """Scores (N, 3) → RGB image on the Ham grid: breaking cells coloured
    by the pooled-limit normalised PCs, background cells flat grey."""
    rgb = np.clip((Y - lo) / (hi - lo + 1e-12), 0, 1)
    out = np.full((Y.shape[0], 3), BG_GRAY)
    out[fg] = rgb[fg]
    return out.reshape(*grid_hw, 3)


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser, image=True)
    args = parser.parse_args()

    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir
    image_path = require_file(args.image, "image")

    print(f"Loading image: {image_path}")
    img_tensor, img_raw, img_hw = ham16.preprocess_image(image_path)
    img_display = ham16.denormalize(img_raw)

    print("Forward pass (hooked)...")
    hook = ham16.HamburgerIOHook()
    hook.register(runtime.model)
    with torch.no_grad():
        logits = runtime.model(img_tensor.to(runtime.device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    hook.remove()
    if prob.shape != img_hw:      # output frame must equal the input size
        prob = cv2.resize(prob, (img_hw[1], img_hw[0]),
                          interpolation=cv2.INTER_NEAREST)
    fg_mask = prob > PRODUCTION_THRESHOLD

    C, H, W = hook.inp.shape[1], hook.inp.shape[2], hook.inp.shape[3]
    feat_in = hook.inp[0].permute(1, 2, 0).reshape(-1, C).numpy()
    feat_out = hook.out[0].permute(1, 2, 0).reshape(-1, C).numpy()
    fg = ham16._downsample_mask(fg_mask, (H, W)).flatten()
    print(f"  Ham grid {H}×{W}, {int(fg.sum())} breaking cells "
          f"({100 * fg.mean():.1f}%)")
    if fg.sum() < N_PCS + 1:
        raise RuntimeError("too few breaking cells for class-conditioned PCA")

    # ── Shared breaking-class PCA basis (fit on input+output cells) ──
    pca = PCA(n_components=N_PCS).fit(np.vstack([feat_in[fg], feat_out[fg]]))
    Y_in, Y_out = pca.transform(feat_in), pca.transform(feat_out)
    ev = pca.explained_variance_ratio_
    print("  shared breaking PCA, explained variance: "
          + ", ".join(f"PC{i + 1} {e:.2f}" for i, e in enumerate(ev)))

    # ── Fixed class axis from the *input* features ──
    mu_brk, mu_bg = feat_in[fg].mean(axis=0), feat_in[~fg].mean(axis=0)
    u = mu_brk - mu_bg
    u /= np.linalg.norm(u) + 1e-12
    mid = 0.5 * (mu_brk + mu_bg)
    proj = {"in": ((feat_in - mid) @ u).reshape(H, W),
            "out": ((feat_out - mid) @ u).reshape(H, W)}
    proj_vmax = max(np.percentile(np.abs(p), PROJ_PCT) for p in proj.values())

    # ── Row statistics (corner annotations; also printed) ──
    # Breaking variance relative to the side's pooled within-class
    # variance RMS_w² (same normalisation as 16/21 panel b): raw
    # variances inflate with the residual module's output norms, and a
    # share of *total* variance would deflate mechanically as the
    # between-class gap grows — RMS_w² cancels the scale and excludes
    # the centroid gap.
    rel_var, sep = {}, {}
    for which, feat in (("in", feat_in), ("out", feat_out)):
        cls_var = float(
            ((feat[fg] - feat[fg].mean(axis=0)) ** 2).sum(axis=1).mean())
        pw = float(np.mean(ham16.within_class_deviation(feat, fg) ** 2))
        rel_var[which] = cls_var / (pw + 1e-12)
        sep[which] = ham21._separation(feat, fg)
        print(f"  {which:3s}: relative breaking variance "
              f"{rel_var[which]:.3f}, separation {sep[which]:.2f}")

    # ════════════════════════════════════════════════════════════════
    # Figure 1 — the edit in feature space
    # (breaking scatter + PC1/PC2 histograms + background scatter)
    # ════════════════════════════════════════════════════════════════
    fig, (ax_s, ax_h1, ax_h2, ax_t) = plt.subplots(
        1, 4, figsize=(PAGE_W, PAGE_W * 0.28), layout="constrained")

    # (a) PC1–PC2 displacement scatter of the breaking cells
    ax = ax_s
    idx = np.flatnonzero(fg)
    if idx.size > MAX_SCATTER:
        idx = rng.choice(idx, MAX_SCATTER, replace=False)
    for i in idx:      # grey before→after segment per cell
        ax.plot([Y_in[i, 0], Y_out[i, 0]], [Y_in[i, 1], Y_out[i, 1]],
                color="0.75", linewidth=0.4, alpha=0.6, zorder=1)
    ax.scatter(Y_in[idx, 0], Y_in[idx, 1], s=9, color=FG_COL, alpha=0.8,
               linewidths=0, zorder=2)
    ax.scatter(Y_out[idx, 0], Y_out[idx, 1], s=9, facecolors="none",
               edgecolors=FG_COL, linewidths=0.6, alpha=0.8, zorder=2)
    # class centroids (filled/open diamonds) + background reference (×)
    ax.scatter(*Y_in[fg].mean(axis=0)[:2], s=45, color=FG_COL, marker="D",
               edgecolor="black", linewidths=0.8, zorder=3)
    ax.scatter(*Y_out[fg].mean(axis=0)[:2], s=45, facecolors="white",
               edgecolors="black", marker="D", linewidths=0.8, zorder=3)
    bg_c = pca.transform(mu_bg[None, :])[0]
    ax.scatter(bg_c[0], bg_c[1], s=45, color="0.35", marker="x",
               linewidths=1.2, zorder=3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Breaking cells", fontsize=FONTSIZE_LABEL)
    handles = [
        Line2D([], [], linestyle="", marker="o", color=FG_COL,
               markersize=4, label="Input"),
        Line2D([], [], linestyle="", marker="o", markerfacecolor="none",
               markeredgecolor=FG_COL, markersize=4, label="Output"),
        Line2D([], [], linestyle="", marker="x", color="0.35",
               markersize=5, label="Background centroid"),
    ]
    styled_legend(ax, handles=handles, loc="best")
    panel_label(ax, "a")

    # (b, c) PC1 / PC2 histograms, input (filled) vs output (dashed)
    for ax, k, lab in ((ax_h1, 0, "b"), (ax_h2, 1, "c")):
        s_in, s_out = Y_in[fg, k], Y_out[fg, k]
        bins = np.histogram_bin_edges(np.concatenate([s_in, s_out]),
                                      bins=N_HIST_BINS)
        ax.hist(s_in, bins=bins, histtype="stepfilled", alpha=0.45,
                facecolor=FG_COL, edgecolor=FG_COL, linewidth=0.8,
                label="Input")
        ax.hist(s_out, bins=bins, histtype="step", edgecolor=FG_COL,
                linewidth=1.0, linestyle="--", label="Output")
        ax.set_xlabel(f"PC{k + 1} score")
        ax.text(0.03, 0.97, f"std {s_in.std():.1f} → {s_out.std():.1f}",
                transform=ax.transAxes, fontsize=FONTSIZE_LEGEND,
                color="0.25", va="top", ha="left")
        panel_label(ax, lab, loc="upper right")
        print(f"  PC{k + 1} std: {s_in.std():.2f} → {s_out.std():.2f}")
    ax_h1.set_ylabel("Breaking cells")
    styled_legend(ax_h1)

    # (c) background displacement scatter in the background's own PCA
    # basis — its variation is invisible at breaking's scale, so the
    # class gets its own panel (subsampled for legible segments)
    pca_bg = PCA(n_components=2).fit(
        np.vstack([feat_in[~fg], feat_out[~fg]]))
    B_in, B_out = pca_bg.transform(feat_in), pca_bg.transform(feat_out)
    idx_bg = np.flatnonzero(~fg)
    if idx_bg.size > N_BG_SCATTER:
        idx_bg = rng.choice(idx_bg, N_BG_SCATTER, replace=False)

    ax = ax_t
    for i in idx_bg:
        ax.plot([B_in[i, 0], B_out[i, 0]], [B_in[i, 1], B_out[i, 1]],
                color="0.75", linewidth=0.4, alpha=0.6, zorder=1)
    ax.scatter(B_in[idx_bg, 0], B_in[idx_bg, 1], s=9, color=BG_COL,
               alpha=0.8, linewidths=0, zorder=2)
    ax.scatter(B_out[idx_bg, 0], B_out[idx_bg, 1], s=9,
               facecolors="none", edgecolors=BG_COL, linewidths=0.6,
               alpha=0.8, zorder=2)
    ax.scatter(*B_in[~fg].mean(axis=0)[:2], s=45, color=BG_COL,
               marker="D", edgecolor="black", linewidths=0.8, zorder=3)
    ax.scatter(*B_out[~fg].mean(axis=0)[:2], s=45, facecolors="white",
               edgecolors="black", marker="D", linewidths=0.8, zorder=3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Background cells", fontsize=FONTSIZE_LABEL)
    panel_label(ax, "d")

    savefig(fig, OUT_DIR / "22_ham_edit_feature_space.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════
    # Figure 2 — the edit in image space (PC1 | projection | diff)
    # ════════════════════════════════════════════════════════════════
    # Distribution-matched PC1 for the maps: z-score each side over its
    # own breaking cells (the basis is shared; this removes the residual
    # mean/scale shift so the rows compare spatial pattern only)
    pc1_z = {}
    for which, scores in (("in", Y_in[:, 0]), ("out", Y_out[:, 0])):
        m, s = scores[fg].mean(), scores[fg].std() + 1e-12
        pc1_z[which] = (scores - m) / s
    pc1_lim = np.percentile(
        np.abs(np.concatenate([pc1_z["in"][fg], pc1_z["out"][fg]])), PCT_SYM)
    dY = Y_out - Y_in
    dpc1_lim = np.percentile(np.abs(dY[fg, 0]), PCT_SYM) + 1e-12
    dmag = np.linalg.norm(dY, axis=1)
    dmag_vmax = np.percentile(dmag[fg], PROJ_PCT) + 1e-12

    fig, axes = plt.subplots(2, 4, figsize=(PAGE_W, PAGE_W * 0.52),
                             layout="constrained")
    col_titles = ["Timestack / prediction", "PC1 (breaking cells)",
                  "Class-axis projection", "Ham edit"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=FONTSIZE_LABEL, fontweight="bold")

    # column 1: context (image, then prediction)
    axes[0, 0].imshow(img_display)
    axes[1, 0].imshow(img_display)
    axes[1, 0].contour(fg_mask.astype(float), levels=[0.5],
                       colors=[FG_COL], linewidths=0.6)

    for row, which in enumerate(("in", "out")):
        # column 2: PC1 per breaking cell, z-scored per side so the two
        # rows have matched colour distributions (pattern comparison)
        ax = axes[row, 1]
        im_pc1 = ax.imshow(
            _upscale(_masked_map(pc1_z[which], fg, (H, W)), img_hw),
            cmap=_grey_cmap(CMAP["diverge"]), vmin=-pc1_lim, vmax=pc1_lim)
        ax.text(0.03, 0.03, f"rel. variance {rel_var[which]:.2f}",
                transform=ax.transAxes, fontsize=FONTSIZE_LEGEND,
                color="black", va="bottom", ha="left")

        # column 3: projection onto the fixed input-derived class axis
        ax = axes[row, 2]
        im_proj = ax.imshow(_upscale(proj[which], img_hw),
                            cmap=CMAP["diverge"],
                            vmin=-proj_vmax, vmax=proj_vmax)
        ax.text(0.03, 0.03, f"separation {sep[which]:.2f}",
                transform=ax.transAxes, fontsize=FONTSIZE_LEGEND,
                color="black", va="bottom", ha="left")

        axes[row, 0].set_ylabel("Input" if which == "in" else "Output",
                                fontsize=FONTSIZE_LABEL, fontweight="bold")

    # column 4: the diff — direction (ΔPC1) on top, magnitude (‖ΔPC‖) below
    ax = axes[0, 3]
    ax.imshow(_upscale(_masked_map(dY[:, 0], fg, (H, W)), img_hw),
              cmap=_grey_cmap(CMAP["diverge"]),
              vmin=-dpc1_lim, vmax=dpc1_lim)
    ax.text(0.03, 0.03, "ΔPC1 (out − in)", transform=ax.transAxes,
            fontsize=FONTSIZE_LEGEND, color="black", va="bottom", ha="left")
    ax = axes[1, 3]
    ax.imshow(_upscale(_masked_map(dmag, fg, (H, W)), img_hw),
              cmap=_grey_cmap("inferno"), vmin=0, vmax=dmag_vmax)
    ax.text(0.03, 0.03, "‖ΔPC‖", transform=ax.transAxes,
            fontsize=FONTSIZE_LEGEND, color="white", va="bottom", ha="left")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    cb = fig.colorbar(im_pc1, ax=[axes[0, 1], axes[1, 1]],
                      location="bottom", shrink=0.85, pad=0.02)
    cb.set_label("PC1 (z-scored per side)", fontsize=FONTSIZE_LEGEND)
    cb = fig.colorbar(im_proj, ax=[axes[0, 2], axes[1, 2]],
                      location="bottom", shrink=0.85, pad=0.02)
    cb.set_label("Feature · class axis", fontsize=FONTSIZE_LEGEND)

    savefig(fig, OUT_DIR / "22_ham_edit_maps.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)

    # ════════════════════════════════════════════════════════════════
    # Figure 3 — original 3-PC RGB overview (qualitative; the module's
    # mean/scale drift along the shared PCs is left in, so rows are not
    # directly comparable — see the docstring caveat)
    # ════════════════════════════════════════════════════════════════
    pooled_brk = np.vstack([Y_in[fg], Y_out[fg]])
    lo = np.percentile(pooled_brk, PCT_LO, axis=0)
    hi = np.percentile(pooled_brk, PCT_HI, axis=0)
    rgb = {"in": _pca_rgb(Y_in, fg, lo, hi, (H, W)),
           "out": _pca_rgb(Y_out, fg, lo, hi, (H, W))}

    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, PAGE_W * 0.66),
                             layout="constrained")
    for col, t in enumerate(["Timestack / prediction",
                             "Breaking-class PCA (RGB)",
                             "Class-axis projection"]):
        axes[0, col].set_title(t, fontsize=FONTSIZE_LABEL, fontweight="bold")
    axes[0, 0].imshow(img_display)
    axes[1, 0].imshow(img_display)
    axes[1, 0].contour(fg_mask.astype(float), levels=[0.5],
                       colors=[FG_COL], linewidths=0.6)
    for row, which in enumerate(("in", "out")):
        ax = axes[row, 1]
        ax.imshow(np.clip(_upscale(rgb[which], img_hw), 0, 1))
        ax.text(0.03, 0.03, f"rel. variance {rel_var[which]:.2f}",
                transform=ax.transAxes, fontsize=FONTSIZE_LEGEND,
                color="black", va="bottom", ha="left")
        ax = axes[row, 2]
        im_proj = ax.imshow(_upscale(proj[which], img_hw),
                            cmap=CMAP["diverge"],
                            vmin=-proj_vmax, vmax=proj_vmax)
        ax.text(0.03, 0.03, f"separation {sep[which]:.2f}",
                transform=ax.transAxes, fontsize=FONTSIZE_LEGEND,
                color="black", va="bottom", ha="left")
        axes[row, 0].set_ylabel("Input" if which == "in" else "Output",
                                fontsize=FONTSIZE_LABEL, fontweight="bold")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    cb = fig.colorbar(im_proj, ax=[axes[0, 2], axes[1, 2]],
                      location="bottom", shrink=0.85, pad=0.02)
    cb.set_label("Feature · class axis", fontsize=FONTSIZE_LEGEND)

    savefig(fig, OUT_DIR / "22_ham_edit_rgb.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
