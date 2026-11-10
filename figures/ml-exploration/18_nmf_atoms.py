"""Deep Feature Factorization atom atlas for the Hamburger decoder.

What this shows
---------------
The Hamburger module (Geng et al. 2021) gives the decoder global context by
factorising its coarse feature map through a small non-negative dictionary:
positions that activate the same dictionary atoms are pulled toward a common
reconstruction. This script asks: *what do those atoms encode, physically?*
For each atom we show its spatial coefficient map across several images —
if an atom consistently lights up on, say, bar breaking or swash or remnant
foam, the dictionary has learned an interpretable decomposition of the
surfzone scene.

Method — Deep Feature Factorization (DFF)
-----------------------------------------
Collins, Achanta & Süsstrunk (2018), "Deep Feature Factorization for
Concept Discovery", ECCV. arXiv:1806.05666.

DFF runs NMF on the (pixels × channels) activation matrix of a CNN layer;
the resulting factor maps localise semantic concepts without supervision.
Here the layer is the Hamburger *input* (the coarse decoder features), so
the offline factorisation directly mirrors what the module's own NMF does
online.

Why not just read out the module's own NMF bases?
-------------------------------------------------
Two reasons, both architectural (see NMF2D in the SegNeXt/Hamburger code):

1. The module *re-initialises its bases randomly on every forward pass*
   (online NMF, `_build_bases`), so "atom r" in one image has no relation
   to "atom r" in another — a cross-image atlas would be meaningless.
2. The module factorises S channel groups independently with separate
   coefficient maps per group.

Fitting one shared sklearn NMF dictionary on features pooled across many
images (the DFF approach) gives a single consistent set of atoms whose
coefficient maps are directly comparable across images. Features are
preprocessed exactly as the module does before its NMF: per-image scalar
standardisation, ReLU, + eps (non-negativity is required by NMF).

Atom selection — breaking lift, not raw activation
--------------------------------------------------
The factorisation uses *all* cells, but the task is a binary breaking /
background segmentation with only ~1 % breaking pixels — ranking atoms by
overall mean coefficient therefore surfaces background texture atoms.
Atoms are *selected* by their breaking **lift**:

    lift_r = mean coef_r over predicted-breaking cells
             ─────────────────────────────────────────
             mean coef_r over all cells

lift ≫ 1 → the atom is selectively part of the breaking reconstruction;
lift ≪ 1 → a background atom. The atlas shows the top atoms by lift plus,
for contrast, the strongest background atom (lowest lift).

Making the zonation legible
---------------------------
The atoms tend to correspond to physical zones (beach, swash, remnant
foam, active breaking, pre-breaking shoaling), so the figure is arranged
to read as a shore-normal transect:

- Atlas rows are **ordered by each atom's mean cross-shore position**
  (image row of its activation centroid), so scanning down the rows walks
  across the surfzone rather than jumping between zones.
- A **dominant-atom row** at the bottom colours every cell by the atom
  with the largest coefficient there. *Every* atom in the dictionary gets
  a colour (assigned in cross-shore order), so the map tiles the whole
  scene into zones with no gaps; the legend lists all atoms with their
  breaking lift. Only the high-lift (+1 background) atoms get their own
  coefficient-map rows above.
- A companion **profile figure** plots every atom's mean coefficient
  against cross-shore position (averaged over the time axis and the
  sample images): the zonation as curves, with each atom's peak marking
  its home zone. Atlas-row atoms are drawn bold, the rest faint. Atom
  colours match across all panels and figures.

Outputs
-------
  18_nmf_atom_atlas.{pdf,png}
      Rows: input images (top), one row per displayed atom (ordered by
      cross-shore position; label gives atom id + lift), and the
      dominant-atom map (bottom). Columns: N_SHOW_IMAGES sample images.
      Atom panels show the coefficient map (inferno, row-shared colour
      scale), NN-upscaled to the exact input-image size so every panel
      renders at the same size, with the predicted breaking boundary
      contoured in white.

  18_nmf_atom_profiles.{pdf,png}
      Mean coefficient vs cross-shore position (image row) per displayed
      atom, averaged over the sample images.

Run with the `dunex_pytorch` pyenv env:
    python 18_nmf_atoms.py
"""

import argparse

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

from exploration_common import (
    ML_DIR,
    PRODUCTION_THRESHOLD,
    add_runtime_arguments,
    import_figure,
    load_runtime,
    require_file,
)
from common.figure_style import FONTSIZE_LABEL, FONTSIZE_LEGEND, PAGE_W, apply_style, savefig

viz13 = import_figure("13_attention_viz")
viz16 = import_figure("16_hamburger_viz")
pooled = import_figure("17_pca_pooled")

OUT_DIR = ML_DIR

# ── Config ───────────────────────────────────────────────────────────────────
R_ATOMS = 12            # dictionary size for the offline DFF factorisation
N_FIT_IMAGES = 64       # images pooled into the dictionary fit
N_SHOW_ATOMS = 6        # atoms displayed (ranked by breaking lift)
N_SHOW_IMAGES = 4       # sample images (columns)
MAX_CELLS_FIT = 40_000  # cap on pooled cells fed to NMF (fixed seed)
NMF_EPS = 1e-6          # matches the module's eps for non-negativity
RNG_SEED = 0

# Sparsity of the factorisation. If an atom is "too broad" (one atom
# soaking up several visually distinct features), add L1 pressure on the
# coefficients (NMF_ALPHA_W > 0, e.g. 1e-4..1e-2) so each cell uses fewer
# atoms and catch-all atoms split — and/or raise R_ATOMS to give the
# dictionary room to separate the mixed features.
NMF_ALPHA_W = 0.0       # L1/L2 penalty on the coefficients W
NMF_ALPHA_H = 0.0       # penalty on the dictionary H
NMF_L1_RATIO = 1.0      # 1.0 = pure L1 (sparsity), 0.0 = pure L2

# Hand-assigned physical labels from inspecting a fit, shown in row
# labels / legend / profiles. Atom indices are only meaningful for one
# specific fit — refitting (different R_ATOMS, RNG_SEED, N_FIT_IMAGES, or
# sparsity) renumbers the atoms — so labels are keyed by the fit config
# and simply do not display for an unlabelled fit: re-inspect the atlas
# and add a new entry after any change.
# (R=12, seed 0, 24-image fit labels: Malcolm, 2026-07-03.)
LABELS_BY_FIT = {
    (12, 0, 24): {
        0: "?",
        1: "surfzone",
        2: "non-breaking",
        3: "swash",
        4: "surfzone",
        5: "non-breaking",
        6: "land",
        7: "breaking",
        8: "surfzone",
        9: "surfzone",
        10: "surfzone boundary",
        11: "non-breaking (broad)",
    },
}
ATOM_LABELS = LABELS_BY_FIT.get((R_ATOMS, RNG_SEED, N_FIT_IMAGES), {})


def _atom_name(r):
    return f"{r}: {ATOM_LABELS[r]}" if r in ATOM_LABELS else f"{r}"


def _select_fit_images(n):
    """n images evenly spaced through the sorted image directory (same
    scheme as 17_pca_pooled.py, but with a locally configurable count)."""
    all_imgs = sorted(pooled.IMAGE_DIR.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"no images in {pooled.IMAGE_DIR}")
    idx = np.unique(np.linspace(0, len(all_imgs) - 1, n).astype(int))
    return [all_imgs[i] for i in idx]


def ham_input_nonneg(model, device, path):
    """One hooked forward pass → non-negative Hamburger-input features.

    Returns (feat_nn (N, C), fg_cells (N,), (H', W'), fg_mask,
    img_display, img_hw). Preprocessing mirrors NMF2D.forward: scalar
    standardisation over the whole tensor, then ReLU + eps — NMF requires
    non-negative input, and matching the module means our atoms factorise
    the same matrix its own online NMF sees. fg_cells labels each Ham-grid
    cell breaking/background via the "any pixel" rule, for atom ranking.
    """
    img_tensor, img_raw, img_hw = viz13.preprocess_image(path)

    ham_hook = viz16.HamburgerIOHook()
    ham_hook.register(model)
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    ham_hook.remove()
    prob = viz13._match_size(prob, img_hw)   # output frame == input size
    fg_mask = prob > PRODUCTION_THRESHOLD

    x = ham_hook.inp[0]                        # (C, H', W')
    x = torch.relu((x - x.mean()) / (x.std() + NMF_EPS)) + NMF_EPS
    C, H, W = x.shape
    feat_nn = x.permute(1, 2, 0).reshape(-1, C).numpy()
    fg_cells = viz13._downsample_mask(fg_mask, (H, W)).flatten()
    return (feat_nn, fg_cells, (H, W), fg_mask,
            viz13.denormalize(img_raw), img_hw)


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser, image=True)
    parser.add_argument(
        "--fit-images", type=int, default=N_FIT_IMAGES,
        help=f"images used to fit the NMF dictionary (default: {N_FIT_IMAGES})",
    )
    parser.add_argument(
        "--show-images", type=int, default=N_SHOW_IMAGES,
        help=f"images shown in the atlas (default: {N_SHOW_IMAGES})",
    )
    parser.add_argument(
        "--max-fit-cells", type=int, default=MAX_CELLS_FIT,
        help=f"maximum pooled cells used for NMF (default: {MAX_CELLS_FIT})",
    )
    args = parser.parse_args()
    if min(args.fit_images, args.show_images, args.max_fit_cells) <= 0:
        parser.error("sampling counts must be positive")

    apply_style()
    rng = np.random.default_rng(RNG_SEED)
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir

    fit_images = _select_fit_images(args.fit_images)
    show_images = [require_file(args.image, "image")]
    for p in fit_images:
        if len(show_images) >= args.show_images:
            break
        if p not in show_images:
            show_images.append(p)

    # ── Pass 1: pool non-negative Ham-input cells across the fit images ──
    # fg labels are carried alongside every sampled cell so atoms can be
    # ranked by their association with the breaking class afterwards.
    print(f"Pooling Ham-input features from {len(fit_images)} images...")
    chunks, flag_chunks = [], []
    for k, path in enumerate(fit_images):
        print(f"  [{k + 1}/{len(fit_images)}] {path.name}")
        feat_nn, fg_cells, _hw, _fg, _img, _ihw = ham_input_nonneg(
            runtime.model, runtime.device, path)
        chunks.append(feat_nn)
        flag_chunks.append(fg_cells)
    pool = np.concatenate(chunks, axis=0)
    flags = np.concatenate(flag_chunks, axis=0)
    if pool.shape[0] > args.max_fit_cells:
        pick = rng.choice(pool.shape[0], args.max_fit_cells, replace=False)
        pool, flags = pool[pick], flags[pick]
    print(f"  pooled matrix: {pool.shape[0]:,} cells × {pool.shape[1]} "
          f"channels ({100 * flags.mean():.1f}% breaking cells)")

    # ── Fit the shared DFF dictionary: pool ≈ W · H, W = coefficients ──
    # nndsvda init is deterministic and suited to dense data (Boutsidis &
    # Gallopoulos 2008); transform() later solves for W with H fixed, so
    # atom identity is consistent across images.
    print(f"Fitting NMF dictionary (R = {R_ATOMS}, "
          f"alpha_W = {NMF_ALPHA_W})...")
    nmf = NMF(n_components=R_ATOMS, init="nndsvda", max_iter=400, tol=1e-4,
              alpha_W=NMF_ALPHA_W, alpha_H=NMF_ALPHA_H,
              l1_ratio=NMF_L1_RATIO, random_state=RNG_SEED)
    W_pool = nmf.fit_transform(pool)
    print(f"  reconstruction error: {nmf.reconstruction_err_:.3f}"
          f"  (n_iter {nmf.n_iter_})")

    # Select atoms by breaking lift (see module docstring) — raw mean
    # activation would surface background atoms since ~99% of cells are
    # background in this binary task.
    overall_mean = W_pool.mean(axis=0) + 1e-12
    fg_mean = W_pool[flags].mean(axis=0) if flags.any() else overall_mean
    lift = fg_mean / overall_mean
    order = np.argsort(lift)[::-1]
    shown = list(order[:N_SHOW_ATOMS]) + [int(order[-1])]  # + strongest BG atom
    for r in order:
        print(f"  atom {r}: lift {lift[r]:.2f}  (fg {fg_mean[r]:.4f} / "
              f"all {overall_mean[r]:.4f})")

    # ── Pass 2: coefficient maps for the sample images ──
    samples = []
    for path in show_images:
        feat_nn, _fgc, (H, W), fg_mask, img_display, img_hw = ham_input_nonneg(
            runtime.model, runtime.device, path)
        coef = nmf.transform(feat_nn)          # (N, R) with the fixed dictionary
        samples.append({
            "tag": pooled._tag(path),
            "coef": coef.reshape(H, W, -1),
            "fg_mask": fg_mask,
            "img_display": img_display,
            "img_hw": img_hw,
        })

    # ── Cross-shore ordering + per-atom colours (ALL atoms) ──
    # Centroid row of each atom's activation (mean over samples): sorting
    # by it makes the rows read as a shore-normal transect. Rows are image
    # rows; the physical beach/offshore direction follows the timestack
    # orientation of the source imagery. Every atom gets a colour in
    # cross-shore order so the dominant-atom map tiles the scene with no
    # grey gaps.
    def _profile(r):
        """Mean coefficient of atom r vs image row, averaged over samples
        (time axis collapsed)."""
        profs = [s["coef"][:, :, r].mean(axis=1) for s in samples]
        n = min(len(p) for p in profs)
        return np.mean([p[:n] for p in profs], axis=0)

    def _centroid(r):
        p = _profile(r)
        return float((np.arange(len(p)) * p).sum() / (p.sum() + 1e-12))

    order_all = sorted(range(R_ATOMS), key=_centroid)
    atom_color = {r: plt.cm.tab20(i % 20) for i, r in enumerate(order_all)}
    shown = sorted(set(shown), key=_centroid)
    row_names = [f"{_atom_name(r)}\nlift {lift[r]:.1f}" for r in shown]

    # ── Atlas: rows = input + atoms (shore-ordered) + dominant map ──
    # Every panel (input and coefficient maps alike) is rendered at the
    # exact input-image size so matplotlib draws them all identically.
    n_rows = 1 + len(shown) + 1
    n_cols = len(samples)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PAGE_W, PAGE_W * 0.24 * n_rows),
                             squeeze=False, layout="constrained")

    for col, s in enumerate(samples):
        axes[0, col].imshow(s["img_display"])
        axes[0, col].set_title(s["tag"], fontsize=FONTSIZE_LEGEND)
    axes[0, 0].set_ylabel("Input", fontsize=FONTSIZE_LABEL, fontweight="bold")

    for row, r in enumerate(shown, start=1):
        # Row-shared colour scale so an atom's intensity is comparable
        # across images (99th pct guards against single-cell spikes)
        vmax = max(np.percentile(s["coef"][:, :, r], 99) for s in samples)
        vmax = max(vmax, 1e-6)
        for col, s in enumerate(samples):
            ax = axes[row, col]
            H_img, W_img = s["img_hw"]
            cmap_up = cv2.resize(s["coef"][:, :, r], (W_img, H_img),
                                 interpolation=cv2.INTER_NEAREST)
            ax.imshow(cmap_up, cmap="inferno", vmin=0, vmax=vmax,
                      interpolation="nearest")
            # White contour = model's predicted breaking boundary
            ax.contour(s["fg_mask"].astype(float), levels=[0.5],
                       colors="white", linewidths=0.4, alpha=0.8)
        axes[row, 0].set_ylabel(row_names[row - 1], fontsize=FONTSIZE_LEGEND,
                                fontweight="bold",
                                color=atom_color[r])

    # Bottom row: dominant atom per cell — the scene segmented into atom
    # zones. All R atoms are coloured (cross-shore-ordered palette), so
    # the map tiles the frame completely.
    dom_row = 1 + len(shown)
    for col, s in enumerate(samples):
        dom = s["coef"].argmax(axis=2)                      # (H', W')
        rgb = np.zeros(dom.shape + (3,), dtype=np.float32)
        for r in range(R_ATOMS):
            rgb[dom == r] = atom_color[r][:3]
        H_img, W_img = s["img_hw"]
        up = cv2.resize(rgb, (W_img, H_img), interpolation=cv2.INTER_NEAREST)
        ax = axes[dom_row, col]
        ax.imshow(up)
        ax.contour(s["fg_mask"].astype(float), levels=[0.5],
                   colors="white", linewidths=0.4, alpha=0.8)
    axes[dom_row, 0].set_ylabel("Dominant\natom", fontsize=FONTSIZE_LEGEND,
                                fontweight="bold")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    # Legend: every atom, in cross-shore order, with its breaking lift
    handles = [mpatches.Patch(color=atom_color[r],
                              label=f"{_atom_name(r)} ({lift[r]:.1f})")
               for r in order_all]
    try:
        # "outside" placement needs constrained layout + matplotlib ≥ 3.7
        fig.legend(handles=handles, loc="outside lower center",
                   ncols=min(R_ATOMS, 6), fontsize=FONTSIZE_LEGEND,
                   title="atom (breaking lift)",
                   title_fontsize=FONTSIZE_LEGEND)
    except (TypeError, ValueError):
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.03),
                   ncols=min(R_ATOMS, 6), fontsize=FONTSIZE_LEGEND,
                   title="atom (breaking lift)",
                   title_fontsize=FONTSIZE_LEGEND)

    fig.suptitle(
        f"Hamburger-input concept atoms (R={R_ATOMS}, "
        f"{len(fit_images)} fit images)",
        fontsize=FONTSIZE_LABEL, fontweight="bold",
    )
    savefig(fig, OUT_DIR / "18_nmf_atom_atlas.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)

    # ── Profile figure: the zonation as curves (all atoms) ──
    fig, ax = plt.subplots(figsize=(PAGE_W * 0.7, PAGE_W * 0.45),
                           layout="constrained")
    for r in order_all:
        p = _profile(r)
        bold = r in shown
        ax.plot(np.arange(len(p)), p, color=atom_color[r],
                linewidth=1.4 if bold else 0.7,
                alpha=1.0 if bold else 0.45,
                label=f"{_atom_name(r)} (lift {lift[r]:.1f})")
    ax.set_xlabel("Cross-shore position (image row)", fontsize=FONTSIZE_LABEL)
    ax.set_ylabel("Mean coefficient", fontsize=FONTSIZE_LABEL)
    ax.legend(fontsize=FONTSIZE_LEGEND, ncols=2)
    ax.set_title("NMF atom zonation — mean coefficient vs cross-shore "
                 f"position ({len(samples)} samples; bold = atlas atoms)",
                 fontsize=FONTSIZE_LABEL, fontweight="bold")
    savefig(fig, OUT_DIR / "18_nmf_atom_profiles.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
