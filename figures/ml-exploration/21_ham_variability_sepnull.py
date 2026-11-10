"""Variant of 21_ham_variability_pooled.py that swaps the third panel.

The pooled figure (21_ham_variability_pooled.py) reports the class
separation with the *isotropic* RMS_w statistic ‖μ_brk − μ_bg‖/RMS_w,
which rises from input to output. That statistic treats the within-class
scatter as a ball (it uses only the trace of S_w), so a centroid gap
opening up along a low-variance direction counts the same as one along a
high-variance direction. When the proper direction-aware statistic — the
bias-corrected Mahalanobis / two-class Fisher separation, which measures
each direction against the classes' scatter along *that* direction — is
used instead, the increase disappears: the module does not make the
classes more separable in the Fisher sense.

This figure keeps the first two panels of the pooled figure verbatim and
replaces panel (c) with that de-biased Fisher separation, so the two
separation claims sit next to the geometry they rest on:

(a) Scree curves (identical to pooled panel a).
(b) Relative intraclass variance boxes (identical to pooled panel b).
(c) Bias-corrected Mahalanobis class separation, input vs output
    (identical to the Fisher-figure panel b): the plug-in
    Δμᵀ S_w⁻¹ Δμ de-biased with the Anderson/Lachenbruch estimator on
    the raw pooled covariance, √max(δ̂², 0) plotted. Output box NOT above
    the input box ⇒ no Fisher-sense separability gain — contrast with
    the RMS_w panel (c) in 21_ham_variability_pooled.py.

All statistics are computed by the machinery in 21_ham_variability_pooled
(imported unchanged); only the figure assembly differs. See that file's
docstring for the full derivation of every statistic.

Run with the `dunex_pytorch` pyenv env:
    python 21_ham_variability_sepnull.py [--fast]
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from exploration_common import (
    ML_DIR,
    add_runtime_arguments,
    import_figure,
    load_runtime,
)
from common.figure_style import IBM, PAGE_W, apply_style, panel_label, savefig, styled_legend

# Reuse the pooled script's model loading, per-image helpers and the
# box/scree drawing helpers — this variant only re-lays-out panels it
# already produces, and uses a trimmed per-image stats pass (below).
pooled = import_figure("21_ham_variability_pooled")
ham16 = pooled.ham16

OUT_DIR = ML_DIR
FG_COL, BG_COL = pooled.FG_COL, pooled.BG_COL
BOX_LW = pooled.BOX_LW
N_SCREE = pooled.N_SCREE
_box = pooled._box


def image_stats(model, device, path):
    """Trimmed per-image stats: only the quantities this figure's three
    panels need — scree (a), relative intraclass variance (b) and the
    bias-corrected Mahalanobis separation (c). Skips the pooled script's
    expensive extras (the 64-step Mahalanobis-vs-components sweep and the
    Ledoit-Wolf whitened-variance fit), which this layout never plots.
    Uses the shared bottom-centred-crop hooked forward pass; returns None
    if either class has too few Ham-grid cells."""
    res = pooled.hooked_crop_features(model, device, path)
    if res is None:
        return None
    fg, feat_in, feat_out = res
    C = feat_in.shape[1]

    stats = {"fg_frac": float(fg.mean()), "scree": {}, "var": {},
             "sep": {}, "msep": {}, "msep2": {}}
    pw = {"in": float(np.mean(
              ham16.within_class_deviation(feat_in, fg) ** 2)),
          "out": float(np.mean(
              ham16.within_class_deviation(feat_out, fg) ** 2))}
    for cls, m in (("Brk", fg), ("Bg", ~fg)):
        for which, feat in (("in", feat_in), ("out", feat_out)):
            stats["scree"][(cls, which)] = pooled._scree(feat, m)
            stats["var"][(cls, which)] = (pooled._trace_var(feat, m)
                                          / (pw[which] + 1e-12))
    n1, n2 = int(fg.sum()), int((~fg).sum())
    for which, feat in (("in", feat_in), ("out", feat_out)):
        stats["sep"][which] = pooled._separation(feat, fg)
        # de-biased squared Mahalanobis on the raw pooled covariance
        mu_b, mu_g = feat[fg].mean(axis=0), feat[~fg].mean(axis=0)
        Xc = np.vstack([feat[fg] - mu_b, feat[~fg] - mu_g])
        S_w_raw = Xc.T @ Xc / (n1 + n2 - 2)
        msep2 = pooled._maha2_unbiased(mu_b - mu_g, S_w_raw, n1, n2, C)
        stats["msep2"][which] = msep2
        stats["msep"][which] = (float(np.sqrt(msep2))
                                if np.isfinite(msep2) and msep2 > 0 else 0.0)
    return stats


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser)
    parser.add_argument("--fast", action="store_true",
                        help=f"use only {pooled.N_FAST} images (quick "
                             "styling iterations)")
    args = parser.parse_args()

    apply_style()
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir

    paths = pooled._select_images(pooled.N_FAST if args.fast
                                  else pooled.N_IMAGES)
    print(f"Selected {len(paths)} images from {pooled.IMAGE_DIR}")

    all_stats = []
    for i, path in enumerate(paths):
        s = image_stats(runtime.model, runtime.device, path)
        if s is None:
            print(f"  [{i + 1}/{len(paths)}] {path.name}: skipped "
                  f"(a class has <= {pooled.N_SCREE} cells)")
            continue
        print(f"  [{i + 1}/{len(paths)}] {path.name}: "
              f"{100 * s['fg_frac']:.1f}% breaking, "
              f"sep {s['sep']['in']:.2f} → {s['sep']['out']:.2f}, "
              f"maha {s['msep']['in']:.2f} → {s['msep']['out']:.2f}")
        all_stats.append(s)
    n = len(all_stats)
    if n < 2:
        raise RuntimeError(f"only {n} usable images — need at least 2")
    print(f"Usable images: {n}/{len(paths)}")

    var_vals = {}
    for cls in ("Brk", "Bg"):
        vin = np.array([s["var"][(cls, "in")] for s in all_stats])
        vout = np.array([s["var"][(cls, "out")] for s in all_stats])
        var_vals[cls] = (vin, vout)
    msep_in = np.array([s["msep"]["in"] for s in all_stats])
    msep_out = np.array([s["msep"]["out"] for s in all_stats])
    sep_in = np.array([s["sep"]["in"] for s in all_stats])
    sep_out = np.array([s["sep"]["out"] for s in all_stats])
    print(f"  RMS_w separation median: {np.median(sep_in):.2f} → "
          f"{np.median(sep_out):.2f}")
    print(f"  bias-corrected Mahalanobis separation median: "
          f"{np.median(msep_in):.2f} → {np.median(msep_out):.2f}")

    # ── Figure: 1 × 3 — pooled (a), pooled (b), Fisher separation (c) ──
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(PAGE_W, PAGE_W * 0.32), layout="constrained")

    # (a) scree: mean over images ± 1 std (identical to pooled panel a)
    ax = ax_a
    stagger = 0
    for cls, col in (("Brk", FG_COL), ("Bg", BG_COL)):
        for which, ls in (("in", "-"), ("out", "--")):
            curves = np.stack([s["scree"][(cls, which)] for s in all_stats])
            mean, std = curves.mean(axis=0), curves.std(axis=0)
            x = np.arange(1, curves.shape[1] + 1)
            ax.plot(x, mean, color=col, linestyle=ls,
                    marker="o", markersize=2.6, markevery=(stagger, 4),
                    markerfacecolor=(col if which == "in" else "white"),
                    markeredgecolor=col, markeredgewidth=0.6)
            ax.fill_between(x, mean - std, mean + std, color=col,
                            alpha=0.15, linewidth=0)
            stagger += 1
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Components")
    ax.set_ylabel("Cumulative explained\nvariance")
    handles = [Line2D([], [], color=FG_COL, label="Breaking"),
               Line2D([], [], color=BG_COL, label="Background")]
    styled_legend(ax, handles=handles, loc="lower right")
    panel_label(ax, "a")

    # (b) relative intraclass variance before/after (identical to pooled b)
    ax = ax_b
    for gi, cls in enumerate(("Brk", "Bg")):
        col = FG_COL if cls == "Brk" else BG_COL
        vin, vout = var_vals[cls]
        _box(ax, gi * 2.0 + 0.6, vin, col, filled=True)
        _box(ax, gi * 2.0 + 1.4, vout, col, filled=False)
    ax.set_xticks([1.0, 3.0], ["Breaking", "Background"])
    ax.set_xlim(0, 4.0)
    top = max(v.max() for pair in var_vals.values() for v in pair)
    ax.set_ylim(0, top * 1.25)
    handles = [mpatches.Patch(facecolor=mcolors.to_rgba("0.2", 0.35),
                              edgecolor="0.2", linewidth=BOX_LW,
                              label="Input"),
               mpatches.Patch(facecolor="none", edgecolor="0.2",
                              linewidth=BOX_LW, linestyle="--",
                              label="Output")]
    styled_legend(ax, handles=handles, loc="upper right", ncol=1)
    ax.set_ylabel("Relative intraclass variance")
    panel_label(ax, "b")

    # (c) bias-corrected Mahalanobis (Fisher) separation before/after —
    # the direction-aware statistic; out box NOT above in box = no
    # Fisher-sense separability gain (contrast with pooled panel c, which
    # uses the isotropic RMS_w statistic and does rise).
    ax = ax_c
    _box(ax, 0.6, msep_in, IBM[1], filled=True)
    _box(ax, 1.4, msep_out, IBM[1], filled=False)
    ax.set_xticks([0.6, 1.4], ["Input", "Output"])
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)
    # LaTeX for the draft caption:
    #   $\sqrt{(\mu_{brk}-\mu_{bg})^{\top} S_w^{-1} (\mu_{brk}-\mu_{bg})}$
    # (Mahalanobis distance between the class centroids — the square root
    # of the de-biased two-class Fisher criterion; unlike the isotropic
    # RMS_w separation it does not increase from input to output)
    ax.set_ylabel("Mahalanobis class separation")
    panel_label(ax, "c")

    savefig(fig, OUT_DIR / "21_ham_variability_sepnull.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
