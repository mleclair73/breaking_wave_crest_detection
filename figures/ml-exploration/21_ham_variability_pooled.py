"""Appendix figure — Hamburger homogenisation statistics over many images.

Multi-image companion to 16_hamburger_viz.py: the single-image figure
shows the effect for one timestack; here the same three statistics are
computed independently on N_IMAGES images so the claim — the Hamburger
NMF (Geng et al. 2021, "Is Attention Better Than Matrix Decomposition?",
ICLR, arXiv:2109.04553) reshapes the class geometry — rests on a
distribution, not an anecdote. Observed signature on this checkpoint:
class separation rises (c) while the breaking class *loses* intraclass
variance relative to the pooled within-class level (b) — the module
compacts the minority-class representation and moves it away from
background (homogenisation together with increased separation). Treat
the sign of the (b) shift as checkpoint-dependent
and read it off the console `relative-variance ratio out/in` line rather
than trusting this comment: ratio < 1 is the homogenisation case above,
> 1 the enrichment one.

Statistics are computed *per image* and then aggregated (not by pooling
cells across images): pooling would mix inter-image variance (lighting,
surf state, camera exposure) into the within-class variance and inflate
it for reasons unrelated to the module. Per-image curves/points keep
each image its own baseline; the spread across images is the robustness
evidence.

Panels (tags and styling use the shared figure-style module):

(a) Scree curves: cumulative PCA explained variance vs number of
    components, fit separately on breaking and background cells, Ham
    input (solid, filled markers) vs output (dashed, open markers).
    The breaking-input and background-output means nearly coincide, so
    the curves carry markers with staggered spacing — both curves stay
    visible where the lines lie on top of each other. Lines are the
    mean over images, bands ±1 std. Structure packed into fewer
    components ⇒ the NMF built a lower-dimensional representation of
    that class.
(b) Box plots over images of the *relative* intraclass variance:
    ⟨‖f − μ_class‖²⟩ (mean squared distance of a cell's features to its
    own class centroid = trace of the class channel covariance) divided
    by RMS_w², the pooled within-class variance of the same side — the
    unit the separation statistic in (c) already uses. Two
    normalisation traps this avoids: raw variances inflate with the
    residual module's larger output norms (global scale), and dividing
    by *total* variance deflates both classes' shares mechanically as
    the between-class gap grows (total = within + between), which would
    fake a homogenisation signal whenever separation rises. RMS_w²
    cancels the scale and excludes the centroid gap, so a breaking
    box *falling* from input to output means breaking cells genuinely
    became more internally uniform relative to the pooled level.
    Background dominates the pooled level (~99 % of cells), so its box
    sits near 1 by construction — the built-in control — and a small
    background *rise* is the expected mirror of a breaking fall (RMS_w² ≈
    background variance, so the background value ≈
    1/(w_bg + w_brk · var_brk/var_bg) rises toward 1/w_bg as the
    breaking/background variance contrast shrinks), not evidence of
    background expansion.
    Box = quartiles, bar = median, whiskers = 1.5 × IQR, dots = outlier
    images only; per-image out/in ratio medians are printed to the
    console.
(c) Box plots over images of the class separation
    ‖μ_brk − μ_bg‖ / RMS_w, before vs after: the distance between the
    breaking and background centroids in units of the pooled
    within-class RMS deviation (a Fisher-style discriminability ratio).
    Output box above the input box ⇒ the classes are more separable
    after the module. Separation is gap-per-spread, so it can rise even
    while the relative breaking variance in (b) rises — centroids
    moving apart while the classes contract. That pairing (separation up,
    relative breaking variance down) is the homogenisation signature;
    separation up with relative breaking variance *up* — centroids
    outrunning a widening class — would have been the enrichment one.

Before/after is encoded the same way in every panel: module input =
solid line / filled box, module output = dashed line / unfilled box
(legend in panel b); colour = what is measured (breaking / background /
separation).

Direction-aware (Fisher) companion figure
-----------------------------------------
The RMS_w statistics treat the within-class scatter as an isotropic
ball — they use only its trace, so a centroid gap along a low-variance
direction counts the same as one along a high-variance direction. The
second output repeats (b) and (c) with the full within-class covariance
S_w pooled over both classes. Panel (a)'s whitening uses Ledoit-Wolf
shrinkage (Ledoit & Wolf 2004, J. Multivar. Anal.) for a stable inverse;
the centroid-separation panels (b, c) use the raw pooled covariance —
(b) with an explicit finite-sample bias correction, (c) as a scale-free
share — because the de-biasing and monotonicity arguments they rest on
need the unshrunk Wishart estimator:

(a) Whitened intraclass variance tr(S_w⁻¹ Σ_class) / C: the class's
    mean variance per direction after whitening by S_w. Direction-aware
    analogue of RMS_class²/RMS_w²; scale-invariant (S_w scales with the
    features); background sits at ≈ 1 by construction (dotted line).
(b) Bias-corrected Mahalanobis class separation: the centroid distance
    with each direction measured against the classes' scatter along
    *that* direction — √ of the two-class Fisher criterion δ² (the RMS_w
    version in the main figure is its isotropic simplification, a
    multivariate Cohen's d). The plug-in Δμᵀ S_w⁻¹ Δμ is inflated by
    O(C/n) finite-sample bias — severe here, C ≈ hundreds against a
    minority class of only tens of cells — so it is de-biased with the
    Anderson/Lachenbruch estimator δ̂² = (N−C−3)/(N−2)·D² − C(1/n_brk +
    1/n_bg) on the raw pooled covariance (N = cell count), and √max(δ̂², 0)
    is plotted. Values sit well below the naive plug-in; images whose
    δ̂² ≤ 0 (separation indistinguishable from zero) floor at 0.
(c) Cumulative *share* of the squared Mahalanobis class separation
    carried by the top-k PCA directions, per side: D²_k / D²_K over the
    leading K_MAX PCA directions (fit unsupervised on all cells of each
    side; raw pooled S_w,k, well-conditioned and monotone in k since
    cells ≫ K_MAX), median and IQR over images. The discriminative twin
    of the scree panel — dividing by the top-K total removes the overall
    separation level (which (b) shows is ~unchanged, even slightly lower
    once de-biased) and isolates *concentration*. An output curve (dashed)
    above the input (solid) at small k means the module packs the same
    class separability into fewer leading directions, the separation
    analogue of the compressed scree spectrum. Showing the share, not the
    absolute per-k separation, keeps the output's marginally-lower
    full-subspace level from reading as a loss of concentration.

Caveats: the whitened intraclass variance (a) still carries finite-sample
bias with C comparable to the cell count, even under shrinkage — its
absolute values are approximate, but the estimator is identical for input
and output, so the before/after comparison is fair. The separation panels
address this directly: (b) via the explicit correction above, (c) by
reporting a scale-free share. A *uniform* drop of both classes' whitened
variance in (a) is a global spectral property, not class-targeted
geometry: the cell-weighted mean of the two values equals
tr(S_w⁻¹ S_w^raw)/C — S_w^raw the pooled within-class covariance before
shrinkage — which falls when a side's spectrum concentrates so that more
directions sit below the shrinkage floor — the same compression the scree
panel shows. The console prints the per-side Ledoit-Wolf shrinkage for
this diagnosis.

Image selection & inclusion rule
--------------------------------
N_IMAGES images evenly spaced through the sorted image directory (same
scheme as 17_pca_pooled.py / 18_nmf_atoms.py); `--fast` drops to N_FAST
images for quick styling iterations. An image only contributes if
*both* classes have more than N_SCREE cells on the Ham grid — that
guarantees every scree curve has the full N_SCREE components and that
the class statistics aren't computed from a handful of cells. Skipped
images are listed on the console.

The breaking/background labels come from the model's own prediction
(P > 0.48) mapped to the Ham grid with the "any pixel" rule, as everywhere
else in these appendix scripts.

Outputs
-------
  21_ham_variability_pooled.{pdf,png}
  21_ham_variability_fisher.{pdf,png}

Task-level ablation (`--bypass-iou`)
------------------------------------
The simplest statement of what the module does, with no representation
statistics at all: bypass the Hamburger (output → ReLU(input), the
19_erf.py ablation) and compare the *predictions* of the bypassed model
against the full model on the same images — breaking-mask IoU,
predicted-area ratio, and mean |ΔP(breaking)| per pixel. Console only,
no figures. High agreement ⇒ the module barely moves the decision; low
IoU ⇒ it is load-bearing for the prediction.

Run with the `dunex_pytorch` pyenv env:
    python 21_ham_variability_pooled.py [--fast] [--bypass-iou]
"""

import argparse

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from exploration_common import (
    ML_DIR,
    PRODUCTION_THRESHOLD,
    add_runtime_arguments,
    import_figure,
    load_runtime,
)
from common.figure_style import IBM, PAGE_W, apply_style, panel_label, savefig, styled_legend

# Single-image script provides preprocessing, the Hamburger I/O hook, mask
# downsampling, and the deviation helper.
ham16 = import_figure("16_hamburger_viz")

OUT_DIR = ML_DIR
IMAGE_DIR = ham16.IMAGE_DIR

# ── Config ───────────────────────────────────────────────────────────────────
N_IMAGES = 128                      # images sampled from the directory
N_FAST = 4                          # --fast: quick styling iterations
N_SCREE = ham16.N_SCREE             # scree components; a class needs
                                    # > N_SCREE cells to be included
FG_COL, BG_COL = ham16.FG_COL, ham16.BG_COL

# The SegNeXt checkpoint was trained on 224×224 crops (config image_size,
# and production inference tiles the frame with 224 patches). Feeding the
# full 500×512 frame is both out-of-distribution and several× slower, so a
# fixed 224 crop is taken before the forward pass — matching the training
# input. The crop is bottom-centred: horizontally centred and flush with
# the bottom edge, where the surf zone / breaking sits in these cross-shore
# timestacks, so a single crop reliably contains both classes.
CROP = 224


def preprocess_crop(path):
    """Bottom-centred CROP×CROP crop of the image (horizontally centred,
    flush with the bottom edge), normalised as in ham16.preprocess_image.
    Returns (tensor, (H, W))."""
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - ham16.IMAGENET_MEAN) / ham16.IMAGENET_STD
    H, W = img.shape[:2]
    top = max(0, H - CROP)              # flush with the bottom edge
    left = max(0, (W - CROP) // 2)      # horizontally centred
    crop = img[top:top + CROP, left:left + CROP]
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, crop.shape[:2]


def hooked_crop_features(model, device, path):
    """Bottom-centred crop → one hooked forward pass. Returns
    (fg, feat_in, feat_out), or None if either class has too few cells on
    the Ham grid (<= N_SCREE)."""
    img_tensor, crop_hw = preprocess_crop(path)
    hook = ham16.HamburgerIOHook()
    hook.register(model)
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    hook.remove()
    if prob.shape != crop_hw:
        prob = cv2.resize(prob, (crop_hw[1], crop_hw[0]),
                          interpolation=cv2.INTER_NEAREST)
    fg_mask = prob > ham16.PRODUCTION_THRESHOLD

    C, H, W = hook.inp.shape[1], hook.inp.shape[2], hook.inp.shape[3]
    feat_in = hook.inp[0].permute(1, 2, 0).reshape(-1, C).numpy()
    feat_out = hook.out[0].permute(1, 2, 0).reshape(-1, C).numpy()
    fg = ham16._downsample_mask(fg_mask, (H, W)).flatten()
    if min(fg.sum(), (~fg).sum()) <= N_SCREE:
        return None
    return fg, feat_in, feat_out


def _select_images(n):
    """n images evenly spaced through the sorted directory listing
    (deterministic; same scheme as 17_pca_pooled.py)."""
    all_imgs = sorted(IMAGE_DIR.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"no images in {IMAGE_DIR}")
    idx = np.unique(np.linspace(0, len(all_imgs) - 1, n).astype(int))
    return [all_imgs[i] for i in idx]


def _scree(feat, m):
    """Cumulative explained-variance curve on the cells in mask m.
    Inclusion rule guarantees m.sum() > N_SCREE, so curves are full
    length and can be averaged element-wise across images."""
    n_pc = int(min(N_SCREE, feat.shape[1], m.sum() - 1))
    return np.cumsum(PCA(n_components=n_pc).fit(feat[m])
                     .explained_variance_ratio_)


def _trace_var(feat, m):
    """Within-class variance: trace of the channel covariance (= total
    feature variance of the class, summed over channels)."""
    return float(feat[m].var(axis=0).sum())


def _separation(feat, fg):
    """‖μ_fg − μ_bg‖ / pooled within-class RMS deviation."""
    mu_gap = np.linalg.norm(feat[fg].mean(axis=0) - feat[~fg].mean(axis=0))
    pooled = np.sqrt(np.mean(
        ham16.within_class_deviation(feat, fg) ** 2))
    return mu_gap / (pooled + 1e-12)


def _maha2_unbiased(dmu, cov, n1, n2, p):
    """Anderson/Lachenbruch bias-corrected *squared* Mahalanobis distance
    between two class centroids.

    The plug-in D² = Δμᵀ Ŝ⁻¹ Δμ from p-dimensional features and a pooled
    sample covariance Ŝ (N−2 dof, N = n1+n2) is inflated by two
    finite-sample effects: the centroid gap carries an offset,
    E[Δμᵀ Σ⁻¹ Δμ] = δ² + p(1/n1 + 1/n2), and the inverse-Wishart Ŝ⁻¹
    scales the estimate by (N−2)/(N−p−3). Undoing both gives an
    (approximately) unbiased estimate of the population δ²:

        δ̂² = (N−p−3)/(N−2) · D² − p(1/n1 + 1/n2).

    Here p ≈ hundreds while the minority class has only tens of cells, so
    the offset term dominates and the naive √D² badly overstates the
    separation. `cov` must be the *raw* pooled covariance (no shrinkage),
    or the Wishart factor does not apply. Returns NaN when N ≤ p + 3 (the
    inverse-Wishart mean is undefined); δ̂² may be ≤ 0 — a legitimate
    "indistinguishable from zero" outcome, left unclipped here.

    On a small crop the Hamburger output occasionally has a few dead
    (zero-variance) ReLU channels, which make `cov` exactly singular. Such
    a channel is constant across all cells, so both class means share that
    constant and its centroid gap is zero — it is a null direction with no
    separation to measure. Those channels are dropped before the inverse
    and the effective dimension p is reduced to match, keeping the
    bias-correction offset consistent with the space actually inverted.
    """
    keep = np.diag(cov) > 1e-12
    if not keep.all():
        cov = cov[np.ix_(keep, keep)]
        dmu = dmu[keep]
        p = int(keep.sum())
    N = n1 + n2
    if N - p - 3 <= 0:
        return np.nan
    d2 = float(dmu @ np.linalg.solve(cov, dmu))
    return (N - p - 3) / (N - 2) * d2 - p * (1.0 / n1 + 1.0 / n2)


def _whitened_stats(feat, fg):
    """Direction-aware (proper Fisher) statistics for one side.

    The RMS_w statistics treat the within-class scatter as an isotropic
    ball (they only use its trace). Here the full within-class
    covariance S_w enters: cells are whitened by S_w, so a direction
    only counts as "far" relative to how much the classes scatter along
    *that* direction. S_w is estimated with Ledoit-Wolf shrinkage
    (Ledoit & Wolf 2004, J. Multivar. Anal.) — the sample covariance is
    ill-conditioned with C channels comparable to the cell count. Both
    sides get the same estimator, so the in/out comparison is fair.

    Returns
    -------
    wvar : dict  {"Brk": ..., "Bg": ...}
        tr(S_w⁻¹ Σ_class) / C — the class's mean variance per direction
        in whitened coordinates (direction-aware analogue of
        RMS_class²/RMS_w²; scale-invariant since S_w scales with the
        features; background ≈ 1 by construction, the control).
    msep : float
        √max(δ̂², 0), the bias-corrected Mahalanobis distance between the
        class centroids (√ of the de-biased two-class Fisher criterion;
        direction-aware analogue of ‖Δμ‖/RMS_w). Built on the raw pooled
        covariance so the Anderson/Lachenbruch correction applies.
    msep2 : float
        the signed de-biased δ̂² (≤ 0 ⇒ separation indistinguishable from
        zero), kept for the console diagnostic.
    """
    mu = {"Brk": feat[fg].mean(axis=0), "Bg": feat[~fg].mean(axis=0)}
    Xc = np.vstack([feat[fg] - mu["Brk"], feat[~fg] - mu["Bg"]])
    lw = LedoitWolf().fit(Xc)
    S_w = lw.covariance_
    C = feat.shape[1]
    wvar = {}
    for cls, m in (("Brk", fg), ("Bg", ~fg)):
        Z = feat[m] - mu[cls]                     # (N_c, C), class-centred
        sol = np.linalg.solve(S_w, Z.T)           # S_w⁻¹ Zᵀ  → (C, N_c)
        wvar[cls] = float(np.einsum("nc,cn->n", Z, sol).mean() / C)
    # Centroid separation: de-bias the squared Mahalanobis on the RAW
    # (unshrunk) pooled covariance so the Wishart correction is valid; the
    # shrunk S_w above is kept only for the whitened-variance trace, where
    # invertibility (not an unbiased inverse) is what matters.
    n1, n2 = int(fg.sum()), int((~fg).sum())
    S_w_raw = Xc.T @ Xc / (n1 + n2 - 2)
    dmu = mu["Brk"] - mu["Bg"]
    msep2 = _maha2_unbiased(dmu, S_w_raw, n1, n2, C)
    msep = float(np.sqrt(msep2)) if np.isfinite(msep2) and msep2 > 0 else 0.0
    # Shrinkage intensity is diagnostic: the cell-weighted mean of the
    # two wvar values equals tr(S_w⁻¹ S_w^raw)/C — S_w^raw the pooled
    # within-class covariance before shrinkage — a global spectral
    # property — it falls when shrinkage differs between sides OR when
    # a side's spectrum concentrates so that more directions drop below
    # the shrinkage floor (the compression the scree panel shows). A
    # uniform drop of BOTH classes' whitened variance is therefore not
    # class-targeted geometry.
    return wvar, msep, msep2, float(lw.shrinkage_)


K_MAX = 64   # depth of the separation-vs-components sweep (2 × N_SCREE)


def _sep_vs_components(feat, fg):
    """Squared Mahalanobis class separation D²_k restricted to the top-k
    PCA directions, k = 1..K_MAX → curve of length K_MAX.

    Returned *squared* (not the distance) because panel (c) plots the
    share D²_k / D²_K, the class-separation analogue of the scree panel's
    cumulative explained-variance ratio — both partition a variance-scale
    quantity across leading directions. The raw pooled covariance is used
    (no shrinkage): cells ≫ K_MAX here, so it is well-conditioned, and it
    makes D²_k exactly monotone non-decreasing in k (nested subspaces), so
    the share lands cleanly in [0, 1]. Left un-de-biased on purpose — the
    share is a shape statistic like the scree ratio; the absolute level is
    reported de-biased in panel (b) instead.

    The PCA is fit *unsupervised* on all cells of this side, so the share
    answers: what fraction of the leading-subspace class separability lives
    in the k dominant feature directions? An output share that saturates at
    smaller k than the input = the module packs the class contrast into
    fewer leading directions ("same separability, fewer features").
    """
    n1, n2 = int(fg.sum()), int((~fg).sum())
    Y = PCA(n_components=K_MAX).fit_transform(feat)
    d2 = np.empty(K_MAX)
    for k in range(1, K_MAX + 1):
        Z = Y[:, :k]
        mu_b, mu_g = Z[fg].mean(axis=0), Z[~fg].mean(axis=0)
        Xc = np.vstack([Z[fg] - mu_b, Z[~fg] - mu_g])
        S_w = Xc.T @ Xc / (n1 + n2 - 2)
        d = mu_b - mu_g
        d2[k - 1] = max(d @ np.linalg.solve(S_w, d), 0.0)
    return d2


def image_stats(model, device, path):
    """Bottom-centred-crop hooked forward pass → per-image statistics
    dict, or None if either class has too few Ham-grid cells."""
    res = hooked_crop_features(model, device, path)
    if res is None:
        return None
    fg, feat_in, feat_out = res

    stats = {"fg_frac": float(fg.mean()), "scree": {},
             "var": {}, "sep": {}}
    # Normalise each class's variance by the side's pooled within-class
    # variance RMS_w² (the unit the separation statistic already uses);
    # the module docstring's panel (b) note explains why RMS_w² — not
    # raw or total variance — is the right denominator.
    pw = {"in": float(np.mean(
              ham16.within_class_deviation(feat_in, fg) ** 2)),
          "out": float(np.mean(
              ham16.within_class_deviation(feat_out, fg) ** 2))}
    for cls, m in (("Brk", fg), ("Bg", ~fg)):
        for which, feat in (("in", feat_in), ("out", feat_out)):
            stats["scree"][(cls, which)] = _scree(feat, m)
            stats["var"][(cls, which)] = (_trace_var(feat, m)
                                          / (pw[which] + 1e-12))
    stats["sep"]["in"] = _separation(feat_in, fg)
    stats["sep"]["out"] = _separation(feat_out, fg)
    # Direction-aware (Fisher) versions for the companion figure
    stats["wvar"], stats["msep"], stats["msep2"] = {}, {}, {}
    stats["shrink"], stats["sepk"] = {}, {}
    for which, feat in (("in", feat_in), ("out", feat_out)):
        wvar, msep, msep2, shrink = _whitened_stats(feat, fg)
        for cls in ("Brk", "Bg"):
            stats["wvar"][(cls, which)] = wvar[cls]
        stats["msep"][which] = msep
        stats["msep2"][which] = msep2
        stats["shrink"][which] = shrink
        stats["sepk"][which] = _sep_vs_components(feat, fg)
    return stats


def bypass_prediction_change(model, device, paths):
    """Task-level view of the module (no labels needed): how much do
    the *predictions* change when the Hamburger is bypassed?

    The bypass hook replaces the module's output with ReLU(input) — the
    same ablation as 19_erf.py — so the coarse decoder features pass
    through but the NMF reconstruction is removed. Per image, the full
    and bypassed predictions are compared directly: IoU of the breaking
    masks, predicted-breaking area ratio, and the mean |Δ P(breaking)|
    over pixels. High agreement ⇒ the module barely moves the decision;
    low IoU ⇒ it is load-bearing for the prediction, not cosmetic.
    Console-only (this is the simple task-anchored summary; the figures
    carry the geometry story).
    """
    ham = model.decode_head.hamburger
    ious, area_ratios, dprobs = [], [], []
    for i, path in enumerate(paths):
        img_tensor, _hw = preprocess_crop(path)
        with torch.no_grad():
            logits = model(img_tensor.to(device))
            p_full = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
        hook = ham.register_forward_hook(
            lambda _m, inp, _o: F.relu(inp[0], inplace=False))
        with torch.no_grad():
            logits = model(img_tensor.to(device))
            p_byp = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
        hook.remove()

        full = p_full > PRODUCTION_THRESHOLD
        byp = p_byp > PRODUCTION_THRESHOLD
        union = np.logical_or(full, byp).sum()
        iou = (np.logical_and(full, byp).sum() / union) if union else 1.0
        area_ratio = byp.sum() / max(int(full.sum()), 1)
        dprob = float(np.abs(p_byp - p_full).mean())
        ious.append(iou)
        area_ratios.append(area_ratio)
        dprobs.append(dprob)
        print(f"  [{i + 1}/{len(paths)}] {path.name}: IoU {iou:.3f}, "
              f"area ×{area_ratio:.2f}, mean|Δprob| {dprob:.4f}")

    ious, area_ratios, dprobs = map(np.array, (ious, area_ratios, dprobs))
    print(f"Bypass vs full ({len(paths)} images):")
    print(f"  breaking-mask IoU: median {np.median(ious):.3f} "
          f"(IQR {np.percentile(ious, 25):.3f}–"
          f"{np.percentile(ious, 75):.3f})")
    print(f"  predicted breaking area ratio (bypass/full): median "
          f"×{np.median(area_ratios):.2f}")
    print(f"  mean |ΔP(breaking)| per pixel: median {np.median(dprobs):.4f}")


BOX_LW = 0.7    # thin box/whisker lines


def _box(ax, pos, vals, col, filled, alpha=0.45):
    """One box. Before/after encoding matches the scree line styles:
    Ham in = filled box with solid edge, Ham out = unfilled box with
    dashed edge. Box = quartiles, black bar = median, whiskers =
    1.5 × IQR; the only individual points drawn are outlier fliers.
    `alpha` sets the fill opacity when filled (default 0.45)."""
    bp = ax.boxplot([np.asarray(vals)], positions=[pos], widths=0.55,
                    patch_artist=True,
                    boxprops={"linewidth": BOX_LW},
                    whiskerprops={"color": col, "linewidth": BOX_LW},
                    capprops={"color": col, "linewidth": BOX_LW},
                    medianprops={"color": "black", "linewidth": 1.0},
                    flierprops={"marker": "o", "markersize": 2.5,
                                    "markerfacecolor": "0.3",
                                    "markeredgecolor": "none"})
    patch = bp["boxes"][0]
    patch.set_edgecolor(col)
    if filled:
        patch.set_facecolor(mcolors.to_rgba(col, alpha))
    else:
        patch.set_facecolor("none")
        patch.set_linestyle("--")


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser)
    parser.add_argument("--fast", action="store_true",
                        help=f"use only {N_FAST} images (quick styling "
                             "iterations)")
    parser.add_argument("--bypass-iou", action="store_true",
                        help="task-level ablation only: compare full vs "
                             "Ham-bypassed predictions (console, no "
                             "figures)")
    args = parser.parse_args()

    apply_style()
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir

    paths = _select_images(N_FAST if args.fast else N_IMAGES)
    print(f"Selected {len(paths)} images from {IMAGE_DIR}")

    if args.bypass_iou:
        bypass_prediction_change(runtime.model, runtime.device, paths)
        return

    all_stats = []
    for i, path in enumerate(paths):
        s = image_stats(runtime.model, runtime.device, path)
        if s is None:
            print(f"  [{i + 1}/{len(paths)}] {path.name}: skipped "
                  f"(a class has <= {N_SCREE} cells)")
            continue
        print(f"  [{i + 1}/{len(paths)}] {path.name}: "
              f"{100 * s['fg_frac']:.1f}% breaking, "
              f"sep {s['sep']['in']:.2f} → {s['sep']['out']:.2f}")
        all_stats.append(s)
    n = len(all_stats)
    if n < 2:
        raise RuntimeError(f"only {n} usable images — need at least 2")
    print(f"Usable images: {n}/{len(paths)}")

    # ── Aggregate: per-image before/after values ──
    sep_in = np.array([s["sep"]["in"] for s in all_stats])
    sep_out = np.array([s["sep"]["out"] for s in all_stats])
    print(f"  separation median: {np.median(sep_in):.2f} → "
          f"{np.median(sep_out):.2f}")
    var_vals = {}
    for cls in ("Brk", "Bg"):
        vin = np.array([s["var"][(cls, "in")] for s in all_stats])
        vout = np.array([s["var"][(cls, "out")] for s in all_stats])
        var_vals[cls] = (vin, vout)
        r = vout / vin
        print(f"  relative-variance ratio out/in, {cls}: median "
              f"{np.median(r):.3f} (range {r.min():.3f}–{r.max():.3f})")
    msep_in = np.array([s["msep"]["in"] for s in all_stats])
    msep_out = np.array([s["msep"]["out"] for s in all_stats])
    print(f"  bias-corrected Mahalanobis separation median: "
          f"{np.median(msep_in):.2f} → {np.median(msep_out):.2f}")
    n_floor = sum((s["msep2"][w] <= 0) for s in all_stats for w in ("in", "out"))
    if n_floor:
        print(f"    ({n_floor}/{2 * n} side-images had de-biased D² ≤ 0 — "
              f"separation indistinguishable from zero, floored to 0)")
    wvar_vals = {}
    for cls in ("Brk", "Bg"):
        win = np.array([s["wvar"][(cls, "in")] for s in all_stats])
        wout = np.array([s["wvar"][(cls, "out")] for s in all_stats])
        wvar_vals[cls] = (win, wout)
        print(f"  whitened-variance ratio out/in, {cls}: median "
              f"{np.median(wout / win):.3f}")
    for which in ("in", "out"):
        sh = np.array([s["shrink"][which] for s in all_stats])
        print(f"  Ledoit-Wolf shrinkage, {which}: median {np.median(sh):.4f}")
    for which in ("in", "out"):
        d2 = np.stack([s["sepk"][which] for s in all_stats])
        share = d2 / (d2[:, -1:] + 1e-12)     # fraction of top-K squared sep
        # first k reaching 90% of that image's top-K squared separation
        k90 = np.argmax(share >= 0.9, axis=1) + 1
        print(f"  components to 90% of top-{K_MAX} separation², {which}: "
              f"median {np.median(k90):.0f} (IQR {np.percentile(k90, 25):.0f}–"
              f"{np.percentile(k90, 75):.0f})")

    # ── Figure: 1 × 3, same layout as the single-image version ──
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(PAGE_W, PAGE_W * 0.32), layout="constrained")

    # (a) scree: mean over images ± 1 std. Colour = class; input =
    # solid line + filled markers, output = dashed line + open markers
    # (same fill encoding as the boxes). The breaking-input and
    # background-output means nearly coincide, so the curves carry
    # markers with staggered spacing — both stay visible where the
    # lines lie on top of each other.
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
    ax.set_ylim(0, 1.02)     # axes always start at 0
    ax.set_xlabel("Components")
    # two lines: the one-line label is taller than the figure and gets
    # clipped even with a tight save bbox
    ax.set_ylabel("Cumulative explained\nvariance")
    handles = [Line2D([], [], color=FG_COL, label="Breaking"),
               Line2D([], [], color=BG_COL, label="Background")]
    styled_legend(ax, handles=handles, loc="lower right")
    panel_label(ax, "a")

    # (b) within-class variance before/after, one box pair per class;
    # out box below its in box = homogenised
    ax = ax_b
    for gi, cls in enumerate(("Brk", "Bg")):
        col = FG_COL if cls == "Brk" else BG_COL
        vin, vout = var_vals[cls]
        _box(ax, gi * 2.0 + 0.6, vin, col, filled=True)
        _box(ax, gi * 2.0 + 1.4, vout, col, filled=False)
    ax.set_xticks([1.0, 3.0], ["Breaking", "Background"])
    ax.set_xlim(0, 4.0)
    # 25 % headroom above the largest value keeps the top strip free of
    # data, so the single-row legend there can never sit on a box
    top = max(v.max() for pair in var_vals.values() for v in pair)
    ax.set_ylim(0, top * 1.25)
    # in/out legend lives here: neutral patches mimicking the box
    # styles (filled solid edge = input, unfilled dashed = output)
    handles = [mpatches.Patch(facecolor=mcolors.to_rgba("0.2", 0.35),
                              edgecolor="0.2", linewidth=BOX_LW,
                              label="Input"),
               mpatches.Patch(facecolor="none", edgecolor="0.2",
                              linewidth=BOX_LW, linestyle="--",
                              label="Output")]
    styled_legend(ax, handles=handles, loc="upper right", ncol=1)
    # LaTeX for the draft caption:
    #   $\langle\,\|f - \mu_{class}\|^2\,\rangle \,/\, \mathrm{RMS}_w^2$
    # (mean squared distance of a cell's features to its class centroid,
    # divided by the pooled within-class variance of the same side —
    # scale-invariant, and unlike a share of total variance it is not
    # deflated mechanically when the between-class gap grows)
    ax.set_ylabel("Relative intraclass variance")
    panel_label(ax, "b")

    # (c) class separation before/after; out box above the in box =
    # classes more separable after the module
    ax = ax_c
    _box(ax, 0.6, sep_in, IBM[1], filled=True)
    _box(ax, 1.4, sep_out, IBM[1], filled=False)
    ax.set_xticks([0.6, 1.4], ["Input", "Output"])
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)
    # LaTeX for the draft caption:
    #   $\|\mu_{brk} - \mu_{bg}\|\,/\,\mathrm{RMS}_w$
    # (centroid gap in units of the pooled within-class RMS deviation)
    ax.set_ylabel("Class separation")
    panel_label(ax, "c")

    savefig(fig, OUT_DIR / "21_ham_variability_pooled.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)

    # ── Second output: direction-aware (Fisher) versions of (b), (c) ──
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(PAGE_W, PAGE_W * 0.32), layout="constrained")

    # (a) whitened intraclass variance per class, input vs output
    ax = ax_a
    for gi, cls in enumerate(("Brk", "Bg")):
        col = FG_COL if cls == "Brk" else BG_COL
        win, wout = wvar_vals[cls]
        _box(ax, gi * 2.0 + 0.6, win, col, filled=True)
        _box(ax, gi * 2.0 + 1.4, wout, col, filled=False)
    ax.axhline(1.0, color="0.6", linestyle=":", linewidth=0.7)
    ax.set_xticks([1.0, 3.0], ["Breaking", "Background"])
    ax.set_xlim(0, 4.0)
    top = max(v.max() for pair in wvar_vals.values() for v in pair)
    ax.set_ylim(0, top * 1.25)
    handles = [mpatches.Patch(facecolor=mcolors.to_rgba("0.2", 0.35),
                              edgecolor="0.2", linewidth=BOX_LW,
                              label="Input"),
               mpatches.Patch(facecolor="none", edgecolor="0.2",
                              linewidth=BOX_LW, linestyle="--",
                              label="Output")]
    styled_legend(ax, handles=handles, loc="upper right", ncol=1)
    # LaTeX for the draft caption:
    #   $\mathrm{tr}\!\left(S_w^{-1}\,\Sigma_{class}\right) / C$
    # (class covariance whitened by the within-class covariance S_w and
    # averaged per direction; dotted line = the within-class average, 1)
    ax.set_ylabel("Whitened intraclass variance")
    panel_label(ax, "a")

    # (b) Mahalanobis centroid separation (√ two-class Fisher criterion)
    ax = ax_b
    _box(ax, 0.6, msep_in, IBM[1], filled=True)
    _box(ax, 1.4, msep_out, IBM[1], filled=False)
    ax.set_xticks([0.6, 1.4], ["Input", "Output"])
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)
    # LaTeX for the draft caption:
    #   $\sqrt{(\mu_{brk}-\mu_{bg})^{\top} S_w^{-1} (\mu_{brk}-\mu_{bg})}$
    # (Mahalanobis distance between the class centroids — the square
    # root of the two-class Fisher criterion)
    ax.set_ylabel("Mahalanobis class separation")
    panel_label(ax, "b")

    # (c) Cumulative *share* of the squared Mahalanobis class separation
    # carried by the top-k PCA directions, per side (D²_k / D²_K). The
    # discriminative twin of the scree panel: dividing by the top-K total
    # removes the overall separation level — which the de-biased panel (b)
    # shows is ~unchanged — and isolates *concentration*. Output curve
    # (dashed) above input (solid) at small k ⇒ the module packs the same
    # separability into fewer leading directions. Same in/out encoding as
    # everywhere else: solid + filled marker = input, dashed + open = output.
    # Plotting the share, not the absolute per-k separation, is what keeps a
    # marginally-lower output level from reading as a loss of concentration.
    ax = ax_c
    d2_in = np.stack([s["sepk"]["in"] for s in all_stats])
    d2_out = np.stack([s["sepk"]["out"] for s in all_stats])
    x = np.arange(1, K_MAX + 1)
    for d2, ls, mfc in ((d2_in, "-", IBM[1]), (d2_out, "--", "white")):
        share = d2 / (d2[:, -1:] + 1e-12)
        med = np.median(share, axis=0)
        ax.plot(x, med, color=IBM[1], linestyle=ls, marker="o", markersize=2.6,
                markevery=4, markerfacecolor=mfc, markeredgecolor=IBM[1],
                markeredgewidth=0.6)
        ax.fill_between(x, np.percentile(share, 25, axis=0),
                        np.percentile(share, 75, axis=0), color=IBM[1],
                        alpha=0.15, linewidth=0)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Components")
    # LaTeX for the draft caption:
    #   $D^2_k / D^2_K$, with $D^2_k = \Delta\mu_k^{\top} S_{w,k}^{-1}
    #   \Delta\mu_k$ over the top-$k$ PCA directions of each side (raw
    #   pooled $S_{w,k}$)
    # (fraction of the leading-subspace squared Mahalanobis class
    # separation carried by the k dominant directions, median and IQR over
    # images; output above input at small k = separation concentrated into
    # fewer directions — the discriminative analogue of the scree panel)
    ax.set_ylabel("Separation share (top-k)")
    handles = [Line2D([], [], color=IBM[1], linestyle="-", label="Input"),
               Line2D([], [], color=IBM[1], linestyle="--", label="Output")]
    styled_legend(ax, handles=handles, loc="lower right")
    panel_label(ax, "c")

    savefig(fig, OUT_DIR / "21_ham_variability_fisher.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
