"""Appendix figure — how the Hamburger NMF reshapes the class geometry.

Shared-basis PCA of Ham input vs output looks nearly identical because the
NMF residual is small relative to the dominant feature axes — the colours
barely move even when the class geometry changes substantially. The
statistics below measure that change directly instead of relying on RGB maps.

What this shows
---------------
The Hamburger module (Geng et al. 2021, "Is Attention Better Than Matrix
Decomposition?", ICLR, arXiv:2109.04553) reconstructs the coarse decoder
features from a small shared dictionary (NMF), linking all spatial
positions. The panels measure how that global step changes the geometry
of the two classes (breaking / background): dimensionality (a),
within-class spread (b), and between-class separation (c). On this
checkpoint separation rises while the breaking class *loses* intraclass
variance relative to the pooled within-class level — the module compacts
the minority-class representation and moves it away from background
(homogenisation with increased separation). The sign of the (b) shift is
checkpoint-dependent: read it off the console
`relative intraclass variance … (ratio …)` line rather than this comment
— ratio < 1 is homogenisation, > 1 enrichment.

Panels (statistics only — the spatial deviation maps were dropped: at
1/8 feature resolution they are too noisy to read and the statistics
carry the claim; panel tags and styling use the shared figure-style module):

(a) Scree curves **per class**: cumulative PCA explained variance vs
    number of components, fit separately on breaking cells and background
    cells, for Ham input (solid) and output (dashed). Legend: colour =
    class, line style = in/out; the number of components to reach 90 % is
    printed to the console. Structure packed into fewer components ⇒ the
    NMF built a lower-dimensional representation of that class.
(b) *Relative* intraclass variance per class, Ham input vs output:
    ⟨‖f − μ_class‖²⟩ (total feature variance over the cells of one
    class, summed over channels) divided by RMS_w², the pooled
    within-class variance of the same side — the unit the separation in
    (c) already uses. Two normalisation traps this avoids: raw
    variances inflate with the residual module's larger output norms
    (global scale), and dividing by *total* variance deflates both
    classes mechanically as the between-class gap grows (total = within
    + between), faking homogenisation whenever separation rises.
    A breaking bar *falling* from input to output ⇒ breaking cells
    became more internally uniform relative to the pooled level
    (homogenisation); a *rising* bar ⇒ more diverse (enrichment) — the
    printed ratio gives the sign. With ~99 % background cells the
    background bar sits near 1 by construction — the built-in control —
    and a small background move in the opposite direction is the
    expected mirror of the breaking shift (RMS_w² ≈ background
    variance), not evidence of a background change on its own.
(c) Class separation before vs after. The y-axis ‖μ_brk − μ_bg‖ / RMS_w
    is the distance between the breaking and background centroids in
    units of the pooled within-class RMS deviation (a Fisher-style
    discriminability ratio). Separation is gap-per-spread, so it can
    rise together with the relative breaking variance in (b) —
    centroids moving apart faster than the classes spread (enrichment),
    as opposed to rising because the classes tightened (homogenisation).

The breaking/background labels come from the model's own prediction
(P > 0.48, the production operating point) mapped to the Ham grid with the
"any pixel" rule, as everywhere
else in these appendix scripts.

Outputs
-------
  16_ham_variability.{pdf,png}

Run after installing the repository environment:
    python figures/16_hamburger_viz.py
"""

from pathlib import Path

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import dunex_paths

from common.figure_style import (apply_style, savefig, panel_label, styled_legend,
                   PAGE_W, FONTSIZE_LEGEND, IBM)
from checkpoint import load_checkpoint_model
from common.normalization import MEAN_NP as IMAGENET_MEAN
from common.normalization import STD_NP as IMAGENET_STD
from segmentation.predict_video import PRODUCTION_THRESHOLD, get_device

# Representative in-repo sample; swap for any file under IMAGE_DIR.
IMAGE_DIR = dunex_paths.REPO_ROOT / "model/data/full_split/images"
IMAGE_PATH = IMAGE_DIR / "ArgusFF_20211007T210100Z_y0600_f1624.png"
# The production Argus model: SegNeXt-T, learned-upsample decoder with H/4 + H/2
# skips (config ablation/configs/segnext_t_learned_up_skip_4_2.yaml).
# ``load_checkpoint_model`` takes the run directory holding ``best_model.pth`` and
# rebuilds the architecture from the config stored inside the checkpoint.
CHECKPOINT_PATH = (
    dunex_paths.OUTPUTS_DIR
    / "promoted/segnext_t_learned_up_skip_4_2/best_model.pth"
)
RUN_DIR = CHECKPOINT_PATH.parent
OUT_DIR = Path(__file__).parent

N_SCREE = 32            # components shown in the scree panel
COLOR_IN, COLOR_OUT = IBM[0], IBM[3]     # blue = Ham in, orange = Ham out
# Class colours: breaking = IBM pink; background = Okabe-Ito bluish green
# (distinct from the blue/orange in/out pair; not grey, not yellow)
FG_COL, BG_COL = IBM[2], "#009E73"


def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor, img, (img.shape[0], img.shape[1])


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


class HamburgerIOHook:
    """Capture features entering and leaving model.decode_head.hamburger."""

    def __init__(self):
        self.inp = None
        self.out = None
        self._hook = None

    def register(self, model):
        def _fn(_module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self.inp = x.detach().cpu()
            self.out = out.detach().cpu()
        self._hook = model.decode_head.hamburger.register_forward_hook(_fn)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def _downsample_mask(mask, feat_hw):
    """Full-res bool mask → feature cells, 'any pixel' rule (INTER_AREA
    block mean > 0 ⟺ at least one covered pixel was True)."""
    H_feat, W_feat = feat_hw
    mean = cv2.resize(mask.astype(np.float32), (W_feat, H_feat),
                      interpolation=cv2.INTER_AREA)
    return mean > 0


def within_class_deviation(feat, fg):
    """Per-cell distance to its own class centroid, ‖f_i − μ_class‖.

    The centroids are computed on the same feature set, per class; this is
    the quantity whose reduction means "NMF pulled the class together".
    """
    d = np.zeros(feat.shape[0])
    for m in (fg, ~fg):
        if m.sum():
            d[m] = np.linalg.norm(feat[m] - feat[m].mean(axis=0), axis=1)
    return d


def main():
    apply_style()

    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {RUN_DIR}")
    model, _ = load_checkpoint_model(RUN_DIR, device)

    print(f"Loading image: {IMAGE_PATH}")
    img_tensor, _, img_hw = preprocess_image(IMAGE_PATH)
    viz_size = (img_hw[1], img_hw[0])

    print("Forward pass (hooked)...")
    hook = HamburgerIOHook()
    hook.register(model)
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    hook.remove()
    if prob.shape != img_hw:      # output frame must equal the input size
        prob = cv2.resize(prob, viz_size, interpolation=cv2.INTER_NEAREST)
    fg_mask = prob > PRODUCTION_THRESHOLD

    ham_inp, ham_out = hook.inp, hook.out
    C, H, W = ham_inp.shape[1], ham_inp.shape[2], ham_inp.shape[3]
    feat_in = ham_inp[0].permute(1, 2, 0).reshape(-1, C).numpy()
    feat_out = ham_out[0].permute(1, 2, 0).reshape(-1, C).numpy()
    fg = _downsample_mask(fg_mask, (H, W)).flatten()
    print(f"  Ham grid {H}×{W}, {100 * fg.mean():.1f}% breaking cells")

    # ── Scree per class: cumulative explained variance ──
    def _scree(feat, m):
        n_pc = int(min(N_SCREE, C, m.sum() - 1))
        if n_pc < 2:
            return None, 0
        ev = np.cumsum(PCA(n_components=n_pc).fit(feat[m])
                       .explained_variance_ratio_)
        return ev, int(np.searchsorted(ev, 0.9)) + 1

    scree = {}
    for cls, m in (("Brk", fg), ("Bg", ~fg)):
        scree[(cls, "in")] = _scree(feat_in, m)
        scree[(cls, "out")] = _scree(feat_out, m)

    # ── Within-class variance (trace of channel covariance) + separation ──
    def _trace_var(feat, m):
        return float(feat[m].var(axis=0).sum()) if m.sum() else np.nan

    # Normalise by each side's pooled within-class variance RMS_w² (the
    # unit the separation statistic uses): scale-invariant against the
    # residual module's inflated output norms, and — unlike a share of
    # total variance — not deflated mechanically when the between-class
    # gap grows
    pw_in = float(np.mean(within_class_deviation(feat_in, fg) ** 2))
    pw_out = float(np.mean(within_class_deviation(feat_out, fg) ** 2))
    var_in, var_out = {}, {}
    for name, m in (("Breaking", fg), ("Background", ~fg)):
        var_in[name] = _trace_var(feat_in, m) / (pw_in + 1e-12)
        var_out[name] = _trace_var(feat_out, m) / (pw_out + 1e-12)
        print(f"  relative intraclass variance, {name}: "
              f"{var_in[name]:.3f} → {var_out[name]:.3f} "
              f"(ratio {var_out[name] / (var_in[name] + 1e-12):.3f})")

    def _separation(feat):
        """‖μ_fg − μ_bg‖ / pooled within-class RMS deviation."""
        mu_gap = np.linalg.norm(feat[fg].mean(axis=0) - feat[~fg].mean(axis=0))
        pooled = np.sqrt(np.mean(within_class_deviation(feat, fg) ** 2))
        return mu_gap / (pooled + 1e-12)

    sep_in, sep_out = _separation(feat_in), _separation(feat_out)
    print(f"  class separation (centroid gap / within-class RMS): "
          f"{sep_in:.2f} → {sep_out:.2f}")
    for cls in ("Brk", "Bg"):
        print(f"  components to 90% variance ({cls}): "
              f"{scree[(cls, 'in')][1]} → {scree[(cls, 'out')][1]}")

    # ── Figure: 1 × 3 statistics panels ──
    fig, axes = plt.subplots(1, 3, figsize=(PAGE_W, PAGE_W * 0.32),
                             layout="constrained")

    # (a) scree per class (colour = class, style = in/out); the legend
    # shows the two encodings separately instead of one entry per curve
    ax = axes[0]
    for cls, col in (("Brk", FG_COL), ("Bg", BG_COL)):
        for which, ls in (("in", "-"), ("out", "--")):
            ev, _n90 = scree[(cls, which)]
            if ev is None:
                continue
            ax.plot(np.arange(1, len(ev) + 1), ev, color=col, linestyle=ls)
    ax.axhline(0.9, color="0.6", linestyle=":", linewidth=0.7)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)     # axes always start at 0
    ax.set_xlabel("Components")
    # two lines: the one-line label is taller than the figure and gets
    # clipped even with a tight save bbox
    ax.set_ylabel("Cumulative explained\nvariance")
    handles = [Line2D([], [], color=FG_COL, label="Breaking"),
               Line2D([], [], color=BG_COL, label="Background"),
               Line2D([], [], color="0.2", linestyle="-", label="Input"),
               Line2D([], [], color="0.2", linestyle="--", label="Output")]
    styled_legend(ax, handles=handles, loc="lower right")
    panel_label(ax, "a")

    # (b) within-class variance per class, Ham in vs out
    ax = axes[1]
    names = ["Breaking", "Background"]
    xs = np.arange(len(names))
    ax.bar(xs - 0.2, [var_in[n] for n in names], width=0.4,
           color=COLOR_IN, label="Input")
    ax.bar(xs + 0.2, [var_out[n] for n in names], width=0.4,
           color=COLOR_OUT, label="Output")
    ax.set_xticks(xs, names)
    # LaTeX for the draft caption:
    #   $\langle\,\|f - \mu_{class}\|^2\,\rangle \,/\, \mathrm{RMS}_w^2$
    # (mean squared distance of a cell's features to its class centroid,
    # divided by the pooled within-class variance of the same side —
    # scale-invariant, and unlike a share of total variance it is not
    # deflated mechanically when the between-class gap grows)
    ax.set_ylabel("Relative intraclass variance")
    styled_legend(ax)
    panel_label(ax, "b")

    # (c) class separation before/after (Fisher-style ratio)
    ax = axes[2]
    bars = ax.bar(["Input", "Output"], [sep_in, sep_out],
                  color=[COLOR_IN, COLOR_OUT], width=0.5)
    ax.bar_label(bars, fmt="%.2f", fontsize=FONTSIZE_LEGEND, padding=2)
    ax.set_ylim(0, max(sep_in, sep_out) * 1.15)
    # LaTeX for the draft caption:
    #   $\|\mu_{brk} - \mu_{bg}\|\,/\,\mathrm{RMS}_w$
    # (centroid gap in units of the pooled within-class RMS deviation)
    ax.set_ylabel("Class separation")
    panel_label(ax, "c")

    savefig(fig, OUT_DIR / "16_ham_variability.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
