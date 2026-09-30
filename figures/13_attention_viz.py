"""Appendix figures — SegNeXt MSCA encoder interpretability.

Paper version of ``osm2026/presentation/13_attention_viz.py``, trimmed to the
figures used in the manuscript appendix and restyled with the shared paper
style:

  13_pca_native_boundary.{pdf,png}
      Class-conditioned PCA of the MSCA encoder features at native
      feature-map resolution, with an explicit boundary class (rows: all
      pixels / breaking / boundary / background; columns: encoder stages
      0-3). Boundary cells are background cells within one cell of the
      breaking mask — isolating them keeps edge-straddling cells from
      contaminating the pure-class PCA fits.

  13_attention_alignment_axis.{pdf,png}
      Attention contribution Δ = SpatialAttention(output − input) projected
      onto the class separating axis d = μ_brk − μ_nonbrk, stage-scaled by
      the mean ‖Δ‖ (Δ·d̂ / mean‖Δ‖). Sign = direction (green = toward the
      breaking centroid, purple = toward non-breaking); intensity = how
      strongly, in units of the stage's typical residual magnitude.

  13_attention_alignment_correct.{pdf,png}
      Same stage-scaled projection signed by each cell's own (predicted)
      class: green = attention pushes the cell toward its correct class
      centroid (reinforces the classification), purple = toward the wrong class.

Method notes (abridged from the source script):
- Class-conditioned PCA (DINOv2 style): PCA is fit only on one class's
  feature-map cells, removing the dominant fg/bg axis so the components
  reveal intra-class structure. Cells are labelled with an "any pixel" rule
  (a cell is breaking if any image pixel in its receptive field is).
- Nearest-neighbour interpolation everywhere: feature vectors are never
  blended across cells, and the blocky rendering shows the encoder's true
  spatial granularity at each stage.
- The model's own prediction (P > 0.48, the production operating point)
  provides the breaking mask, so the figures are self-contained (no labels
  required).

All per-pixel products (prediction masks, upscaled feature maps) are forced
to the exact input-image size via ``_match_size`` — fully convolutional
models can pad to stride multiples, and a silent off-by-a-few-pixels
mismatch would misalign every mask/feature overlay.

References
----------
- Guo, M.-H. et al. (2022). SegNeXt: Rethinking Convolutional Attention
  Design for Semantic Segmentation. NeurIPS. arXiv:2209.08575
- Oquab, M. et al. (2024). DINOv2: Learning Robust Visual Features without
  Supervision. TMLR. arXiv:2304.07193  (class-conditioned PCA technique)
- Caron, M. et al. (2021). Emerging Properties in Self-Supervised Vision
  Transformers (DINO). ICCV. arXiv:2104.14294  (attention-salience maps;
  our Δ-residual maps are the convolutional-gating analogue)

Run after installing the repository environment:
    python figures/13_attention_viz.py
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import dunex_paths

from common.figure_style import apply_style, savefig, PAGE_W, FONTSIZE_LABEL, FONTSIZE_LEGEND
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

NUM_STAGES = 4
BOUNDARY_WIDTH = 1      # boundary band width in cells (BG side of the edge)
BG_GRAY = 0.85          # fill for cells excluded from a class-conditioned PCA
PRED_TINT = np.array([1.0, 0.15, 0.15])     # red overlay: predicted breaking
PRED_ALPHA = 0.6                            # prediction-overlay opacity (0=image, 1=solid tint)
BOUNDARY_TINT = np.array([0.3, 1.0, 0.3])   # green overlay: boundary cells
ALIGN_CMAP = "PRGn"     # diverging map for alignment: purple (−) → green (+)


# ---------------------------------------------------------------------------
# Image + prediction
# ---------------------------------------------------------------------------

def preprocess_image(image_path):
    """Load, ImageNet-normalise, and return (tensor, normalised HWC, (H, W))."""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor, img, (img.shape[0], img.shape[1])


def denormalize(img):
    return np.clip(img * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def _tint_overlay(img, mask, tint, alpha=PRED_ALPHA, outline=True):
    """Alpha-blend a flat ``tint`` over a copy of ``img`` where ``mask`` is True.

    A solid 2-px outline traces the mask edge so the highlighted region stays
    legible even where the semi-transparent fill sits over bright foam.
    """
    out = img.copy()
    out[mask] = out[mask] * (1 - alpha) + tint * alpha
    if outline:
        m = mask.astype(np.uint8)
        edge = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=2) - m
        out[edge.astype(bool)] = tint
    return out


def _match_size(arr, img_hw):
    """Force a per-pixel map to the exact input-image size.

    Fully convolutional models may pad inputs to a stride multiple, so the
    output frame can be a few pixels larger than the input; every mask and
    overlay downstream assumes they align exactly. No-op when they already
    match; nearest-neighbour resize otherwise (never blends values).
    """
    H, W = img_hw
    if arr.shape[:2] != (H, W):
        arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    return arr


def mask_from_logits(logits, img_hw):
    """Model's own breaking prediction at the production threshold as a
    bool mask, guaranteed to be exactly the input-image size.

    Takes logits from the caller so the prediction reuses the same forward
    pass the hooks recorded — a second forward would re-run the Hamburger
    decode head, whose NMF bases are randomly re-initialised each call."""
    prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    return _match_size(prob, img_hw) > PRODUCTION_THRESHOLD


def get_prediction_mask(model, img_tensor, device, img_hw):
    """Forward pass + ``mask_from_logits`` in one call.

    Kept as the module's public entry point for analyses that need the mask
    without intermediate-activation hooks. This script's ``main`` instead
    reuses the hooked pass's logits so the decode head is not evaluated twice."""
    with torch.no_grad():
        logits = model(img_tensor.to(device))
    return mask_from_logits(logits, img_hw)


# ---------------------------------------------------------------------------
# Forward hooks — capture SpatialAttention input/output per MSCAN block
# ---------------------------------------------------------------------------

class AttentionHook:
    """Record (input, output) of every MSCAN block's SpatialAttention module.

    The residual Δ = output − input isolates exactly what the attention
    branch adds at each spatial position.
    """

    def __init__(self):
        self.activations = defaultdict(list)
        self.hooks = []

    def register_hooks(self, model):
        for stage_idx in range(NUM_STAGES):
            block = getattr(model.backbone, f"block{stage_idx + 1}")
            for blk_idx, blk in enumerate(block):
                self.hooks.append(
                    blk.attn.register_forward_hook(
                        self._io_hook(f"spatial_attn_stage{stage_idx}_blk{blk_idx}")
                    )
                )

    def _io_hook(self, name):
        def fn(_module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            self.activations[f"{name}_input"].append(x.detach().cpu())
            self.activations[f"{name}_output"].append(out.detach().cpu())
        return fn

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Shared numeric helpers
# ---------------------------------------------------------------------------

def _normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def _downsample_factor(img_h, feat_h):
    """Nominal power-of-2 downsample ratio for panel titles (e.g. '1/8')."""
    if feat_h <= 0:
        return "?"
    return 2 ** round(np.log2(img_h / feat_h))


def _downsample_mask(mask, feat_hw):
    """Full-res bool mask → feature-map cells, 'any pixel' rule.

    INTER_AREA computes the block mean; mean > 0 ⟺ at least one source
    pixel was True, so a cell is labelled breaking if any pixel in its
    receptive field is.
    """
    H_feat, W_feat = feat_hw
    mean = cv2.resize(mask.astype(np.float32), (W_feat, H_feat),
                      interpolation=cv2.INTER_AREA)
    return mean > 0


def _fg_bg_boundary_cells(fg_mask_fullres, feat_hw, boundary_width=BOUNDARY_WIDTH):
    """Partition feature-map cells into FG / boundary / BG.

    FG cells use the 'any pixel' rule; the boundary is the band of BG cells
    within `boundary_width` cells of FG (dilate FG, intersect with BG); BG
    is the remainder. The three masks are mutually exclusive and cover all
    cells, and the FG row is identical to a plain two-way split.
    """
    H, W = feat_hw
    fg_cells = _downsample_mask(fg_mask_fullres, (H, W))
    bg_cells = ~fg_cells

    if boundary_width <= 0:
        return fg_cells, np.zeros((H, W), dtype=bool), bg_cells

    kernel = np.ones((3, 3), np.uint8)
    fg_d = cv2.dilate(fg_cells.astype(np.uint8), kernel,
                      iterations=boundary_width).astype(bool)
    boundary = fg_d & bg_cells
    return fg_cells, boundary, bg_cells & ~boundary


def _gather_stage_feats(model, activations):
    """Last block's SpatialAttention output per stage (most refined)."""
    blocks_per_stage = [
        len(getattr(model.backbone, f"block{i + 1}")) for i in range(NUM_STAGES)
    ]
    feats, resolutions = [], []
    for stage_idx in range(NUM_STAGES):
        key = (f"spatial_attn_stage{stage_idx}"
               f"_blk{blocks_per_stage[stage_idx] - 1}_output")
        if activations.get(key):
            feat = activations[key][0]
            feats.append(feat)
            resolutions.append((feat.shape[2], feat.shape[3]))
        else:
            feats.append(None)
            resolutions.append((0, 0))
    return feats, resolutions


def _gather_attention_contributions(activations):
    """(input, output) of the last hooked block per stage."""
    stage_contributions = {}
    for stage_idx in range(NUM_STAGES):
        blk_nums = set()
        for key in activations:
            if f"spatial_attn_stage{stage_idx}" in key and "_input" in key:
                for part in key.split("_"):
                    if part.startswith("blk"):
                        blk_nums.add(int(part[3:]))
        if not blk_nums:
            continue
        last = max(blk_nums)
        in_key = f"spatial_attn_stage{stage_idx}_blk{last}_input"
        out_key = f"spatial_attn_stage{stage_idx}_blk{last}_output"
        if activations.get(in_key) and activations.get(out_key):
            stage_contributions[stage_idx] = (
                activations[in_key][0], activations[out_key][0]
            )
    return stage_contributions


# ---------------------------------------------------------------------------
# PCA (native feature-map resolution, class conditioned)
# ---------------------------------------------------------------------------

def _pca_rgb_class(feat_flat, mask_flat, n_components=3):
    """Fit PCA on masked cells only; project all cells; normalise to the
    masked cells' range so colours are calibrated to the target class."""
    if mask_flat.sum() < n_components + 1:
        return None
    pca = PCA(n_components=min(n_components, feat_flat.shape[1]))
    masked = feat_flat[mask_flat]
    pca.fit(masked)
    projected = pca.transform(feat_flat)
    proj_masked = pca.transform(masked)
    for i in range(projected.shape[1]):
        lo, hi = proj_masked[:, i].min(), proj_masked[:, i].max()
        projected[:, i] = (projected[:, i] - lo) / (hi - lo + 1e-8)
    return np.clip(projected[:, :3], 0, 1)


def _native_pca_all(feat):
    """Unconditioned PCA RGB at the feature map's own resolution."""
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    feat_flat = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()
    pca = PCA(n_components=min(3, C))
    proj = pca.fit_transform(feat_flat).reshape(H, W, -1)
    for i in range(proj.shape[2]):
        proj[:, :, i] = _normalize(proj[:, :, i])
    return np.clip(proj[:, :, :3], 0, 1)


def _native_pca_cells(feat, cell_mask):
    """Class-conditioned PCA on an explicit cell mask; other cells gray."""
    C, H, W = feat.shape[1], feat.shape[2], feat.shape[3]
    feat_flat = feat[0].permute(1, 2, 0).reshape(-1, C).numpy()
    mask = cell_mask.flatten()
    rgb = _pca_rgb_class(feat_flat, mask)
    canvas = np.full((H * W, 3), BG_GRAY, dtype=np.float32)
    if rgb is not None:
        canvas[mask] = rgb[mask]
    return canvas.reshape(H, W, 3)


# ---------------------------------------------------------------------------
# Attention alignment (separating axis, stage-scaled signed projection)
# ---------------------------------------------------------------------------

def _attention_class_alignment_axis(stage_contributions, fg_mask):
    """Project Δ onto d̂ = (μ_fg − μ_bg)/‖·‖, stage-scaled by the mean ‖Δ‖.

    The value at each cell is Δ·d̂ / mean‖Δ‖ — the signed component of the
    attention residual along the class-separating axis, in units of that
    stage's typical residual magnitude. Unlike a pure cos(θ) this keeps
    magnitude: a cell that is barely reshaped reads near zero even when its
    tiny Δ happens to point toward breaking, while a strongly-driven cell
    reads large. Dividing by the per-stage mean ‖Δ‖ (rather than each cell's
    own ‖Δ‖, which would give cos θ) puts the four stages — whose feature
    scales differ by orders of magnitude — on a common, comparable axis so
    the per-stage panels and their average can share one colour scale.

    Positive = attention pushes the feature toward the breaking centroid,
    negative = toward non-breaking.

    Returns
    -------
    per_stage : {stage: (projection map (H, W), fg cell mask (H, W))}
    """
    per_stage = {}
    for si in sorted(stage_contributions):
        s_in, s_out = stage_contributions[si]
        delta = s_out - s_in
        C, H, W = delta.shape[1], delta.shape[2], delta.shape[3]

        out_flat = s_out[0].permute(1, 2, 0).reshape(-1, C)
        delta_flat = delta[0].permute(1, 2, 0).reshape(-1, C)
        fg_cells = _downsample_mask(fg_mask, (H, W))
        mask_ds = fg_cells.flatten()

        if mask_ds.sum() < 2 or (~mask_ds).sum() < 2:
            per_stage[si] = (np.zeros((H, W)), fg_cells)
            continue

        d = out_flat[mask_ds].mean(dim=0) - out_flat[~mask_ds].mean(dim=0)
        d_norm = d.norm()
        if d_norm < 1e-8:
            per_stage[si] = (np.zeros((H, W)), fg_cells)
            continue

        # Signed projection Δ·d̂ (feature-magnitude units), then divide by the
        # stage's mean ‖Δ‖ so all stages share a common magnitude-aware scale.
        proj = (delta_flat * (d / d_norm).unsqueeze(0)).sum(dim=1).numpy()
        mean_norm = delta_flat.norm(dim=1).mean().item()
        if mean_norm > 1e-8:
            proj = proj / mean_norm
        per_stage[si] = (proj.reshape(H, W), fg_cells)

    return per_stage


def _combine_maps(maps, img_hw):
    """Average per-stage maps (shared stage-scaled unit) at image resolution."""
    viz_size = (img_hw[1], img_hw[0])
    combined = np.zeros((img_hw[0], img_hw[1]), dtype=np.float64)
    for amap in maps:
        combined += cv2.resize(amap, viz_size, interpolation=cv2.INTER_NEAREST)
    if maps:
        combined /= len(maps)
    return combined


# ---------------------------------------------------------------------------
# Figure — class-conditioned native PCA with boundary class
# ---------------------------------------------------------------------------

def figure_pca_native_boundary(model, activations, img_display, img_hw,
                               fg_mask):
    """4 rows (all / breaking / boundary / background) × (pred | stages 0-3)."""
    feats, resolutions = _gather_stage_feats(model, activations)
    viz_size = (img_hw[1], img_hw[0])

    all_pca, fg_pca, bd_pca, bg_pca = [], [], [], []
    for feat in feats:
        if feat is not None:
            fg_cells, boundary, bg_cells = _fg_bg_boundary_cells(
                fg_mask, (feat.shape[2], feat.shape[3])
            )
            all_pca.append(_native_pca_all(feat))
            fg_pca.append(_native_pca_cells(feat, fg_cells))
            bd_pca.append(_native_pca_cells(feat, boundary))
            bg_pca.append(_native_pca_cells(feat, bg_cells))
        else:
            for lst in (all_pca, fg_pca, bd_pca, bg_pca):
                lst.append(None)

    n_rows, n_cols = 4, 1 + NUM_STAGES
    # Height derived from the image aspect so the grid packs with no dead
    # space in either direction (paired with set_box_aspect below).
    aspect = img_hw[0] / img_hw[1]
    col_w = PAGE_W / (n_cols + 0.15)      # 0.15 ≈ left ylabel share
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(PAGE_W, n_rows * col_w * aspect + 0.25),
                             sharex=True, sharey=True,
                             layout="constrained")

    row_labels = ["All pixels", "Foreground", "Boundary", "Background"]
    row_data = [all_pca, fg_pca, bd_pca, bg_pca]

    # Column 0: input, prediction overlays, and boundary-cell overlay
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title("Input", fontsize=FONTSIZE_LABEL, pad=2)

    pred_fg = _tint_overlay(img_display, fg_mask, PRED_TINT)
    axes[1, 0].imshow(pred_fg)

    bd_overlay = img_display.copy()
    finest = next((i for i, r in enumerate(resolutions) if r[0] > 0), None)
    if finest is not None:
        _, boundary_f, _ = _fg_bg_boundary_cells(fg_mask, resolutions[finest])
        bd_full = cv2.resize(boundary_f.astype(np.uint8), viz_size,
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        bd_overlay[bd_full] = bd_overlay[bd_full] * 0.5 + BOUNDARY_TINT * 0.5
    axes[2, 0].imshow(bd_overlay)

    pred_bg = _tint_overlay(img_display, ~fg_mask, PRED_TINT)
    axes[3, 0].imshow(pred_bg)

    for row in range(n_rows):
        axes[row, 0].set_ylabel(row_labels[row], fontsize=FONTSIZE_LABEL,
                                rotation=90, labelpad=6)

    for si in range(NUM_STAGES):
        col = 1 + si
        H, W = resolutions[si]
        ds = _downsample_factor(img_hw[0], H)
        title = f"Stage {si} (1/{ds})"       
        for row in range(n_rows):
            img = row_data[row][si]
            if img is not None:
                up = cv2.resize(img, viz_size, interpolation=cv2.INTER_NEAREST)
                axes[row, col].imshow(up)
            else:
                axes[row, col].text(0.5, 0.5, "N/A", ha="center", va="center")
            if row == 0:
                axes[row, col].set_title(title, fontsize=FONTSIZE_LABEL, pad=2)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_box_aspect(aspect)

    savefig(fig, OUT_DIR / "13_pca_native_boundary.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figures — separating-axis alignment (fg/bg and correct/incorrect)
# ---------------------------------------------------------------------------

def _plot_alignment_row(per_stage_maps, combined, img_display, img_hw,
                        fg_mask, out_name, pos_label, neg_label):
    """1 row: prediction | stages 0-3 | combined, PRGn on shared ±1.

    A single colorbar on the right serves all stage/combined panels; its
    ends are labelled ``pos_label`` (+) and ``neg_label`` (−).
    """
    viz_size = (img_hw[1], img_hw[0])
    labels = ["pred"] + [f"s{i}" for i in range(NUM_STAGES)] + ["comb"]

    # Figure height matched to the image aspect so the panels fill their axes
    # (no dead vertical space) and the shared colorbar reads the same height.
    aspect = img_hw[0] / img_hw[1]
    panel_w = PAGE_W / (len(labels) + 0.3)   # 0.3 ≈ colorbar + gutter share
    fig, ax_dict = plt.subplot_mosaic(
        [labels],
        figsize=(PAGE_W, panel_w * aspect + 0.35),
        layout="constrained",
    )
    # Reserve a right strip for the colorbar so the panels don't run to the
    # edge; the bar itself is placed as an inset on the last panel below.
    fig.get_layout_engine().set(rect=(0, 0, 0.92, 1))

    pred_overlay = _tint_overlay(img_display, fg_mask, PRED_TINT)
    ax_dict["pred"].imshow(pred_overlay)
    ax_dict["pred"].set_title("Prediction", fontsize=FONTSIZE_LEGEND, pad=2)

    im_ref = None
    for si in range(NUM_STAGES):
        ax = ax_dict[f"s{si}"]
        if si in per_stage_maps:
            amap = per_stage_maps[si]
            H, W = amap.shape
            up = cv2.resize(amap, viz_size, interpolation=cv2.INTER_NEAREST)
            im_ref = ax.imshow(up, cmap=ALIGN_CMAP, vmin=-1, vmax=1,
                               interpolation="nearest")
            ds = _downsample_factor(img_hw[0], H)
            ax.set_title(f"Stage {si} (1/{ds})", fontsize=FONTSIZE_LEGEND, pad=2)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")

    ax_comb = ax_dict["comb"]
    im_comb = ax_comb.imshow(combined, cmap=ALIGN_CMAP, vmin=-1, vmax=1,
                             interpolation="nearest")
    ax_comb.set_title("Combined", fontsize=FONTSIZE_LEGEND, pad=2)
    if im_ref is None:
        im_ref = im_comb

    for ax in ax_dict.values():
        ax.set_xticks([]); ax.set_yticks([])
        # Pin each axes box to the image aspect so the image fills it exactly
        # (no internal margins) — this is what lets the colorbar match height.
        ax.set_box_aspect(aspect)

    # Colorbar as an inset on the last panel, spanning y: 0→1 of that axes.
    # With the box_aspect above the image fills the axes, so the bar height
    # equals the panel/image height exactly.
    cax = ax_comb.inset_axes([1.12, 0.0, 0.14, 1.0])
    cb = fig.colorbar(im_ref, cax=cax)
    cb.set_ticks([-1, 1])
    cb.set_ticklabels([neg_label, pos_label])
    cb.ax.tick_params(labelsize=FONTSIZE_LEGEND)

    savefig(fig, OUT_DIR / out_name, facecolor="white", edgecolor="none")
    plt.close(fig)

def figure_attention_alignment(activations, img_display, img_hw, fg_mask):
    """Both alignment views from one set of stage-scaled projection maps.

    Axis view: sign = which class centroid Δ points toward (green = breaking),
    intensity = how strongly (in units of the stage's typical ‖Δ‖).
    Correct-class view: the same projection signed by each cell's own
    predicted class, so green always means "reinforces the classification"
    (FG cell pushed toward FG, or BG cell pushed toward BG) and purple means
    the attention residual opposes the cell's class.
    """
    stage_contributions = _gather_attention_contributions(activations)
    per_stage = _attention_class_alignment_axis(stage_contributions, fg_mask)

    axis_maps = {si: amap for si, (amap, _fg) in per_stage.items()}
    # Sign by own class: +proj on FG cells, −proj on BG cells
    correct_maps = {
        si: np.where(fg_cells, amap, -amap)
        for si, (amap, fg_cells) in per_stage.items()
    }

    _plot_alignment_row(
        axis_maps, _combine_maps(list(axis_maps.values()), img_hw),
        img_display, img_hw, fg_mask,
        "13_attention_alignment_axis.png",
        pos_label="breaking", neg_label="non-breaking",
    )
    _plot_alignment_row(
        correct_maps, _combine_maps(list(correct_maps.values()), img_hw),
        img_display, img_hw, fg_mask,
        "13_attention_alignment_correct.png",
        pos_label="reinforces", neg_label="opposes",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    apply_style()

    device = get_device()
    print(f"Device: {device}")

    print(f"Loading model: {RUN_DIR}")
    model, _ = load_checkpoint_model(RUN_DIR, device)

    print(f"Loading image: {IMAGE_PATH}")
    img_tensor, img_raw, img_hw = preprocess_image(IMAGE_PATH)
    img_display = denormalize(img_raw)

    # Single forward pass: the hooks record the MSCA activations while the
    # same logits give the prediction mask. A second forward would re-run the
    # Hamburger decode head, whose NMF bases are re-randomised on every call.
    print("Forward pass (collecting activations + prediction)...")
    hook = AttentionHook()
    hook.register_hooks(model)
    with torch.no_grad():
        logits = model(img_tensor.to(device))

    fg_mask = mask_from_logits(logits, img_hw)
    print(f"  Breaking: {fg_mask.sum()} / {fg_mask.size} px "
          f"({100 * fg_mask.mean():.1f}%)")

    acts = hook.activations

    print("Figure: boundary-class native PCA (13_pca_native_boundary)...")
    figure_pca_native_boundary(model, acts, img_display, img_hw, fg_mask)

    print("Figures: alignment axis + correct-class (13_attention_alignment_*)...")
    figure_attention_alignment(acts, img_display, img_hw, fg_mask)

    hook.remove_hooks()
    print("Done.")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
