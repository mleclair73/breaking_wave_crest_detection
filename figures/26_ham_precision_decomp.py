"""Appendix figure — how the Hamburger earns its precision: a false-
positive decomposition of the Full-vs-bypass predictions.

A paired ablation shows that bypassing the module (output → ReLU(input))
reduces validation IoU, driven by precision at a slight recall cost. This
figure shows *what* the module changes to buy that precision, and — crucially
— whether it does so by removing whole spurious detections or merely by
shaving mask boundaries.

Inference. Both conditions are evaluated with the ablation study's full-frame
tiled policy: the frame is covered
by overlapping 224-px tiles — the training crop size — stitched with
Hanning blending, so every pixel is predicted at the scale the model was
trained on and the whole frame is scored at once. The same frame is
scored for both conditions (paired). The Ham-bypass hook wraps the
prediction so it applies to every tile forward pass.

Both predictions are scored against the hand-labelled mask into true
positives, false positives and false negatives (the ~99.5 %-background
true-negative cell is uninformative and omitted), summed over the whole
split. The precision story is the false-positive count: Full carries
fewer FP pixels than the bypass at near-equal TP, so the gain is
false-positive suppression, not a recall trade.

Spurious-detection count. Predicted-breaking connected components are
matched to the (dilated) GT: a component touching a labelled crest is a
true detection, one touching none is spurious (a hallucinated crest).
Totalling spurious components over the split, Full vs bypass, tests the
mechanism directly — fewer spurious blobs under Full means the module
suppresses whole false detections, not boundary pixels. Dilation
tolerates the thin polyline labels vs the model's thick crests.

Panels
------
(a) Total TP / FP / FN pixels over the split, Full vs Ham bypassed — the
    lower FP bar under Full is the precision gain.
(b) Per-image false-positive crest detections — predicted-breaking connected
    components overlapping no labelled crest — bypass vs Full.

Output
------
  26_ham_precision_decomp.{pdf,png}

Run after installing the repository environment:
    python figures/26_ham_precision_decomp.py [--split val|train|all] [--fast]
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch.nn.functional as F

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from common.figure_style import IBM, PAGE_W, apply_style, panel_label, savefig, styled_legend
from common.module_loading import load_module_from_path

ham16 = load_module_from_path(
    "_breaking_wave_figure_16_hamburger_viz",
    Path(__file__).with_name("16_hamburger_viz.py"),
)
from checkpoint import load_checkpoint_model
from tiling import predict_tiled
from segmentation.predict_video import get_device

OUT_DIR = Path(__file__).parent
DATASET_ROOT = (Path(__file__).resolve().parent.parent
                / "model/data/full_split")

N_FAST = 6
PATCH_SIZE = 224        # tile size = training crop size
OVERLAP = 86            # complete-image evaluation policy (tile_overlap)
GT_DILATE = 5           # px; tolerate thin polyline labels when matching blobs


def load_samples(split):
    """(image_path, mask_path) pairs from the mapping CSV."""
    samples = []
    with open(DATASET_ROOT / "image_mask_mapping.csv") as f:
        for row in csv.DictReader(f):
            if split != "all" and row["split"] != split:
                continue
            samples.append((DATASET_ROOT / "images" / row["image_name"],
                            DATASET_ROOT / "masks" / row["mask_name"]))
    if not samples:
        raise FileNotFoundError(f"no samples for split '{split}' in "
                                f"{DATASET_ROOT}")
    return samples


def _load(img_path, mask_path):
    """Full raw RGB image in [0, 1] (H, W, 3) and boolean GT mask."""
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127
    return img, gt


def _predict(model, device, img01, ham_bypass):
    """Full-frame tiled breaking mask, optionally bypassing the Hamburger."""
    hook = None
    if ham_bypass:
        hook = model.decode_head.hamburger.register_forward_hook(
            lambda _m, inp, _o: F.relu(inp[0], inplace=False))
    try:
        prob = predict_tiled(
            model,
            img01,
            patch_size=PATCH_SIZE,
            overlap_y=OVERLAP,
            overlap_x=OVERLAP,
            device=device,
        )
    finally:
        if hook is not None:
            hook.remove()
    return prob > ham16.PRODUCTION_THRESHOLD


def _spurious_count(pred, gt_dil):
    """Number of predicted-breaking connected components that touch no
    labelled crest (hallucinated detections)."""
    n, labels = cv2.connectedComponents(pred.astype(np.uint8))
    spurious = 0
    for lab in range(1, n):
        if not gt_dil[labels == lab].any():
            spurious += 1
    return spurious


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", default="val",
                        choices=("val", "train", "all"))
    parser.add_argument("--fast", action="store_true",
                        help=f"use only {N_FAST} images (smoke test)")
    args = parser.parse_args()

    apply_style()
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading model: {ham16.RUN_DIR}")
    model, _ = load_checkpoint_model(ham16.RUN_DIR, device)

    samples = load_samples(args.split)
    if args.fast:
        samples = samples[:N_FAST]
    print(f"Decomposing {len(samples)} images ({args.split} split), "
          f"full-frame tiled ({PATCH_SIZE}px / {OVERLAP}px overlap)")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (GT_DILATE, GT_DILATE))
    conf = {c: {"TP": 0, "FP": 0, "FN": 0} for c in ("Full", "bypass")}
    # Per-image false-positive crest-detection counts, so panel (b) can show
    # the distribution over images (boxplot) rather than a single total.
    spur_img = {"Full": [], "bypass": []}
    for i, (img_path, mask_path) in enumerate(samples):
        img01, gt = _load(img_path, mask_path)
        gt_dil = cv2.dilate(gt.astype(np.uint8), kernel).astype(bool)
        fp_img = {}
        for cond, byp_flag in (("Full", False), ("bypass", True)):
            pred = _predict(model, device, img01, ham_bypass=byp_flag)
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            conf[cond]["TP"] += tp
            conf[cond]["FP"] += fp
            conf[cond]["FN"] += fn
            spur_img[cond].append(_spurious_count(pred, gt_dil))
            fp_img[cond] = fp
        print(f"  [{i + 1}/{len(samples)}] {img_path.name}: "
              f"FP {fp_img['bypass']}→{fp_img['Full']}")

    def _dataset_pr(c):
        tp, fp, fn = (conf[c][k] for k in ("TP", "FP", "FN"))
        return tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    print()
    for c in ("bypass", "Full"):
        p, r = _dataset_pr(c)
        print(f"  {c:7s}: precision {p:.3f}  recall {r:.3f}  "
              f"(total FP {conf[c]['FP']})")
    spur_tot = {c: int(np.sum(spur_img[c])) for c in ("bypass", "Full")}
    spur_med = {c: float(np.median(spur_img[c])) for c in ("bypass", "Full")}
    print(f"  false-positive crest detections (total): bypass "
          f"{spur_tot['bypass']} → Full {spur_tot['Full']}  "
          f"(median/image {spur_med['bypass']:.1f} → {spur_med['Full']:.1f})")

    # ── Figure: 1 × 2 — (a) confusion totals, (b) per-image spurious boxplot ──
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(PAGE_W * 0.62, PAGE_W * 0.30), layout="constrained")

    # (a) confusion totals: TP / FP / FN pixels summed over the split,
    # Full vs Ham bypassed — the FP bar is the precision story
    ax = ax_a
    cats = ["TP", "FP", "FN"]
    x = np.arange(len(cats))
    full_v = [conf["Full"][k] for k in cats]
    byp_v = [conf["bypass"][k] for k in cats]
    ax.bar(x - 0.2, byp_v, width=0.38, color=IBM[3], label="Ham bypassed")
    ax.bar(x + 0.2, full_v, width=0.38, color=IBM[0], label="Full")
    ax.set_xticks(x, cats)
    ax.set_ylim(0, max(max(full_v), max(byp_v)) * 1.30)
    ax.set_ylabel("Total pixels")
    styled_legend(ax, loc="upper right")
    panel_label(ax, "a")

    # (b) per-image false-positive crest-detection counts as a boxplot,
    # bypass (orange) then Full (blue) — same colours and order as (a).
    # The Hamburger's precision gain is that its distribution sits lower.
    ax = ax_b
    positions = [0.6, 1.4]
    box_colors = [IBM[3], IBM[0]]           # bypass orange, Full blue
    data = [spur_img["bypass"], spur_img["Full"]]
    edge_lw = 0.7                            # box/whisker/cap edges (== median)
    bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", linewidth=0.7),
                    whiskerprops=dict(color="black", linewidth=edge_lw),
                    capprops=dict(color="black", linewidth=edge_lw),
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="black",
                                    markeredgecolor="none"))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(edge_lw)
    ax.set_xticks(positions, ["Ham\nbypassed", "Full"])
    ax.set_xlim(0, 2.0)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("False-positive crest\ndetections / image")
    panel_label(ax, "b")

    savefig(fig, OUT_DIR / "26_ham_precision_decomp.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
