"""Task-level Hamburger ablation against labelled masks: ΔIoU.

The simplest statement of what the NMF module contributes: bypass it
(output → ReLU(input), the same ablation hook as 19_erf.py and 21's
--bypass-iou mode) and score both models against the hand-labelled
breaking masks. One number — the IoU drop — anchors the appendix's
representation-geometry analysis (16/21/22) at the task level. This is
option 3 from the "simpler approach" discussion: no representation
statistics at all, just predictions vs labels.

Data
----
model/data/full_split: 512×500 timestacks with binary breaking masks from
CVAT polyline annotations (mask_<image>.png; 172 train / 43 validation).
Evaluation runs
on full frames with no cropping — unlike training, which uses 224-px
crops — so the absolute IoU here is not comparable to training-time
validation scores; only the Full-vs-bypass *difference* matters, and
both conditions see identical inputs. Default split: val (held out
from training).

Metrics (breaking class, production P > 0.48)
---------------------------------
- Dataset IoU per condition: intersection and union accumulated over
  all images before dividing (the standard semantic-segmentation IoU;
  robust to images with little breaking).
- Per-image IoU median + IQR per condition.
- Paired per-image ΔIoU (Full − bypass), median + IQR — the paired
  difference is the honest per-image statistic, as in 21's Fisher
  panel (c).
- Dataset precision / recall per condition, to show *how* the bypass
  fails (over- vs under-prediction), not just how much.

Observed (val split, 44 images, run 2026-07-14)
-----------------------------------------------
Dataset IoU 0.258 (Full) vs 0.257 (bypassed); paired ΔIoU median
+0.002 (IQR −0.003…+0.008) — the module is removable at *inference*
with no measurable accuracy cost, even though it moves ~14 % of the
predicted breaking pixels (21 --bypass-iou: agreement IoU 0.856).
Caveat: an inference-time bypass does not measure the module's
contribution during *training* (a model trained without it could
differ). The low absolute IoU / precision (0.30) with high recall
(0.65) reflects the thin polyline-derived label masks vs the model's
thick crest predictions — a protocol property affecting both
conditions identically, not a Full-vs-bypass signal.

Outputs
-------
  23_ham_ablation_iou.{pdf,png} — single panel, per-image IoU boxes,
  Full (filled) vs Ham bypassed (open dashed), styled as 21.
  Aggregate numbers print to the console.

Run:
    python figures/ml-exploration/23_ham_ablation_iou.py [--fast]
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from common.normalization import MEAN as IMAGENET_MEAN
from common.normalization import STD as IMAGENET_STD
from exploration_common import (
    DEFAULT_DATASET,
    ML_DIR,
    PRODUCTION_THRESHOLD,
    add_runtime_arguments,
    import_figure,
    load_runtime,
)
from common.figure_style import COL_W, IBM, apply_style, savefig

# Figure 21 provides the shared box styling.
ham21 = import_figure("21_ham_variability_pooled")

OUT_DIR = ML_DIR
DATASET_ROOT = DEFAULT_DATASET

N_FAST = 4


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


def load_pair(img_path, mask_path):
    """Load one normalized image tensor and its boolean ground-truth mask."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt is None:
        raise FileNotFoundError(mask_path)
    if gt.shape != img.shape[:2]:
        raise ValueError(f"mask/image size mismatch for {img_path.name}")

    img01 = torch.from_numpy(img.transpose(2, 0, 1))          # (3,H,W) [0,1]
    m = torch.from_numpy((gt > 127).astype(np.float32))[None]  # (1,H,W)
    img_tensor = ((img01 - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0)
    gt_bool = m.squeeze(0).numpy() > 0.5
    return img_tensor, gt_bool, (img01.shape[1], img01.shape[2])


def predict(model, device, img_tensor, img_hw, ham_bypass):
    """Breaking mask for one image, optionally with the Ham bypassed."""
    hook = None
    if ham_bypass:
        hook = model.decode_head.hamburger.register_forward_hook(
            lambda _m, inp, _o: F.relu(inp[0], inplace=False))
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    if hook is not None:
        hook.remove()
    if prob.shape != img_hw:      # output frame must equal the input size
        prob = cv2.resize(prob, (img_hw[1], img_hw[0]),
                          interpolation=cv2.INTER_NEAREST)
    return prob > PRODUCTION_THRESHOLD


def main():
    global DATASET_ROOT, OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_runtime_arguments(parser)
    parser.add_argument("--split", default="val",
                        choices=("val", "train", "all"),
                        help="dataset split to evaluate (default: val)")
    parser.add_argument("--fast", action="store_true",
                        help=f"use only {N_FAST} images (smoke test)")
    parser.add_argument(
        "--dataset-root", type=Path, default=DEFAULT_DATASET,
        help="dataset containing image_mask_mapping.csv, images/, and masks/",
    )
    args = parser.parse_args()

    apply_style()
    runtime = load_runtime(args)
    OUT_DIR = runtime.output_dir
    DATASET_ROOT = args.dataset_root.expanduser().resolve()

    samples = load_samples(args.split)
    if args.fast:
        samples = samples[:N_FAST]
    print(
        f"Evaluating {len(samples)} clean images "
        f"({args.split} split) from {DATASET_ROOT}"
    )

    conditions = ("Full", "Ham bypassed")
    inter = {c: 0 for c in conditions}      # dataset-level accumulators
    union = {c: 0 for c in conditions}
    tp = {c: 0 for c in conditions}
    fp = {c: 0 for c in conditions}
    fn = {c: 0 for c in conditions}
    per_image = {c: [] for c in conditions}

    for i, (img_path, mask_path) in enumerate(samples):
        img_tensor, gt, img_hw = load_pair(img_path, mask_path)
        line = f"  [{i + 1}/{len(samples)}] {img_path.name}:"
        for cond in conditions:
            pred = predict(
                runtime.model,
                runtime.device,
                img_tensor,
                img_hw,
                ham_bypass=(cond == "Ham bypassed"),
            )
            n_i = int(np.logical_and(pred, gt).sum())
            n_u = int(np.logical_or(pred, gt).sum())
            inter[cond] += n_i
            union[cond] += n_u
            tp[cond] += n_i
            fp[cond] += int(np.logical_and(pred, ~gt).sum())
            fn[cond] += int(np.logical_and(~pred, gt).sum())
            per_image[cond].append(n_i / n_u if n_u else 1.0)
            line += f"  {cond} IoU {per_image[cond][-1]:.3f}"
        delta = per_image["Full"][-1] - per_image["Ham bypassed"][-1]
        print(line + f"  (Δ {delta:+.3f})")

    # ── Aggregate ──
    print(
        f"\nBreaking-class scores, {args.split} split "
        f"({len(samples)} clean images):"
    )
    for cond in conditions:
        iou_ds = inter[cond] / max(union[cond], 1)
        prec = tp[cond] / max(tp[cond] + fp[cond], 1)
        rec = tp[cond] / max(tp[cond] + fn[cond], 1)
        pi = np.array(per_image[cond])
        print(f"  {cond:13s}: dataset IoU {iou_ds:.3f}  "
              f"precision {prec:.3f}  recall {rec:.3f}  "
              f"per-image IoU median {np.median(pi):.3f} "
              f"(IQR {np.percentile(pi, 25):.3f}–"
              f"{np.percentile(pi, 75):.3f})")
    diff = np.array(per_image["Full"]) - np.array(per_image["Ham bypassed"])
    print(f"  paired ΔIoU (Full − bypassed): median {np.median(diff):+.3f} "
          f"(IQR {np.percentile(diff, 25):+.3f}–"
          f"{np.percentile(diff, 75):+.3f}); dataset ΔIoU "
          f"{inter['Full'] / max(union['Full'], 1) - inter['Ham bypassed'] / max(union['Ham bypassed'], 1):+.3f}")
    pos = float((diff > 0).mean())
    p = wilcoxon(diff).pvalue
    p_note = f", Wilcoxon signed-rank p = {p:.2g}"
    print(f"  per-image mean ΔIoU: {diff.mean():+.4f}; "
          f"{100 * pos:.0f}% of images favour Full{p_note}")

    # Paired lines preserve the experimental design; box summaries sit on top.
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85),
                           layout="constrained")
    full = np.asarray(per_image["Full"])
    bypassed = np.asarray(per_image["Ham bypassed"])
    for left, right in zip(full, bypassed):
        ax.plot([0.6, 1.4], [left, right], color="0.75", linewidth=0.5,
                alpha=0.55, zorder=1)
    ax.scatter(np.full_like(full, 0.6), full, s=8, color=IBM[0], alpha=0.45,
               linewidths=0, zorder=2)
    ax.scatter(np.full_like(bypassed, 1.4), bypassed, s=8,
               facecolors="white", edgecolors=IBM[0], alpha=0.55,
               linewidths=0.5, zorder=2)
    ham21._box(ax, 0.6, per_image["Full"], IBM[0], filled=True)
    ham21._box(ax, 1.4, per_image["Ham bypassed"], IBM[0], filled=False)
    ax.set_xticks([0.6, 1.4], ["Full", "Ham bypassed"])
    ax.set_xlim(0, 2.0)
    upper = min(1.0, max(0.4, 1.12 * max(full.max(), bypassed.max())))
    ax.set_ylim(0, upper)
    ax.set_ylabel("Breaking IoU")
    ax.text(
        0.5, 0.98, f"median paired ΔIoU = {np.median(diff):+.3f}",
        transform=ax.transAxes, ha="center", va="top",
    )
    savefig(
        fig,
        OUT_DIR / "23_ham_ablation_iou.png",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
