"""Scree plot of the Hamburger module's input vs output feature space.

Cumulative PCA explained variance vs number of components, fit separately
on breaking and background cells, for the Hamburger input (solid, filled
markers) vs output (dashed, open markers). Curves are the mean over
N_IMAGES images, bands ±1 std. Structure packed into fewer components ⇒
the module built a lower-dimensional representation of that class.

Run after installing the repository environment:
    python figures/21_ham_variability.py [--fast]
"""

import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from common.figure_style import PAGE_W, apply_style, savefig, styled_legend
from common.module_loading import load_module_from_path

ham16 = load_module_from_path(
    "_breaking_wave_figure_16_hamburger_viz",
    Path(__file__).with_name("16_hamburger_viz.py"),
)
from segmentation.predict_video import get_device
from checkpoint import load_checkpoint_model

OUT_DIR = Path(__file__).parent
IMAGE_DIR = ham16.IMAGE_DIR
N_IMAGES = 128
N_FAST = 4
N_SCREE = ham16.N_SCREE
FG_COL, BG_COL = ham16.FG_COL, ham16.BG_COL
CROP = 224


def preprocess_crop(path):
    """Return a normalized, bottom-centered training-size crop."""
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - ham16.IMAGENET_MEAN) / ham16.IMAGENET_STD
    height, width = image.shape[:2]
    top = max(0, height - CROP)
    left = max(0, (width - CROP) // 2)
    crop = image[top:top + CROP, left:left + CROP]
    tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor, crop.shape[:2]


def hooked_crop_features(model, device, path):
    """Return predicted classes and Hamburger features for one crop."""
    image, crop_hw = preprocess_crop(path)
    hook = ham16.HamburgerIOHook()
    hook.register(model)
    with torch.no_grad():
        logits = model(image.to(device))
        probability = torch.softmax(logits, dim=1)[:, 1].cpu().squeeze().numpy()
    hook.remove()
    if probability.shape != crop_hw:
        probability = cv2.resize(
            probability,
            (crop_hw[1], crop_hw[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    foreground_mask = probability > ham16.PRODUCTION_THRESHOLD

    channels, height, width = hook.inp.shape[1:]
    features_in = hook.inp[0].permute(1, 2, 0).reshape(-1, channels).numpy()
    features_out = hook.out[0].permute(1, 2, 0).reshape(-1, channels).numpy()
    foreground = ham16._downsample_mask(
        foreground_mask, (height, width)
    ).flatten()
    if min(foreground.sum(), (~foreground).sum()) <= N_SCREE:
        return None
    return foreground, features_in, features_out


def _select_images(n):
    all_imgs = sorted(IMAGE_DIR.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"no images in {IMAGE_DIR}")
    idx = np.unique(np.linspace(0, len(all_imgs) - 1, n).astype(int))
    return [all_imgs[i] for i in idx]


def _scree(feat, m):
    n_pc = int(min(N_SCREE, feat.shape[1], m.sum() - 1))
    return np.cumsum(PCA(n_components=n_pc).fit(feat[m])
                     .explained_variance_ratio_)


def image_scree(model, device, path):
    """Per-image scree curves for both classes, Ham input and output, from
    a bottom-centred 224 crop; None if either class has too few cells."""
    res = hooked_crop_features(model, device, path)
    if res is None:
        return None
    fg, feat_in, feat_out = res

    out = {}
    for cls, m in (("Brk", fg), ("Bg", ~fg)):
        for which, feat in (("in", feat_in), ("out", feat_out)):
            out[(cls, which)] = _scree(feat, m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fast", action="store_true",
                        help=f"use only {N_FAST} images")
    args = parser.parse_args()

    apply_style()
    device = get_device()
    print(f"Device: {device}")
    model, _ = load_checkpoint_model(ham16.RUN_DIR, device)

    paths = _select_images(N_FAST if args.fast else N_IMAGES)
    print(f"Selected {len(paths)} images from {IMAGE_DIR}")

    curves = []
    for i, path in tqdm(enumerate(paths)):
        s = image_scree(model, device, path)
        if s is None:
            print(f"  [{i + 1}/{len(paths)}] {path.name}: skipped")
            continue
        print(f"  [{i + 1}/{len(paths)}] {path.name}")
        curves.append(s)
    n = len(curves)
    if n < 2:
        raise RuntimeError(f"only {n} usable images — need at least 2")
    print(f"Usable images: {n}/{len(paths)}")

    fig, ax = plt.subplots(figsize=(PAGE_W * 0.4, PAGE_W * 0.33),
                           layout="constrained")
    stagger = 0
    for cls, col in (("Brk", FG_COL), ("Bg", BG_COL)):
        for which, ls in (("in", "-"), ("out", "--")):
            arr = np.stack([c[(cls, which)] for c in curves])
            mean, std = arr.mean(axis=0), arr.std(axis=0)
            x = np.arange(1, arr.shape[1] + 1)
            ax.plot(x, mean, color=col, linestyle=ls, marker="o",
                    markersize=2.6, markevery=(stagger, 4),
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
               Line2D([], [], color=BG_COL, label="Background"),
               Line2D([], [], color="0.3", linestyle="-", label="Input"),
               Line2D([], [], color="0.3", linestyle="--", label="Output")]
    styled_legend(ax, handles=handles, loc="lower right")

    savefig(fig, OUT_DIR / "21_ham_variability.png",
            facecolor="white", edgecolor="none")
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
