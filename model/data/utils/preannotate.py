#!/usr/bin/env python3
"""Write CVAT 1.1 preannotations (model-suggested `breaker` polylines) for a folder of crops.

Tiled inference -> threshold -> `cvat_io.mask_to_polylines` -> CVAT XML. Output is a
review aid, not ground truth. Run from `model/`:

    python data/utils/preannotate.py data/ood_test/images out.xml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(MODEL_ROOT))
from checkpoint import get_device, load_checkpoint_model  # noqa: E402
from cvat_io import CvatImage, mask_to_polylines, write_cvat_xml  # noqa: E402
from tiling import predict_tiled  # noqa: E402

DEFAULT_RUN = MODEL_ROOT / "outputs" / "released" / "segnext_t_learned_up_skip_4_2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", type=Path, help="folder of RGB crops (*.png)")
    parser.add_argument("output", type=Path, help="CVAT XML to write")
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN, help="checkpoint dir with best_model.pth")
    parser.add_argument("--threshold", type=float, default=None, help="default: run's thresholds.json")
    parser.add_argument("--overlap", type=int, default=None, help="tile overlap px; default: run's tile_overlap")
    parser.add_argument("--min-size", type=int, default=20, help="drop components smaller than this (px)")
    parser.add_argument("--epsilon", type=float, default=1.0, help="RDP simplification tolerance (px)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-probabilities", type=Path, default=None, help="optional dir for *.npy maps")
    args = parser.parse_args()

    device = get_device(args.device)
    model, config = load_checkpoint_model(args.run / "best_model.pth", device)
    threshold = args.threshold
    if threshold is None:
        threshold = float(next(iter(json.loads((args.run / "thresholds.json").read_text()).values())))
    overlap = args.overlap if args.overlap is not None else int(config["tile_overlap"])
    patch = int(config["image_size"])

    paths = sorted(args.images.glob("*.png"))
    if not paths:
        sys.exit(f"no PNGs in {args.images}")
    images = []
    for image_id, path in enumerate(paths):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        with torch.no_grad():
            probability = predict_tiled(model, rgb, patch_size=patch, overlap_y=overlap, overlap_x=overlap,
                                        device=device)
        if args.save_probabilities:
            args.save_probabilities.mkdir(parents=True, exist_ok=True)
            np.save(args.save_probabilities / f"{path.stem}.npy", probability)
        lines = mask_to_polylines(probability >= threshold, min_size=args.min_size, epsilon=args.epsilon)
        images.append(CvatImage(image_id, path.name, rgb.shape[1], rgb.shape[0], lines))
    write_cvat_xml(images, args.output, include_metadata=False)
    print(f"wrote {sum(len(i.polylines) for i in images)} polylines for {len(images)} images to {args.output} "
          f"(threshold {threshold}, patch {patch}, overlap {overlap}, {device})")


if __name__ == "__main__":
    main()
