#!/usr/bin/env python3
"""Rebuild masks/ from annotations_canonical.xml and compare pixels to the stored masks.

Default is check-only (exit 1 on any mismatch). `--write` rewrites the PNGs;
that changes file bytes and therefore `dataset_sha256` in verification.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "utils"))
from cvat_io import rasterize_polylines, read_cvat_xml, to_mask_image  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--write", action="store_true", help="overwrite masks/ with the rasterized XML")
    args = parser.parse_args()

    images = read_cvat_xml(args.root / "annotations_canonical.xml")
    mismatched = []
    for image in images:
        mask = to_mask_image(rasterize_polylines(image.polylines, (image.height, image.width)))
        path = args.root / "masks" / f"mask_{image.name}"
        if args.write:
            Image.fromarray(mask, mode="L").save(path)
        elif not path.exists() or not np.array_equal(np.asarray(Image.open(path)), mask):
            mismatched.append(path.name)
    if args.write:
        print(f"wrote {len(images)} masks")
        return
    if mismatched:
        sys.exit(f"{len(mismatched)}/{len(images)} masks differ from the XML: {mismatched[:5]}")
    print(f"all {len(images)} masks match annotations_canonical.xml")


if __name__ == "__main__":
    main()
