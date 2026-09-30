#!/usr/bin/env python3
"""Check train/val/test recordings in split_manifest.csv against the Stage-1
inference video pool (data/video_dataset/argus/).

Reports, per split, recordings that overlap the pool by exact acquisition
(same recording) and by day. The dataset invariant is that **no training
recording appears verbatim in the pool** (training must be disjoint from the
videos the model is later run on); the script exits non-zero if that is
violated.

Usage:
    python check_pool_overlap.py
    python check_pool_overlap.py --manifest <path> --pool <dir>
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "split_manifest.csv"
DEFAULT_POOL = HERE.parents[2] / "data" / "video_dataset" / "argus"
SPLITS = ("train", "val", "test")
STAMP = re.compile(r"\d{8}T\d{6}Z")


def stamp(text: str) -> str | None:
    """Acquisition stamp (e.g. 20211026T160100Z) from a source id / filename."""
    match = STAMP.search(text)
    return match.group(0) if match else None


def load_splits(manifest: Path) -> dict[str, dict[str, int]]:
    """{split: {acquisition_stamp: crop_count}} from the manifest."""
    out: dict[str, dict[str, int]] = {s: defaultdict(int) for s in SPLITS}
    with manifest.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            s = stamp(row["source_video"])
            if s is None:
                raise ValueError(f"no acquisition stamp in source_video: {row['source_video']!r}")
            out[row["split"]][s] += 1
    return out


def load_pool(pool: Path) -> set[str]:
    if not pool.is_dir():
        raise FileNotFoundError(f"inference pool not found: {pool}")
    return {s for f in pool.glob("*.avi") if (s := stamp(f.name))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    args = ap.parse_args()

    splits = load_splits(args.manifest)
    pool_stamps = load_pool(args.pool)
    pool_days = {s[:8] for s in pool_stamps}

    print(f"manifest : {args.manifest}")
    print(f"pool     : {args.pool}  ({len(pool_stamps)} videos, {len(pool_days)} days)\n")

    train_exact_leak: list[str] = []
    for split in SPLITS:
        recs = splits[split]
        exact = sorted(s for s in recs if s in pool_stamps)
        same_day = sorted(s for s in recs if s[:8] in pool_days and s not in pool_stamps)
        print(f"[{split}] {len(recs)} recordings, {sum(recs.values())} crops")
        print(f"    exact recording in pool : {len(exact)}"
              + (f"  {exact}" if exact else ""))
        print(f"    same-day (not exact)    : {len(same_day)}"
              + (f"  {same_day}" if same_day else ""))
        for s in exact:
            print(f"      ! {split} {s} ({recs[s]} crops) is verbatim in the inference pool")
        if split == "train":
            train_exact_leak = exact
        print()

    if train_exact_leak:
        print(f"FAIL: {len(train_exact_leak)} training recording(s) overlap the inference pool "
              f"verbatim: {train_exact_leak}")
        return 1
    print("PASS: no training recording appears in the inference pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
