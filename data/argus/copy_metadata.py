#!/usr/bin/env python3
"""
Copy metadata pickle files matching encoded MKV files to a destination directory.
Matches on the timestamp prefix: ArgusFF_<timestamp>_*.
Make sure we have the associated metadata files

For reference this is how you'd convert back to avi ffmpeg -i input.mkv -c:v png -pix_fmt gray output.avi

"""

import shutil
import argparse
import sys
from pathlib import Path


def extract_timestamp(name: str) -> str | None:
    """Extract 'ArgusFF_<timestamp>' prefix from a filename."""
    parts = name.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None


def main():
    parser = argparse.ArgumentParser(description="Copy metadata files matching encoded MKVs")
    parser.add_argument("mkv_dir", type=Path, help="Directory containing encoded MKV files")
    parser.add_argument("metadata_dir", type=Path, help="Source metadata directory")
    parser.add_argument("dst_dir", type=Path, help="Destination directory for metadata")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for d in (args.mkv_dir, args.metadata_dir):
        if not d.is_dir():
            print(f"Error: {d} is not a directory")
            sys.exit(1)

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    # build timestamp -> mkv mapping
    mkv_timestamps = {
        extract_timestamp(f.name)
        for f in args.mkv_dir.glob("*.mkv")
        if extract_timestamp(f.name)
    }

    print(f"Found {len(mkv_timestamps)} MKV timestamps in {args.mkv_dir}")

    all_meta = list(args.metadata_dir.glob("*.pickle"))
    matched, skipped, missing = [], [], []

    for meta in sorted(all_meta):
        ts = extract_timestamp(meta.name)
        if ts in mkv_timestamps:
            matched.append(meta)
        else:
            skipped.append(meta)

    # warn about MKV timestamps with no metadata
    meta_timestamps = {extract_timestamp(f.name) for f in all_meta}
    for ts in sorted(mkv_timestamps):
        if ts not in meta_timestamps:
            missing.append(ts)

    print(f"Matched: {len(matched)}  |  No MKV match: {len(skipped)}  |  MKVs missing metadata: {len(missing)}")
    if args.dry_run:
        print("  [dry-run — no files will be copied]")
    print()

    for meta in matched:
        dst = args.dst_dir / meta.name
        if dst.exists():
            print(f"  [skip] {meta.name}")
            continue
        print(f"  {'[dry] ' if args.dry_run else ''}copy {meta.name}")
        if not args.dry_run:
            shutil.copy2(meta, dst)

    if missing:
        print("\nWarning — no metadata found for:")
        for ts in missing:
            print(f"  {ts}")


if __name__ == "__main__":
    main()
