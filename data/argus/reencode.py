#!/usr/bin/env python3
"""
Batch re-encode AVI files to FFV1/MKV grayscale in a destination directory.
Verifies each output is pixel-identical to the source.
This reduces data size 25% from yuv8 grayscale avi, more from color avi.
This will be used to re-encode the data for archive on dryad

It can be undone with 
ffmpeg -i input.mkv -c:v png -pix_fmt gray output.avi

"""

import subprocess
import argparse
import sys
from pathlib import Path


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def encode_ffv1(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-c:v", "ffv1",
        "-level", "3",
        "-coder", "1",
        "-context", "1",
        "-g", "1",
        "-slices", "16",
        "-slicecrc", "1",
        "-pix_fmt", "gray",
        "-c:a", "copy",
        "-stats",
        str(dst),
    ]
    result = subprocess.run(cmd, stderr=None)  # let stderr flow to terminal
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)


def get_frame_checksums(path: Path) -> list[str]:
    """Decode video to raw frames and return per-frame MD5 checksums."""
    result = subprocess.run([
        "ffmpeg", "-i", str(path),
        "-f", "framemd5",
        "-",
    ], capture_output=True, text=True)
    # framemd5 lines look like:
    # 0, 0, 0, 1, 800000, b5fb0e781ca6ab03a3ff251f932e3817
    checksums = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) >= 6:
            checksums.append(parts[-1].strip())
    return checksums


def verify_lossless(src: Path, dst: Path) -> bool:
    src_sums = get_frame_checksums(src)
    dst_sums = get_frame_checksums(dst)

    if not src_sums or not dst_sums:
        print(f"  [debug] empty checksums: src={len(src_sums)} dst={len(dst_sums)}")
        return False

    if len(src_sums) != len(dst_sums):
        print(f"  [debug] frame count mismatch: src={len(src_sums)} dst={len(dst_sums)}")
        return False

    mismatches = [i for i, (a, b) in enumerate(zip(src_sums, dst_sums)) if a != b]
    if mismatches:
        print(f"  [debug] {len(mismatches)} mismatched frames: {mismatches[:5]}...")
        return False

    return True


def fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024**2:.1f} MB"


def process_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if dst.exists():
        print(f"  [skip] already exists: {dst.name}")
        return True

    print(f"  src : {src.name} ({fmt_mb(src)})")

    if dry_run:
        print(f"  [dry] would encode -> {dst}")
        return True

    print("  encoding ...", flush=True)
    try:
        encode_ffv1(src, dst)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] encoding failed:\n{e.stderr}")
        return False

    savings = 100 * (1 - dst.stat().st_size / src.stat().st_size)
    print(f"  dst : {dst.name} ({fmt_mb(dst)}, {savings:.1f}% smaller)")
    print("  verifying ...", flush=True)

    if not verify_lossless(src, dst):
        print("  [ERROR] verification FAILED — deleting output")
        dst.unlink()
        return False

    print("  verified ✓")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch re-encode AVI -> FFV1/MKV")
    parser.add_argument("src_dir", type=Path, help="Source directory containing AVI files")
    parser.add_argument("dst_dir", type=Path, help="Destination directory for MKV files")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without encoding")
    args = parser.parse_args()

    if not args.src_dir.is_dir():
        print(f"Error: {args.src_dir} is not a directory")
        sys.exit(1)

    args.dst_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.avi" if args.recursive else "*.avi"
    files = sorted(args.src_dir.glob(pattern))

    if not files:
        print(f"No AVI files found in {args.src_dir}")
        sys.exit(0)

    print(f"Found {len(files)} AVI file(s)")
    print(f"  src : {args.src_dir}")
    print(f"  dst : {args.dst_dir}")
    if args.dry_run:
        print("  [dry-run — no files will be modified]")
    print()

    ok, failed = 0, []
    for i, src in enumerate(files, 1):
        # Mirror subdirectory structure if recursive
        rel = src.relative_to(args.src_dir)
        dst = (args.dst_dir / rel).with_suffix(".mkv")
        dst.parent.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(files)}] {rel}")
        if process_file(src, dst, dry_run=args.dry_run):
            ok += 1
        else:
            failed.append(src)
        print()

    print(f"Done: {ok}/{len(files)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}):")
        for f in failed:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
