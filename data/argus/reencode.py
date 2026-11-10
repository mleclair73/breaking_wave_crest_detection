#!/usr/bin/env python3
"""
Batch re-encode AVI files to FFV1/MKV grayscale in a destination directory.
Verifies each output is pixel-identical to the source before atomically moving
it into place.
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


def encode_ffv1(src: Path, dst: Path, fps: float | None = None) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
    ]
    if fps is not None:
        # Preserve every frame while assigning timestamps at the acquisition
        # cadence. Merely passing -r would drop frames when an input container
        # has the wrong nominal rate.
        cmd.extend(["-vf", f"setpts=N/({fps}*TB)", "-r", str(fps)])
    cmd.extend([
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
    ])
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


def get_frame_rate(path: Path) -> float:
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0",
        str(path),
    ])
    num, den = map(int, result.stdout.strip().split("/"))
    return num / den


def fmt_mb(path: Path) -> str:
    return f"{path.stat().st_size / 1024**2:.1f} MB"


def process_file(src: Path, dst: Path, dry_run: bool, fps: float | None = None) -> bool:
    if dst.exists():
        print(f"  verifying existing output: {dst.name}", flush=True)
        rate_ok = fps is None or get_frame_rate(dst) == fps
        if verify_lossless(src, dst) and rate_ok:
            print("  [skip] existing output verified ✓")
            return True
        print("  [warning] existing output is incomplete or differs; rebuilding")

    print(f"  src : {src.name} ({fmt_mb(src)})")

    if dry_run:
        print(f"  [dry] would encode -> {dst}")
        return True

    # ffmpeg can leave a playable but truncated file when it is interrupted.
    # Encode to a sibling temporary MKV and expose the final name only after the
    # full pixel-level verification succeeds.
    tmp = dst.with_name(f".{dst.stem}.tmp{dst.suffix}")
    if tmp.exists():
        tmp.unlink()

    print("  encoding ...", flush=True)
    try:
        encode_ffv1(src, tmp, fps=fps)
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] encoding failed:\n{e.stderr}")
        tmp.unlink(missing_ok=True)
        return False

    savings = 100 * (1 - tmp.stat().st_size / src.stat().st_size)
    print(f"  dst : {dst.name} ({fmt_mb(tmp)}, {savings:.1f}% smaller)")
    print("  verifying ...", flush=True)

    if not verify_lossless(src, tmp):
        print("  [ERROR] verification FAILED — deleting temporary output")
        tmp.unlink()
        return False
    if fps is not None and get_frame_rate(tmp) != fps:
        print(f"  [ERROR] frame-rate verification FAILED — expected {fps:g} fps")
        tmp.unlink()
        return False

    tmp.replace(dst)
    print("  verified ✓")
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch re-encode AVI -> FFV1/MKV")
    parser.add_argument("src_dir", type=Path, help="Source directory containing AVI files")
    parser.add_argument("dst_dir", type=Path, help="Destination directory for MKV files")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without encoding")
    parser.add_argument(
        "--fps", type=float,
        help="Retimestamp every frame at this acquisition rate (no frame dropping)",
    )
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
        if process_file(src, dst, dry_run=args.dry_run, fps=args.fps):
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
