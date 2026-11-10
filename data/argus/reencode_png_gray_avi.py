#!/usr/bin/env python3
"""Losslessly compact raw 8-bit grayscale AVIs in place.

Outputs use PNG compression with ``pix_fmt=gray`` in the original AVI
container, matching the repository's compact analysis-video convention. A
temporary file is checked for geometry, timing, frame count, and decoded
frame-by-frame MD5 identity before replacement.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def probe(path: Path) -> dict[str, object]:
    result = run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate",
        "-show_entries", "format=size,duration", "-of", "json", str(path),
    ], capture=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    stream.update(payload["format"])
    return stream


def frame_md5(path: Path) -> list[str]:
    result = run([
        "ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
        "-pix_fmt", "gray", "-f", "framemd5", "-",
    ], capture=True)
    return [
        line.rsplit(",", 1)[-1].strip()
        for line in result.stdout.splitlines()
        if line and not line.startswith("#")
    ]


def names_from_file(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Accept either one filename per line or a saved ``cp source dest`` list.
        names.append(Path(shlex.split(line)[-1]).name)
    return names


def encode_one(source: Path, *, replace: bool) -> tuple[int, int]:
    before = probe(source)
    if before.get("codec_name") == "png" and before.get("pix_fmt") == "gray":
        size = source.stat().st_size
        print(f"SKIP already compact: {source.name} ({size / 1024**3:.3f} GiB)", flush=True)
        return size, size
    if before.get("pix_fmt") != "gray":
        raise RuntimeError(f"expected 8-bit grayscale source, got {before.get('pix_fmt')}: {source}")

    temporary = source.with_name(f".{source.stem}.reencode_tmp.avi")
    if temporary.exists():
        temporary.unlink()
    print(f"ENCODE {source.name} ({source.stat().st_size / 1024**3:.3f} GiB)", flush=True)
    run([
        "ffmpeg", "-y", "-v", "error", "-nostats", "-i", str(source),
        "-map", "0:v:0", "-c:v", "png", "-pix_fmt", "gray", "-an", str(temporary),
    ])

    after = probe(temporary)
    structural = ("width", "height", "nb_frames", "r_frame_rate", "duration")
    mismatched = {key: (before.get(key), after.get(key)) for key in structural if before.get(key) != after.get(key)}
    if mismatched:
        raise RuntimeError(f"structural mismatch for {source.name}: {mismatched}")
    source_md5 = frame_md5(source)
    encoded_md5 = frame_md5(temporary)
    if not source_md5 or source_md5 != encoded_md5:
        mismatch_count = sum(a != b for a, b in zip(source_md5, encoded_md5))
        raise RuntimeError(
            f"decoded-frame verification failed for {source.name}: "
            f"source={len(source_md5)}, output={len(encoded_md5)}, mismatches={mismatch_count}"
        )

    original_size = source.stat().st_size
    encoded_size = temporary.stat().st_size
    savings = 100 * (1 - encoded_size / original_size)
    print(f"VERIFY {len(source_md5)} identical frames; {encoded_size / 1024**3:.3f} GiB ({savings:.1f}% smaller)", flush=True)
    if replace:
        os.replace(temporary, source)
        print(f"REPLACE {source.name}", flush=True)
    else:
        print(f"KEEP temporary output: {temporary}", flush=True)
    return original_size, encoded_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_dir", type=Path, help="directory containing source AVIs")
    parser.add_argument("--files-from", type=Path, help="optional filename or cp-command list")
    parser.add_argument("--replace", action="store_true", help="replace each source only after verification")
    parser.add_argument("--limit", type=int, help="process only the first N selected files")
    args = parser.parse_args()

    if not args.video_dir.is_dir():
        raise NotADirectoryError(args.video_dir)
    names = names_from_file(args.files_from) if args.files_from else [
        path.name for path in sorted(args.video_dir.glob("*.avi"))
        if not path.name.startswith(".") and ".reencode_tmp" not in path.name
    ]
    names = list(dict.fromkeys(names))
    if args.limit is not None:
        names = names[:args.limit]
    missing = [name for name in names if not (args.video_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing selected videos: {missing}")

    original_total = encoded_total = 0
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}]", flush=True)
        before, after = encode_one(args.video_dir / name, replace=args.replace)
        original_total += before
        encoded_total += after
    print(
        f"DONE files={len(names)} before={original_total / 1024**3:.3f} GiB "
        f"after={encoded_total / 1024**3:.3f} GiB "
        f"saved={(original_total - encoded_total) / 1024**3:.3f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
