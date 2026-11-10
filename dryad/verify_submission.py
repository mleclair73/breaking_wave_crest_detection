#!/usr/bin/env python3
"""Validate the structure and media contracts of the Dryad submission."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path


EXPECTED_KEYS = {
    "beginGoodTimes",
    "cams_all",
    "captureArray",
    "captures",
    "endGoodTimes",
    "endSkipping",
    "missingFrames",
    "startSkipping",
    "time",
    "totalCaptured",
}

ONE_HZ_TIMESTAMPS = {
    "20210919T153000Z",
    "20210919T163000Z",
    "20210919T180000Z",
    "20210919T183000Z",
}


def timestamp(path: Path, suffix: str) -> str:
    prefix = "ArgusFF_"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"unexpected filename: {path.name}")
    return path.name[len(prefix) : -len(suffix)]


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    fps = num / den
    duration = float(data["format"]["duration"])
    return {
        "codec": stream["codec_name"],
        "pix_fmt": stream["pix_fmt"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": round(duration * fps),
        "duration": duration,
    }


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)
    print(f"ERROR: {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "dryad",
    )
    parser.add_argument("--expected", type=int, default=40)
    args = parser.parse_args()

    root = args.root.resolve()
    failures: list[str] = []
    specs = {
        "imagery": ("_RectifiedVideo.mkv", root / "imagery"),
        "crests": ("_DetectedCrests.mkv", root / "crests"),
        "metadata": ("_headerData.pickle", root / "metadata"),
    }
    groups: dict[str, dict[str, Path]] = {}

    for label, (suffix, directory) in specs.items():
        if not directory.is_dir():
            fail(f"missing directory: {directory}", failures)
            groups[label] = {}
            continue
        files = sorted(p for p in directory.iterdir() if not p.name.startswith("."))
        try:
            groups[label] = {timestamp(p, suffix): p for p in files}
        except ValueError as exc:
            fail(str(exc), failures)
            groups[label] = {}
        if len(groups[label]) != args.expected:
            fail(f"{label}: expected {args.expected} files, found {len(groups[label])}", failures)

    timestamp_sets = {label: set(paths) for label, paths in groups.items()}
    if timestamp_sets and len({frozenset(v) for v in timestamp_sets.values()}) != 1:
        all_stamps = set().union(*timestamp_sets.values())
        for label, stamps in timestamp_sets.items():
            missing = sorted(all_stamps - stamps)
            if missing:
                fail(f"{label}: missing timestamps {missing}", failures)

    total_frames = 0
    total_seconds = 0.0
    for stamp in sorted(set.intersection(*(set(g) for g in groups.values()))):
        image_info = probe(groups["imagery"][stamp])
        crest_info = probe(groups["crests"][stamp])
        for label, info in (("imagery", image_info), ("crests", crest_info)):
            expected_fps = 1.0 if stamp in ONE_HZ_TIMESTAMPS else 2.0
            contract = (info["codec"], info["pix_fmt"], info["width"], info["height"], info["fps"])
            if contract != ("ffv1", "gray", 500, 1600, expected_fps):
                fail(f"{stamp} {label}: unexpected media contract {contract}", failures)
        if image_info["frames"] != crest_info["frames"]:
            fail(
                f"{stamp}: imagery has {image_info['frames']} frames but crest mask has "
                f"{crest_info['frames']}",
                failures,
            )
        total_frames += int(image_info["frames"])
        total_seconds += float(image_info["duration"])

        with groups["metadata"][stamp].open("rb") as handle:
            metadata = pickle.load(handle)
        if set(metadata) != EXPECTED_KEYS:
            fail(f"{stamp}: metadata keys differ from documented keys", failures)
        if len(metadata.get("time", ())) != 3840:
            fail(f"{stamp}: metadata time grid is not 3840 elements", failures)
        capture = metadata.get("captureArray")
        if getattr(capture, "shape", None) != (3840, len(metadata.get("cams_all", ()))):
            fail(f"{stamp}: captureArray shape does not match cams_all", failures)

    print(f"root: {root}")
    print(f"collections: {len(groups.get('imagery', {}))}")
    print(f"frames: {total_frames}")
    print(f"duration: {total_seconds / 3600:.3f} hours")
    print(f"failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
