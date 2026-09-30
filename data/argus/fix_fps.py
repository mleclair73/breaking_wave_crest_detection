from pathlib import Path
import argparse
import cv2
import subprocess

from common.datetime_utils import (
    get_time_resolution_for_date,
    parse_datetime_from_filename,
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Correct the declared frame rate of rectified Argus MKVs."
    )
    parser.add_argument("video_dir", type=Path)
    args = parser.parse_args()
    video_dir = args.video_dir
    files = sorted(video_dir.glob("ArgusFF_*_RectifiedVideo.mkv"))

    if not files:
        print("No matching files found.")
    
    for f in files:
        dt = parse_datetime_from_filename(f.name)
        time_res, expected_fps = get_time_resolution_for_date(dt)

        cap = cv2.VideoCapture(str(f))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if abs(actual_fps - expected_fps) < 0.01:
            print(f"{f.name}  →  {actual_fps:.2f} Hz ")
        else:
            print(f"{f.name}  →  expected {expected_fps:.1f} Hz, actual {actual_fps:.2f} Hz  — fixing...")
            out = f.with_name(f.stem + "_fixed.mkv")
            subprocess.run([
                "ffmpeg", "-r", str(int(expected_fps)),
                "-i", str(f),
                "-c", "copy",
                str(out)
            ], check=True)
            print(f"  → Written: {out.name}")
