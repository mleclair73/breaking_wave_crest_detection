"""Download the FRF 8-m array monthly wave files listed in SHA256SUMS and verify them.

Source: CHL THREDDS frf/oceanography/waves/8m-array/<year>/

Usage:
    python download.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import thredds  # noqa: E402


def remote_path(name):
    # FRF-ocean_waves_8m-array_YYYYMM.nc
    year = name.rsplit("_", 1)[1][:4]
    return f"oceanography/waves/8m-array/{year}/{name}"


if __name__ == "__main__":
    thredds.main(HERE, remote_path)
