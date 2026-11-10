"""Download the FRF survey DEMs and elevation transects listed in SHA256SUMS and verify them.

Source: CHL THREDDS frf/geomorphology/DEMs/surveyDEM/data/ and
frf/geomorphology/elevationTransects/survey/data/

Usage:
    python download.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import thredds  # noqa: E402


def remote_path(name):
    if name.startswith("FRF_geomorphology_DEMs_surveyDEM_"):
        return f"geomorphology/DEMs/surveyDEM/data/{name}"
    if name.startswith("FRF_geomorphology_elevationTransects_survey_"):
        return f"geomorphology/elevationTransects/survey/data/{name}"
    raise ValueError(f"unknown bathy product: {name}")


if __name__ == "__main__":
    thredds.main(HERE, remote_path)
