#!/usr/bin/env python3
"""Condition-coverage figures for the Stage-1 argus inference pool.

Derives per-video environmental conditions (waves, water level, wind, cloud,
solar geometry) from the repository NetCDF sources, samples grayscale intensity
histograms from the rectified videos, and renders coverage figures in the same
style as model/data/full_split/figures:

  argus_wave_conditions.png          water level, Hs, mean direction, peak freq
  argus_lighting_met_conditions.png  solar geometry, wind, cloud cover
  argus_intensity_histograms.png     per-video grayscale intensity PDFs

Each condition gets its own panel; the argus pool has no train/val/test splits,
so panels plot value vs acquisition date across the campaign.

Conditions derivation matches the validated full_split pipeline: waves from the
8m-array, water level from the NOAA tide record, wind + cloud from the Duck met
station (duck_met_cloud_hourly, not NDBC), and a hand-coded NOAA solar position.

Outputs (all gitignored, carried outside git):
  figures/argus_video_conditions.csv   derived condition table
  figures/argus_video_histograms.npz   cached grayscale histograms
  figures/argus_*.png                  the coverage figures
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARGUS = HERE / "argus"
FIGURES = HERE / "figures"

LAT, LON = 36.1826, -75.7492  # FRF Duck NC
WAVES = {
    9: REPO / "data/8m_array_waves/FRF-ocean_waves_8m-array_202109.nc",
    10: REPO / "data/8m_array_waves/FRF-ocean_waves_8m-array_202110.nc",
    11: REPO / "data/8m_array_waves/FRF-ocean_waves_8m-array_202111.nc",
}
TIDE = REPO / "data/water_level/frf_eopNoaaTide_water_levels_2021_aug_nov.nc"
CLOUD = REPO / "data/met/duck_met_cloud_hourly_2021_aug_nov.nc"
BATHY = REPO / "data/bathy"  # FRF surveyDEMs; dates parsed from filenames

N_FRAMES = 200  # evenly-spaced frames sampled per video for the intensity histogram
POINT_COLOR = "#4c78a8"  # matches full_split train color
STAMP = re.compile(r"(\d{8}T\d{6}Z)")

WAVE_PANELS = ["water level (m)", "Hs (m)", "mean direction (°)", "peak frequency (Hz)"]
MET_PANELS = [
    "local hour (EDT)", "solar elevation (°)", "solar azimuth (°)",
    "wind speed (m/s)", "wind direction (°)", "wind gust (m/s)",
    "total cloud cover", "low cloud cover", "mid cloud cover", "high cloud cover",
]


def stamp_to_utc(text: str) -> dt.datetime:
    """Parse the acquisition stamp (e.g. 20211026T160100Z) to a UTC datetime."""
    m = STAMP.search(text)
    if not m:
        raise ValueError(f"no acquisition stamp in {text!r}")
    return dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ")


def solar(t: dt.datetime, lat: float = LAT, lon: float = LON) -> tuple[float, float]:
    """NOAA solar-position algorithm: (elevation_deg, azimuth_deg) for UTC time t."""
    y, mo = t.year, t.month
    day = t.day + (t.hour + t.minute / 60 + t.second / 3600) / 24
    if mo <= 2:
        y -= 1
        mo += 12
    a = y // 100
    b = 2 - a + a // 4
    jd = int(365.25 * (y + 4716)) + int(30.6001 * (mo + 1)) + day + b - 1524.5
    jc = (jd - 2451545.0) / 36525.0
    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (
        math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
        + math.sin(2 * mr) * (0.019993 - 0.000101 * jc)
        + math.sin(3 * mr) * 0.000289
    )
    true_long = l0 + c
    omega = 125.04 - 1934.136 * jc
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    obl = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obl_c = obl + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(
        math.asin(math.sin(math.radians(obl_c)) * math.sin(math.radians(app_long)))
    )
    vary = math.tan(math.radians(obl_c / 2)) ** 2
    eqtime = 4 * math.degrees(
        vary * math.sin(2 * math.radians(l0))
        - 2 * e * math.sin(mr)
        + 4 * e * vary * math.sin(mr) * math.cos(2 * math.radians(l0))
        - 0.5 * vary * vary * math.sin(4 * math.radians(l0))
        - 1.25 * e * e * math.sin(2 * mr)
    )
    tst = (t.hour * 60 + t.minute + t.second / 60 + eqtime + 4 * lon) % 1440
    ha = tst / 4 + 180 if tst / 4 < 0 else tst / 4 - 180
    latr, declr, har = math.radians(lat), math.radians(decl), math.radians(ha)
    zenith = math.degrees(
        math.acos(
            math.sin(latr) * math.sin(declr)
            + math.cos(latr) * math.cos(declr) * math.cos(har)
        )
    )
    elev = 90 - zenith
    za = math.radians(zenith)
    caz = (math.sin(latr) * math.cos(za) - math.sin(declr)) / (
        math.cos(latr) * math.sin(za)
    )
    caz = max(-1.0, min(1.0, caz))
    az = math.degrees(math.acos(caz))
    az = (az + 180) % 360 if ha > 0 else (540 - az) % 360
    return elev, az


def _sel(ds: xr.Dataset, t: dt.datetime, var: str) -> float:
    """Nearest-time value of var from an already-open dataset."""
    return float(ds[var].sel(time=np.datetime64(t), method="nearest").values)


def survey_dates() -> list[dt.date]:
    """All FRF surveyDEM dates, parsed from data/bathy filenames."""
    dates = [
        dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        for p in BATHY.glob("FRF_geomorphology_DEMs_surveyDEM_*.nc")
        if (m := re.search(r"(\d{8})\.nc$", p.name))
    ]
    if not dates:
        raise FileNotFoundError(f"no surveyDEM files under {BATHY}")
    return sorted(dates)


def signed_days_to_nearest(when: dt.date, surveys: list[dt.date]) -> int:
    """Signed days from the closest survey to the video (>0 survey before, <0 after)."""
    return min(((when - s).days for s in surveys), key=abs)


def derive_conditions() -> pd.DataFrame:
    """Per-video condition table for every argus recording (full_split columns)."""
    videos = sorted(ARGUS.glob("*.avi"))
    tide = xr.open_dataset(TIDE)
    cloud = xr.open_dataset(CLOUD)
    waves = {m: xr.open_dataset(p) for m, p in WAVES.items()}
    surveys = survey_dates()

    rows = []
    for path in videos:
        t = stamp_to_utc(path.name)
        w = waves[t.month]
        tp = _sel(w, t, "waveTp")
        el, az = solar(t)
        rows.append(
            {
                "source_video": path.stem,
                "acquisition_utc": t.isoformat() + "+00:00",
                "water level (m)": _sel(tide, t, "water_level"),
                "Hs (m)": _sel(w, t, "waveHs"),
                "mean direction (°)": _sel(w, t, "waveMeanDirection"),
                "peak frequency (Hz)": 1.0 / tp,
                "local hour (EDT)": t.hour + t.minute / 60 - 4,
                "solar elevation (°)": el,
                "solar azimuth (°)": az,
                "wind speed (m/s)": _sel(cloud, t, "wind_speed"),
                "wind direction (°)": _sel(cloud, t, "wind_dir"),
                "wind gust (m/s)": _sel(cloud, t, "wind_gust"),
                "total cloud cover": _sel(cloud, t, "cloud_cover_total"),
                "low cloud cover": _sel(cloud, t, "cloud_cover_low"),
                "mid cloud cover": _sel(cloud, t, "cloud_cover_mid"),
                "high cloud cover": _sel(cloud, t, "cloud_cover_high"),
                "days to nearest DEM survey": signed_days_to_nearest(t.date(), surveys),
            }
        )
    for ds in (tide, cloud, *waves.values()):
        ds.close()
    return pd.DataFrame(rows).sort_values("acquisition_utc").reset_index(drop=True)


def video_histogram(path: Path, n_frames: int = N_FRAMES) -> tuple[np.ndarray, int]:
    """Accumulate a 256-bin grayscale histogram over evenly-spaced frames.

    No-data (exact-zero) pixels are excluded so the distribution reflects the
    imaged water/beach lighting. Returns (counts[256], frames_used).
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.unique(np.linspace(0, total - 1, n_frames).astype(int))
    hist = np.zeros(256, np.int64)
    used = 0
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[gray > 0]
        hist += np.bincount(gray, minlength=256)
        used += 1
    cap.release()
    return hist, used


def sample_histograms(conditions: pd.DataFrame, force: bool = False) -> np.ndarray:
    """Per-video histograms as an (n_videos, 256) array, cached to .npz."""
    cache = FIGURES / "argus_video_histograms.npz"
    ids = conditions["source_video"].to_numpy()
    if cache.exists() and not force:
        data = np.load(cache, allow_pickle=True)
        if list(data["source_video"]) == list(ids):
            print(f"loaded cached histograms <- {cache}")
            return data["hist"]
        print("cache stale (source list changed); resampling")
    hist = np.zeros((len(ids), 256), np.int64)
    used = np.zeros(len(ids), int)
    for k, sid in enumerate(ids):
        hist[k], used[k] = video_histogram(ARGUS / f"{sid}.avi")
        print(f"  [{k + 1:2d}/{len(ids)}] {sid}  {used[k]} frames")
    np.savez(cache, source_video=ids, hist=hist, frames_used=used, n_frames=N_FRAMES)
    print(f"saved histograms -> {cache}")
    return hist


def condition_plot(frame: pd.DataFrame, columns: list[str], shape: tuple[int, int],
                   output: str, title: str) -> Path:
    """One panel per condition: value vs acquisition date (full_split style)."""
    times = pd.to_datetime(frame["acquisition_utc"]).dt.tz_localize(None)
    fig, axes = plt.subplots(*shape, figsize=(4 * shape[1], 3.6 * shape[0]), squeeze=False)
    for axis, column in zip(axes.flat, columns):
        axis.scatter(times, frame[column], color=POINT_COLOR, s=38, alpha=0.9,
                     edgecolor="k", linewidth=0.3)
        axis.set_ylabel(column)
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        if "cloud cover" in column:
            axis.set_ylim(-0.02, 1.02)
    for axis in axes.flat[len(columns):]:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, output)


def fig_survey_offset(conditions: pd.DataFrame, surveys: list[dt.date]) -> Path:
    """Signed days from the nearest surveyDEM to each video, vs acquisition date."""
    times = pd.to_datetime(conditions["acquisition_utc"]).dt.tz_localize(None)
    offset = conditions["days to nearest DEM survey"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for s in surveys:
        ax.axvline(pd.Timestamp(s), color="#999999", linestyle="--", linewidth=1, zorder=0)
    ax.axvline(pd.Timestamp(surveys[0]), color="#999999", linestyle="--", linewidth=1,
               zorder=0, label="surveyDEM date")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.scatter(times, offset, color=POINT_COLOR, s=45, alpha=0.9,
               edgecolor="k", linewidth=0.3, zorder=3)
    ax.set_ylabel("days to nearest surveyDEM\n(>0 survey before video, <0 after)")
    ax.set_title(
        f"Argus pool — bathymetry recency ({len(conditions)} videos, "
        f"{len(surveys)} surveyDEMs)"
    )
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="best", fontsize=9)
    return _save(fig, "argus_days_since_survey.png")


def fig_intensity_histograms(conditions: pd.DataFrame, hist: np.ndarray) -> Path:
    """Overlaid per-video intensity PDFs, colored by solar elevation."""
    fig, ax = plt.subplots(figsize=(11, 5))
    elev = conditions["solar elevation (°)"].to_numpy()
    norm = Normalize(vmin=elev.min(), vmax=elev.max())
    cmap = plt.get_cmap("cividis")
    bins = np.arange(256)
    for row, e in zip(hist, elev):
        pdf = row / row.sum()
        ax.plot(bins, pdf, color=cmap(norm(e)), alpha=0.7, linewidth=1.0)
    ax.set_xlim(0, 255)
    ax.set_xlabel("grayscale intensity (no-data excluded)")
    ax.set_ylabel("probability density")
    ax.set_title(f"Argus pool — per-video intensity histograms ({N_FRAMES} frames/video)")
    ax.grid(alpha=0.25)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                 label="solar elevation (°)", pad=0.01)
    return _save(fig, "argus_intensity_histograms.png")


def _save(fig, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh-hist", action="store_true", help="force re-sample video histograms")
    args = ap.parse_args()

    FIGURES.mkdir(exist_ok=True)
    conditions = derive_conditions()
    out = FIGURES / "argus_video_conditions.csv"
    conditions.to_csv(out, index=False)
    print(f"{len(conditions)} videos -> {out}")

    hist = sample_histograms(conditions, force=args.refresh_hist)

    figs = [
        condition_plot(conditions, WAVE_PANELS, (1, 4), "argus_wave_conditions.png",
                       f"Argus pool — wave conditions ({len(conditions)} videos)"),
        condition_plot(conditions, MET_PANELS, (2, 5), "argus_lighting_met_conditions.png",
                       f"Argus pool — lighting and meteorological conditions ({len(conditions)} videos)"),
        fig_survey_offset(conditions, survey_dates()),
        fig_intensity_histograms(conditions, hist),
    ]
    print("figures:")
    for p in figs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
