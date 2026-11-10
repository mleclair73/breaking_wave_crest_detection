"""
Download NDBC DUKN7 standard meteorological data, filter to the water-level time range,
and save as NetCDF (to sit alongside the FRF eopNoaaTide record in ../water_level/).

Source: NOAA NDBC station DUKN7 (Duck Pier, NC; USACE FRF), standard meteorological archive:
    https://www.ndbc.noaa.gov/station_page.php?station=dukn7
    https://www.ndbc.noaa.gov/view_text_file.php?filename=dukn7h{year}.txt.gz&dir=data/historical/stdmet/
6-minute sampling, timezone GMT (UTC), matching the eopNoaaTide 6-min cadence.

DUKN7 only reports wind (dir/speed/gust), barometric pressure, and air temperature; water
temperature, waves, dew point, visibility, and tide are all missing (sentinel values) for this
station, so only the real fields are kept. Missing sentinels are converted to NaN.

Usage:
    python download.py
    python download.py --year 2021 --months 8 9 10 11 --out <file.nc>
"""
import argparse
import ssl
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

TEXT_URL = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename=dukn7h{year}.txt.gz&dir=data/historical/stdmet/"
)
STATION_PAGE = "https://www.ndbc.noaa.gov/station_page.php?station=dukn7"

# Raw NDBC stdmet columns (2 header lines: names, then units).
RAW_COLS = [
    "YY", "MM", "DD", "hh", "mm",
    "WDIR", "WSPD", "GST", "WVHT", "DPD", "APD", "MWD",
    "PRES", "ATMP", "WTMP", "DEWP", "VIS", "TIDE",
]
# Fields DUKN7 actually reports -> snake_case names.
KEEP = {
    "WDIR": "wind_dir",       # degT
    "WSPD": "wind_speed",     # m/s
    "GST": "wind_gust",       # m/s
    "PRES": "pressure",       # hPa
    "ATMP": "air_temp",       # degC
}
UNITS = {
    "wind_dir": "degrees_true",
    "wind_speed": "m/s",
    "wind_gust": "m/s",
    "pressure": "hPa",
    "air_temp": "degC",
}
# NDBC missing-data sentinels (any value >= these in the raw column is "no data").
SENTINELS = (99.0, 99.00, 999.0, 9999.0)


def _download_text(url):
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
        return r.read().decode("utf-8", errors="replace")


def download_and_process(year, months, out_path):
    print(f"Downloading DUKN7 stdmet {year} ...")
    text = _download_text(TEXT_URL.format(year=year))

    # Skip the 2 comment/header lines (start with '#'); parse whitespace-delimited.
    df = pd.read_csv(
        StringIO(text), sep=r"\s+", comment="#", header=None, names=RAW_COLS,
    )

    time = pd.to_datetime(
        dict(year=df.YY, month=df.MM, day=df.DD, hour=df.hh, minute=df.mm), utc=True
    ).dt.tz_localize(None)
    df = df.set_index(time.rename("time"))

    df = df[df.MM.isin(months)]

    out = pd.DataFrame(index=df.index)
    for raw, name in KEEP.items():
        col = df[raw].astype(float)
        col = col.mask(col.isin(SENTINELS))       # exact NDBC missing sentinels -> NaN
        out[name] = col
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="first")]

    ds = xr.Dataset.from_dataframe(out)
    for name, u in UNITS.items():
        ds[name].attrs["units"] = u
    ds.attrs["station"] = "DUKN7 (Duck Pier, NC; USACE FRF)"
    ds.attrs["source"] = "NOAA NDBC standard meteorological archive"
    ds.attrs["source_url"] = STATION_PAGE
    ds.attrs["timezone"] = "GMT"
    ds.attrs["sampling"] = "6 min"
    ds.attrs["processing_notes"] = (
        f"DUKN7 stdmet {year}, months {months[0]}..{months[-1]}. Kept only fields DUKN7 "
        "reports (wind dir/speed/gust, pressure, air temp); water temp, waves, dew point, "
        "visibility, tide are absent for this station. NDBC missing sentinels -> NaN."
    )

    out_path = Path(out_path)
    ds.to_netcdf(out_path)
    n = ds.sizes["time"]
    print(f"Saved {out_path}  ({n} timesteps, {out.index.min()} .. {out.index.max()} GMT)")
    for name in UNITS:
        v = ds[name].values.astype(float)
        good = np.isfinite(v)
        pct = 100 * good.mean() if good.size else 0.0
        if good.any():
            print(f"  {name:11s} mean={np.nanmean(v):8.2f}  valid={pct:5.1f}%")
        else:
            print(f"  {name:11s} (all NaN)  valid={pct:5.1f}%")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download + process NDBC DUKN7 met to NetCDF")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--months", type=int, nargs="+", default=[8, 9, 10, 11])
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "ndbc_dukn7_met_2021_aug_nov.nc"),
    )
    args = ap.parse_args()
    download_and_process(args.year, args.months, args.out)
