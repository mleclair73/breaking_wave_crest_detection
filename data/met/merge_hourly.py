"""
Build one hourly met dataset for Duck, NC by downsampling the 6-min DUKN7 record to hourly and
merging it with the hourly ERA5 cloud cover. Water level is intentionally left untouched (stays
6-min in ../water_level/).

Inputs (produced by download.py / download_era5_cloud.py):
    ndbc_dukn7_met_2021_aug_nov.nc   6-min DUKN7 wind/pressure/air-temp
    era5_cloud_cover_2021_aug_nov.nc  hourly ERA5 total/low/mid/high cloud cover

Hourly aggregation of DUKN7 (label = start of hour, GMT):
    wind_dir    vector (u/v) mean direction of the ~10 six-min samples
    wind_speed  scalar mean of the six-min speeds
    wind_gust   hourly max
    pressure    hourly mean
    air_temp    hourly mean
ERA5 cloud is already hourly and is joined on the timestamp.

Usage:
    python merge_hourly.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).parent


def dukn7_to_hourly(met_nc, min_count=3):
    df = xr.open_dataset(met_nc).to_dataframe()

    # Wind as u/v (meteorological "from" convention) for a proper vector-mean direction.
    dir_rad = np.deg2rad(df["wind_dir"])
    u = -df["wind_speed"] * np.sin(dir_rad)
    v = -df["wind_speed"] * np.cos(dir_rad)

    r = df.resample("1h")
    u_h = u.resample("1h").mean()
    v_h = v.resample("1h").mean()
    wind_dir = np.rad2deg(np.arctan2(-u_h, -v_h)) % 360.0

    hourly = pd.DataFrame({
        "wind_dir": wind_dir,
        "wind_speed": r["wind_speed"].mean(),
        "wind_gust": r["wind_gust"].max(),
        "pressure": r["pressure"].mean(),
        "air_temp": r["air_temp"].mean(),
    })
    # Drop hours with too few valid 6-min samples (data gaps -> NaN, not a fake mean).
    counts = r["wind_speed"].count().reindex(hourly.index)
    hourly = hourly.where(counts >= min_count, axis=0)
    hourly.index.name = "time"
    return hourly


def merge(met_nc, era5_nc, out_path, min_count):
    hourly = dukn7_to_hourly(met_nc, min_count=min_count)
    met = xr.Dataset.from_dataframe(hourly)

    cloud = xr.open_dataset(era5_nc)

    merged = xr.merge([met, cloud], join="inner")

    # Re-attach units.
    units = {
        "wind_dir": "degrees_true", "wind_speed": "m/s", "wind_gust": "m/s",
        "pressure": "hPa", "air_temp": "degC",
    }
    for name, u in units.items():
        merged[name].attrs["units"] = u
    for name in cloud.data_vars:
        merged[name].attrs["units"] = cloud[name].attrs.get("units", "fraction (0-1)")

    merged.attrs["title"] = "Hourly met + cloud cover, Duck NC (FRF)"
    merged.attrs["station"] = "DUKN7 (wind/pressure/air temp) + ERA5 nearest grid point (cloud)"
    merged.attrs["timezone"] = "GMT"
    merged.attrs["sampling"] = "1 hour (hour-start label)"
    merged.attrs["processing_notes"] = (
        "DUKN7 6-min downsampled to hourly: wind dir = u/v vector mean, wind speed = scalar "
        "mean, gust = hourly max, pressure/air_temp = hourly mean; hours with <"
        f"{min_count} valid 6-min samples set to NaN. ERA5 cloud joined on the hour (inner)."
    )

    out_path = Path(out_path)
    merged.to_netcdf(out_path)
    n = merged.sizes["time"]
    t0 = np.datetime_as_string(merged.time.min().values, unit="m")
    t1 = np.datetime_as_string(merged.time.max().values, unit="m")
    print(f"Saved {out_path}  ({n} hourly steps, {t0} .. {t1} GMT)")
    for name in merged.data_vars:
        v = merged[name].values.astype(float)
        good = np.isfinite(v)
        pct = 100 * good.mean() if good.size else 0.0
        print(f"  {name:18s} mean={np.nanmean(v):8.3f}  valid={pct:5.1f}%")
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge hourly DUKN7 met + ERA5 cloud")
    ap.add_argument("--met", default=str(HERE / "ndbc_dukn7_met_2021_aug_nov.nc"))
    ap.add_argument("--era5", default=str(HERE / "era5_cloud_cover_2021_aug_nov.nc"))
    ap.add_argument("--out", default=str(HERE / "duck_met_cloud_hourly_2021_aug_nov.nc"))
    ap.add_argument("--min-count", type=int, default=3)
    args = ap.parse_args()
    merge(args.met, args.era5, args.out, args.min_count)
