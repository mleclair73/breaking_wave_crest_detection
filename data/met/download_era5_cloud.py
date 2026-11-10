"""
Download ERA5 cloud cover for Duck, NC (USACE FRF) over the water-level time range and save
as NetCDF alongside the DUKN7 met record.

Source: Copernicus Climate Data Store, ERA5 hourly data on single levels (reanalysis):
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
Requires a free CDS account and ~/.cdsapirc (url + key). Variables retrieved:
    total_cloud_cover, low_cloud_cover, medium_cloud_cover, high_cloud_cover  (fraction 0-1)

ERA5 is on a 0.25 deg grid at hourly cadence (GMT). We pull a small box around the FRF pier
(~36.18 N, -75.75 W), then keep the single nearest grid point. Native hourly cadence is kept
(no resampling to the 6-min WL/DUKN7 grid).

Usage:
    python download_era5_cloud.py
    python download_era5_cloud.py --year 2021 --months 8 9 10 11 --out <file.nc>
"""
import argparse
import tempfile
import zipfile
from pathlib import Path

import cdsapi
import numpy as np
import xarray as xr

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "total_cloud_cover",
    "low_cloud_cover",
    "medium_cloud_cover",
    "high_cloud_cover",
]
# FRF pier, Duck NC.
FRF_LAT, FRF_LON = 36.1817, -75.7501
# Small box around the pier (N, W, S, E) so the request returns a few grid points.
AREA = [36.5, -76.25, 35.75, -75.25]

# ERA5 short names -> snake_case; all are cloud-area fraction 0-1.
RENAME = {
    "tcc": "cloud_cover_total",
    "lcc": "cloud_cover_low",
    "mcc": "cloud_cover_mid",
    "hcc": "cloud_cover_high",
}


def _open_any(path):
    """Open the CDS result whether it is a plain .nc or a .zip of .nc files."""
    try:
        return xr.open_dataset(path)
    except (OSError, ValueError):
        if not zipfile.is_zipfile(path):
            raise
        with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".nc")]
            z.extractall(tmp)
            dsets = [xr.open_dataset(Path(tmp) / n).load() for n in names]
            return xr.merge(dsets)


def download_cloud(year, months, out_path):
    req = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        raw = tmp.name
    print(f"Requesting ERA5 {DATASET} {year}-{months[0]:02d}..{months[-1]:02d} ...")
    cdsapi.Client().retrieve(DATASET, req, raw)

    ds = _open_any(raw)

    # New CDS files use 'valid_time'; normalize to 'time'.
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    ds = ds.sel(latitude=FRF_LAT, longitude=FRF_LON, method="nearest")
    got_lat = float(ds.latitude)
    got_lon = float(ds.longitude)

    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds})
    ds = ds[list(RENAME.values())]
    for name in RENAME.values():
        ds[name].attrs["units"] = "fraction (0-1)"
    ds = ds.drop_vars([c for c in ("latitude", "longitude", "number", "expver") if c in ds])

    ds.attrs["station"] = "ERA5 nearest grid point to FRF pier (Duck, NC)"
    ds.attrs["source"] = "Copernicus CDS ERA5 hourly single levels (reanalysis)"
    ds.attrs["source_url"] = f"https://cds.climate.copernicus.eu/datasets/{DATASET}"
    ds.attrs["timezone"] = "GMT"
    ds.attrs["sampling"] = "1 hour"
    ds.attrs["frf_target_lat_lon"] = f"{FRF_LAT}, {FRF_LON}"
    ds.attrs["era5_grid_lat_lon"] = f"{got_lat:.4f}, {got_lon:.4f}"

    out_path = Path(out_path)
    ds.to_netcdf(out_path)
    n = ds.sizes["time"]
    t0 = np.datetime_as_string(ds.time.min().values, unit="m")
    t1 = np.datetime_as_string(ds.time.max().values, unit="m")
    print(f"Saved {out_path}  ({n} hourly steps, {t0} .. {t1} GMT)")
    print(f"  grid point: {got_lat:.4f}, {got_lon:.4f} (target {FRF_LAT}, {FRF_LON})")
    for name in RENAME.values():
        v = ds[name].values.astype(float)
        print(f"  {name:18s} mean={np.nanmean(v):5.3f}  min={np.nanmin(v):5.3f}  max={np.nanmax(v):5.3f}")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download ERA5 cloud cover for the FRF over the WL range")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--months", type=int, nargs="+", default=[8, 9, 10, 11])
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "era5_cloud_cover_2021_aug_nov.nc"),
    )
    args = ap.parse_args()
    download_cloud(args.year, args.months, args.out)
