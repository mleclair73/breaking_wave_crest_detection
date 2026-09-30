"""
Download FRF eopNoaaTide water levels from the CHL THREDDS server, combine the monthly files
into one dataset, and save.

Source: USACE/CHL/COAB -- FRF-ocean_waterlevel_eopNoaaTide, the NOAA tide record at the FRF
"NOAA Tide Station" (Duck, NC; the NOAA CO-OPS 8651370 gauge). Served as monthly NetCDFs from
the CHL THREDDS catalog:
    https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waterlevel/eopNoaaTide/
Datum NAVD88, timezone GMT, ~6-minute sampling.

Variables are renamed to snake_case; `water_level` (measured tide) is the primary field expected
by create_merged_dataset*.py, alongside `predicted_water_level` and `residual_water_level`.

Usage:
    python download.py
    python download.py --year 2021 --months 8 9 10 11 --out <file.nc>
"""
import argparse
import ssl
import tempfile
import urllib.request
from pathlib import Path

import xarray as xr

FILESERVER = (
    "https://chldata.erdc.dren.mil/thredds/fileServer/frf/oceanography/"
    "waterlevel/eopNoaaTide/{year}/FRF-ocean_waterlevel_eopNoaaTide_{year}{month:02d}.nc"
)
CATALOG_URL = (
    "https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waterlevel/eopNoaaTide/"
)
RENAME = {
    "waterLevel": "water_level",
    "predictedWaterLevel": "predicted_water_level",
    "residualWaterLevel": "residual_water_level",
}


def _download(url, dest):
    # CHL THREDDS cert chain isn't always in the default trust store; skip verification.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90, context=ctx) as r, open(dest, "wb") as f:
        f.write(r.read())


def download_and_combine(year, months, out_path):
    with tempfile.TemporaryDirectory() as tmp:
        monthly = []
        for m in months:
            url = FILESERVER.format(year=year, month=m)
            dest = Path(tmp) / f"{year}{m:02d}.nc"
            print(f"Downloading {year}-{m:02d} ...")
            _download(url, dest)
            monthly.append(xr.open_dataset(dest).load())

        print("Combining ...")
        ds = xr.concat(monthly, dim="time").sortby("time").drop_duplicates("time", keep="first")
        ds = ds.rename({k: v for k, v in RENAME.items() if k in ds})

    ds.attrs["source"] = "USACE/CHL/COAB FRF-ocean_waterlevel_eopNoaaTide (CHL THREDDS)"
    ds.attrs["source_url"] = CATALOG_URL
    ds.attrs["datum"] = "NAVD88"
    ds.attrs["timezone"] = "GMT"
    ds.attrs["processing_notes"] = (
        f"Monthly eopNoaaTide files {year}-{months[0]:02d}..{year}-{months[-1]:02d} "
        "concatenated on time, sorted, de-duplicated."
    )

    out_path = Path(out_path)
    ds.to_netcdf(out_path)
    wl = ds["water_level"]
    print(f"Saved {out_path}  ({ds.sizes['time']} timesteps)")
    print(f"  water_level (NAVD88): mean={float(wl.mean()):.3f} "
          f"min={float(wl.min()):.3f} max={float(wl.max()):.3f} m")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download + combine FRF eopNoaaTide water levels")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--months", type=int, nargs="+", default=[8, 9, 10, 11])
    ap.add_argument(
        "--out",
        default=str(Path(__file__).parent / "frf_eopNoaaTide_water_levels_2021_aug_nov.nc"),
    )
    args = ap.parse_args()
    download_and_combine(args.year, args.months, args.out)
