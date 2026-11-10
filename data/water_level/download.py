"""
Download FRF eopNoaaTide monthly water levels into raw/, verify them against raw/SHA256SUMS,
combine them into one dataset, and save.

Source: USACE/CHL/COAB -- FRF-ocean_waterlevel_eopNoaaTide, the NOAA tide record at the FRF
"NOAA Tide Station" (Duck, NC; the NOAA CO-OPS 8651370 gauge). Served as monthly NetCDFs from
the CHL THREDDS catalog:
    https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waterlevel/eopNoaaTide/
Datum NAVD88, timezone GMT, ~6-minute sampling.

Variables are renamed to snake_case; `water_level` (measured tide) is the primary field expected
by create_merged_dataset.py, alongside `predicted_water_level` and `residual_water_level`.

Usage:
    python download.py
    python download.py --out <file.nc>
"""
import argparse
import sys
from pathlib import Path

import xarray as xr

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
sys.path.insert(0, str(HERE.parent))
import thredds  # noqa: E402

CATALOG_URL = (
    "https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waterlevel/eopNoaaTide/"
)
RENAME = {
    "waterLevel": "water_level",
    "predictedWaterLevel": "predicted_water_level",
    "residualWaterLevel": "residual_water_level",
}


def remote_path(name):
    # FRF-ocean_waterlevel_eopNoaaTide_YYYYMM.nc
    year = name.rsplit("_", 1)[1][:4]
    return f"oceanography/waterlevel/eopNoaaTide/{year}/{name}"


def combine(out_path):
    names = sorted(thredds.read_sums(RAW / "SHA256SUMS"))
    monthly = [xr.open_dataset(RAW / n).load() for n in names]
    ds = xr.concat(monthly, dim="time").sortby("time").drop_duplicates("time", keep="first")
    ds = ds.rename({k: v for k, v in RENAME.items() if k in ds})

    ds.attrs["source"] = "USACE/CHL/COAB FRF-ocean_waterlevel_eopNoaaTide (CHL THREDDS)"
    ds.attrs["source_url"] = CATALOG_URL
    ds.attrs["datum"] = "NAVD88"
    ds.attrs["timezone"] = "GMT"
    ds.attrs["processing_notes"] = (
        f"Monthly eopNoaaTide files {', '.join(names)} "
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
    ap.add_argument("--out", default=str(HERE / "frf_eopNoaaTide_water_levels_2021_aug_nov.nc"))
    args = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    thredds.main(RAW, remote_path)
    combine(args.out)
