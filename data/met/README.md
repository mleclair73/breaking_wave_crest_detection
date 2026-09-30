# Meteorology and cloud cover

This directory combines two external sources for August--November 2021:

- `ndbc_dukn7_met_2021_aug_nov.nc`: six-minute wind, pressure, and air
  temperature from NOAA NDBC station DUKN7, built with `download.py`.
- `era5_cloud_cover_2021_aug_nov.nc`: hourly ERA5 cloud cover at the grid point
  nearest the FRF, built with `download_era5_cloud.py`. A configured Copernicus
  CDS account is required.
- `duck_met_cloud_hourly_2021_aug_nov.nc`: the hourly joined product, built with
  `merge_hourly.py` after the two source files are present.

Install the acquisition dependency before downloading ERA5 data:

```bash
uv sync --locked --extra acquisition
```

Run the scripts from this directory. The generated NetCDF files are external
assets and are intentionally ignored by Git.
