# External data inventory

The repository keeps acquisition and processing code in Git, but not the large
scientific, image, or video assets. Stage those assets at the paths below before
running the corresponding workflow.

| Path | Expected local content | Used by |
|---|---|---|
| `8m_array_waves/` | Three monthly FRF 8-m-array NetCDF files for September--November 2021 | merged physical dataset and wave-condition figures |
| `bathy/` | FRF survey DEM and elevation-transect NetCDF files | water-depth interpolation and bathymetry figures |
| `met/` | NDBC DUKN7 meteorology, ERA5 cloud cover, and their hourly merged NetCDF | acquisition-condition analysis |
| `water_level/` | Combined August--November 2021 FRF tide record | still-water depth and merged physical dataset |
| `video_dataset/argus/` | 36 rectified Argus videos from the published production set | production inference |
| `video_dataset/extra/` | Four supplemental rectified Argus videos | supplemental inference analyses |
| `video_dataset/good_data_mask.nc` and `ignore_mask.npy` | spatial validity masks | downstream filtering and figures |
| `argus/` | source videos and metadata used by dataset-building utilities | dataset regeneration only |

The Argus production videos are distributed with the associated Dryad dataset:
<https://doi.org/10.5061/dryad.r2280gbsn>. FRF wave, bathymetry, and water-level
products come from the CHL THREDDS catalog. Each subdirectory README records the
specific source. The meteorology and water-level directories include download
scripts; ERA5 retrieval additionally requires a configured Copernicus CDS
account.

Model-ready image and mask datasets are inventoried separately in
`model/data/README.md`. NetCDF, AVI, NumPy, and other generated/binary files in
this tree remain local and must not be added to ordinary Git history.
