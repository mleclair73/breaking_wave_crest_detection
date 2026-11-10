# External data inventory

The repository keeps acquisition and processing code in Git, but not the raw data.
Stage those assets at the paths below before running the corresponding workflow.

## Where to get it

| Path | Content | Source | How to obtain |
|---|---|---|---|
| `8m_array_waves/` | FRF 8-m array wave NetCDF, monthly, Sep--Nov 2021 | FRF, CHL THREDDS | `python download.py`; see `8m_array_waves/README.md` |
| `bathy/` | FRF survey DEMs and elevation transects | FRF, CHL THREDDS | `python download.py`; see `bathy/README.md` |
| `water_level/` | FRF eopNoaaTide water level, Aug--Nov 2021 | FRF, CHL THREDDS | `python download.py`; see `water_level/README.md` |
| `met/` | NDBC DUKN7 meteorology, ERA5 cloud cover, hourly merge | NOAA NDBC; Copernicus CDS (ERA5) | `download.py`, `download_era5_cloud.py`, `merge_hourly.py`; see `met/README.md` |
| `video_dataset/argus/` | 40 rectified Argus videos (`.avi`) | Dryad `imagery/` | remux from `.mkv`; see `video_dataset/argus/README.md` |
| `argus/metadata/*.pickle` | Argus header metadata: 40 in Dryad, 332 in the full archive | Dryad `metadata/` (40); full archive not published | copy; see `../dryad/README.md` |
| `argus/*.avi` | 9 OOD source videos; 2 are also in Dryad | Argus archive, not published | only needed to rebuild `model/data/ood_test/` |
| `video_dataset/good_data_mask.nc` | spatial validity mask | generated asset | included in Git |
| `dunex_breaking_wave_crest_dataset.nc` | final merged crest-speed and environmental dataset | generated | `create_merged_dataset.py`; see top-level `README.md` |

Source links:

- Dryad: <https://doi.org/10.5061/dryad.r2280gbsn>. Download into `data/dryad/`
  and follow `dryad/README.md` to verify it and place files.
- CHL THREDDS: <https://chldata.erdc.dren.mil/thredds/> (`frf/oceanography/` and
  `frf/geomorphology/`). Each THREDDS folder's `download.py` fetches the files
  listed in its `SHA256SUMS` via `thredds.py` and verifies them; files already
  present are only verified.
- ERA5 retrieval requires a configured Copernicus CDS account.

## Code-only directories

`argus/` also holds Argus video utilities (re-encoding, metadata copy, FPS fix),
dropout/intensity diagnostics, and pixel-resolution maps (`argus_error.py`).

Model-ready image and mask datasets are inventoried separately in
`model/data/README.md`.
