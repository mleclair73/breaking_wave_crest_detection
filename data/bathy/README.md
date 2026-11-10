# FRF Geomorphology — Bathymetry (DUNEX)

Nearshore bathymetry from the U.S. Army Corps of Engineers Field Research Facility (FRF),
Duck, NC, covering the DUNEX window (Aug–Nov 2021). Used to derive still-water depth for
wave-speed / depth inversion, breaker index.

## Contents
- `FRF_geomorphology_DEMs_surveyDEM_YYYYMMDD.nc` — gridded survey DEMs.
- `FRF_geomorphology_elevationTransects_survey_YYYYMMDD.nc` — cross-shore elevation transects.

| Survey date | Videos matched | File |
|---|---|---|
| 2021-09-13 | 9 / 40 | `FRF_geomorphology_DEMs_surveyDEM_20210913.nc` |
| 2021-09-28 | 18 / 40 | `FRF_geomorphology_DEMs_surveyDEM_20210928.nc` |
| 2021-10-21 | 13 / 40 | `FRF_geomorphology_DEMs_surveyDEM_20211021.nc` |

Other DEMs present here (2021-08-30, 09-05, 11-17) bracket the window but are not used by
the depth inversion in `create_merged_dataset.py`. The transect surveys are not used.

## Source
FRF geomorphology product, CHL THREDDS catalog:
<https://chldata.erdc.dren.mil/thredds/> → `frf/geomorphology/DEMs/` and
`frf/geomorphology/elevationTransects/`.

Fetch and verify with `python download.py` (hashes in `SHA256SUMS`). The pinned files are
the server versions re-issued 2025-06-13; relative to the 2021 originals, DEM elevation and
xFRF/yFRF are unchanged, and transects differ only in metadata, point order, and two points
removed from the 2021-11-17 survey. `gridded_surveys.nc` was rebuilt from these.
