# FRF Geomorphology — Bathymetry (DUNEX)

Nearshore bathymetry from the U.S. Army Corps of Engineers Field Research Facility (FRF),
Duck, NC, covering the DUNEX window (Aug–Nov 2021). Used to derive still-water depth for 
wave-speed / depth inversion, breaker index.

## Contents
- `FRF_geomorphology_DEMs_surveyDEM_YYYYMMDD.nc` — gridded survey DEMs.
- `FRF_geomorphology_elevationTransects_survey_YYYYMMDD.nc` — cross-shore elevation transects.

| Survey date | Observations matched | File |
|---|---|---|
| 2021-09-13 | 9 / 36 | `FRF_geomorphology_DEMs_surveyDEM_20210913.nc` |
| 2021-09-28 | 14 / 36 | `FRF_geomorphology_DEMs_surveyDEM_20210928.nc` |
| 2021-10-21 | 13 / 36 | `FRF_geomorphology_DEMs_surveyDEM_20211021.nc` |

Other DEMs present here (2021-08-30, 09-05, 11-17) bracket the window but are not used by
the merged-dataset/ depth inversion. The transect surveys were not used in the paper but
provide additional context

## Source
FRF geomorphology product, CHL THREDDS catalog:
<https://chldata.erdc.dren.mil/thredds/> → `frf/geomorphology/DEMs/` and
`frf/geomorphology/elevationTransects/`.
