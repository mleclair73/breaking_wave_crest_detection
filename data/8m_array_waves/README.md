# FRF 8-m Array — Wave Conditions (DUNEX)

Offshore wave conditions from the **FRF 8-m array** directional wave gauge (Duck, NC) during the
**DUNEX** window (Sept-Nov 2021). Provides incident wave parameters (Hs, Tp, direction, spectra)
used by `create_merged_dataset.py` and the dissipation figures.

## Contents
- `FRF-ocean_waves_8m-array_YYYYMM.nc` — monthly wave records (2021-09, 2021-10, 2021-11).

## Source
FRF 8-m array product, CHL THREDDS catalog:
<https://chldata.erdc.dren.mil/thredds/> → `frf/oceanography/waves/8m-array/`.

Fetch and verify with `python download.py` (hashes in `SHA256SUMS`).