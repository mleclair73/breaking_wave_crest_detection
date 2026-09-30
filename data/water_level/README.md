# Water Level — FRF eopNoaaTide (Duck, NC)

`frf_eopNoaaTide_water_levels_2021_aug_nov.nc` — measured tide (`water_level`, NAVD88, GMT,
~6-min sampling), Aug–Nov 2021, at the FRF NOAA Tide Station (NOAA CO-OPS gauge 8651370). Also
carries `predicted_water_level` and `residual_water_level`. Used for still-water depth in
`create_merged_dataset*.py`.

**Source:** USACE/CHL/COAB FRF eopNoaaTide, CHL THREDDS (also in `source.txt`):
<https://chldata.erdc.dren.mil/thredds/catalog/frf/oceanography/waterlevel/eopNoaaTide/>
Rebuild (download the monthly files + combine) with `python download.py`.