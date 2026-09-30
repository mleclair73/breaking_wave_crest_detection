"""
Merge wave-speed + bathymetry + water level + wave data into one combined dataset.

Updated version of create_merged_dataset.py: the speed source now defaults to THIS run's
wave-speed NetCDFs (model/predictions/wave_speeds/, from the argus_full inference +
calculate_wave_speeds.py) instead of the old mae_dunex_revamp/output_segnext_savgol/.
Speed dir and output path are CLI args so it's reusable across runs.

Usage:
    python create_merged_dataset.py
    python create_merged_dataset.py --speeds-dir <dir> --out <file.nc> [--no-lerp]
        [--speed-override <replacement.nc> ...]
"""
import argparse
from pathlib import Path

import dunex_paths
import xarray as xr

# Default to this run's speeds (argus_full -> calculate_wave_speeds.py *_speeds.nc)
DEFAULT_SPEEDS_DIR = dunex_paths.PREDICTIONS_DIR / 'wave_speeds'


def select_speed_files(speeds_dir, speed_overrides=None):
    """Return base speed files with explicitly named replacements applied.

    An override must have the same basename as a file in ``speeds_dir``. It
    replaces that input in place; it can never add a second scene or timestamp
    to the merge.
    """
    speeds_dir = Path(speeds_dir)
    no_sidecar = lambda paths: [p for p in paths if not p.name.startswith('._')]
    speed_files = (no_sidecar(sorted(speeds_dir.glob('*_speeds.nc')))
                   or no_sidecar(sorted(speeds_dir.glob('*.nc'))))
    if not speed_files:
        raise FileNotFoundError(f"No speed NetCDFs found in {speeds_dir}")

    selected = {path.name: path for path in speed_files}
    overrides_seen = set()
    for override_value in speed_overrides or []:
        override = Path(override_value)
        if not override.is_file():
            raise FileNotFoundError(f"Speed override does not exist: {override}")
        if override.name not in selected:
            raise ValueError(
                f"Speed override has no matching base input: {override.name}"
            )
        if override.name in overrides_seen:
            raise ValueError(f"Duplicate speed override: {override.name}")
        selected[override.name] = override
        overrides_seen.add(override.name)

    return [selected[name] for name in sorted(selected)]


def process_and_save_dataset(speeds_dir, out_path, lerp=True, speed_overrides=None):
    """Merge speed, bathymetry, and wave data into a single unified dataset."""

    # 1. Load Speed Data
    print(f"Loading speed data from {speeds_dir} ...")
    speed_files = select_speed_files(speeds_dir, speed_overrides)
    print(f"  {len(speed_files)} speed file(s)")
    for override in speed_overrides or []:
        print(f"  override: {Path(override).name} <- {override}")

    def preprocess(ds):
        return ds.assign(source=ds.attrs['source'])

    ds = xr.open_mfdataset(speed_files, preprocess=preprocess)
    print("Computing speed data into memory...")
    ds = ds.compute()

    # 2. Load Bathymetry & Water Level
    print("Loading bathymetry")
    bathy_dir = dunex_paths.BATHY_DIR
    bathy_ds = xr.open_mfdataset(
        list(bathy_dir.glob('FRF_geomorphology_DEMs_surveyDEM_*')),
        drop_variables=['project']
    )
    bathy_ds['survey_time'] = bathy_ds.time
    bathy_ds_selected = bathy_ds.sel(time=ds.time, method='nearest')
    bathy_ds_selected['time'] = ds.time
    print("Interpolating bathymetry to speed grid...")
    bathy_ds_selected = bathy_ds_selected.interp(
        xFRF=ds.xFRF, yFRF=ds.yFRF, method='linear' if lerp else 'nearest'
    ).compute()

    print("Loading water levels")
    water_level_ds = xr.load_dataset(dunex_paths.WATER_LEVEL_NC)
    water_level_interp = water_level_ds.water_level.interp(time=ds.time)

    ds['elevation'] = bathy_ds_selected.elevation
    ds['survey_time'] = bathy_ds_selected.survey_time
    ds['water_level'] = water_level_interp
    ds['depth'] = ds.water_level - ds.elevation

    # 3. Load Wave Data (8m array)
    print("Loading wave data...")
    wave_ds = xr.open_mfdataset(
        sorted(dunex_paths.WAVES_8M_DIR.glob('*.nc'))
    )
    # Keep time-varying water depth at the 8m array sensor as its own variable
    wave_ds = wave_ds.rename({'depth': 'depth_8m_array'})

    print('Interpolating wave data')
    wave_ds_interp = wave_ds.interp(time=ds.time, method='nearest').compute()
    ds = xr.merge([ds, wave_ds_interp])

    # 4. Compute radiation-stress wave setup
    print("Computing radiation-stress wave setup...")
    from utils.setup_1d import compute_setup_1_5d
    FRF_SHORE_NORMAL_DEG = 71.8535   # FRF cross-shore axis angle from true north
    RHO, G = 1025.0, 9.81
    E0 = (RHO * G * ds.waveEnergyDensity.integrate('waveFrequency')).values
    eta = compute_setup_1_5d(
        E0,
        ds.waveTp.values,
        ds.depth.values,
        peak_direction=ds.wavePeakDirectionPeakFrequency.values,
        shore_normal_deg=FRF_SHORE_NORMAL_DEG,
    )
    ds['setup'] = xr.DataArray(
        eta, dims=ds.depth.dims, coords=ds.depth.coords,
        attrs={'long_name': 'Radiation-stress wave setup', 'units': 'm'}
    )

    if 'fps' in ds:
        print("WARNING: 'fps' already in dataset, skipping fps assignment")
    else:
        is_sep19_2021 = (ds.time.dt.year == 2021) & (ds.time.dt.month == 9) & (ds.time.dt.day == 19)
        ds['fps'] = xr.where(is_sep19_2021, 1.0, 2.0)

    print(f'Saving to {out_path}')
    ds.to_netcdf(out_path)
    print(f'Done. {len(speed_files)} videos merged.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--speeds-dir', default=str(DEFAULT_SPEEDS_DIR),
                    help='Dir of *_speeds.nc from calculate_wave_speeds.py')
    ap.add_argument('--out', default=None,
                    help='Output combined NetCDF (default: combined_dunex_dataset[_lerp].nc)')
    ap.add_argument(
        '--speed-override', action='append', default=[], metavar='REPLACEMENT_NC',
        help=('Replace the base speed NetCDF having the same basename. Repeat '
              'for multiple overrides; replacements cannot add new inputs.'),
    )
    ap.add_argument('--no-lerp', action='store_true', help='Nearest instead of linear interp')
    args = ap.parse_args()

    lerp = not args.no_lerp
    out = args.out or str(dunex_paths.REPO_ROOT / f'combined_dunex_dataset{"_lerp" if lerp else ""}.nc')
    process_and_save_dataset(
        args.speeds_dir,
        out,
        lerp=lerp,
        speed_overrides=args.speed_override,
    )
