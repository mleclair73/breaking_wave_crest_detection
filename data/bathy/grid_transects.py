"""Grid FRF elevation-transect surveys onto a common cross-shore grid, by profile line.

Each ``FRF_geomorphology_elevationTransects_survey_YYYYMMDD.nc`` is a point cloud of
(xFRF, yFRF, elevation) sampled along fixed FRF profile lines (``profileNumber``). This
grids every profile in every survey to a common 1 m cross-shore grid, one time slice per
survey date, and writes a single netCDF plus an overview plot.

Run directly to (re)build the product:  ``python grid_transects.py``
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BATHY_DIR = Path(__file__).parent
PATTERN   = 'FRF_geomorphology_elevationTransects_survey_*.nc'
OUT_NC    = BATHY_DIR / 'gridded_surveys.nc'
OUT_PNG   = BATHY_DIR / 'gridded_surveys_overview.png'
OUT_SHEET = BATHY_DIR / 'gridded_surveys_contact_sheet.png'
DX        = 1.0  # m, cross-shore grid resolution


def _survey_date(path):
    """Survey date from a ``..._survey_YYYYMMDD.nc`` filename."""
    return pd.to_datetime(Path(path).stem.split('_')[-1])


def _grid_profile(xFRF, elevation, x_grid):
    """Linear-interpolate one profile's points onto ``x_grid``; NaN outside its range."""
    ok = np.isfinite(xFRF) & np.isfinite(elevation)
    if ok.sum() < 2:
        return np.full(x_grid.size, np.nan)
    x, z = xFRF[ok], elevation[ok]
    x, idx = np.unique(x, return_index=True)  # sorts and drops duplicate x
    return np.interp(x_grid, x, z[idx], left=np.nan, right=np.nan)


def grid_surveys(bathy_dir=BATHY_DIR, dx=DX):
    """Grid all survey files into an ``(time, profile, xFRF)`` elevation dataset.

    Alongshore identity is the native ``profileNumber``; ``yFRF`` is stored as the mean
    cross-shore-line position per profile.
    """
    files = sorted(Path(bathy_dir).glob(PATTERN))
    if not files:
        raise FileNotFoundError(f'no survey files matching {PATTERN} in {bathy_dir}')

    # pass 1: global cross-shore extent and the profile set (with representative yFRF)
    x_min, x_max = np.inf, -np.inf
    prof_y = {}
    for f in files:
        with xr.open_dataset(f) as ds:
            x_min = min(x_min, float(np.nanmin(ds.xFRF)))
            x_max = max(x_max, float(np.nanmax(ds.xFRF)))
            pn, yv = np.round(ds.profileNumber.values), ds.yFRF.values
            for p in np.unique(pn[np.isfinite(pn)]):
                prof_y.setdefault(int(p), []).append(np.nanmean(yv[pn == p]))

    x_grid   = np.arange(np.floor(x_min), np.ceil(x_max) + dx, dx)
    profiles = np.array(sorted(prof_y))
    yFRF     = np.array([np.mean(prof_y[p]) for p in profiles])
    pidx     = {p: i for i, p in enumerate(profiles)}

    # pass 2: grid every profile of every survey onto the common grid
    times, elev = [], []
    for f in files:
        with xr.open_dataset(f) as ds:
            pn, xv, zv = np.round(ds.profileNumber.values), ds.xFRF.values, ds.elevation.values
            grid = np.full((profiles.size, x_grid.size), np.nan)
            for p in np.unique(pn[np.isfinite(pn)]):
                m = pn == p
                grid[pidx[int(p)]] = _grid_profile(xv[m], zv[m], x_grid)
        times.append(_survey_date(f))
        elev.append(grid)

    return xr.Dataset(
        {'elevation': (('time', 'profile', 'xFRF'), np.stack(elev)),
         'yFRF':      (('profile',), yFRF)},
        coords={'time': times, 'profile': profiles, 'xFRF': x_grid},
        attrs={'description': 'FRF elevation-transect surveys gridded by profile line',
               'source_pattern': PATTERN, 'dx_m': dx,
               'convention': 'elevation = NAVD88 bottom elevation [m]; yFRF = mean line position [m]'},
    ).sortby('time')


def overview_plot(ds, path=OUT_PNG):
    """Two-panel overview: latest-survey plan view and cross-shore profile evolution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 5))

    last = ds.isel(time=-1)
    pc = a0.pcolormesh(ds.xFRF, ds.yFRF, last.elevation, shading='nearest', cmap='terrain')
    a0.set(xlim=(0, 700), xlabel='xFRF [m]', ylabel='yFRF [m]',
           title=f'Elevation — survey {pd.to_datetime(ds.time.values[-1]):%Y-%m-%d}')
    fig.colorbar(pc, ax=a0, label='elevation [m]')

    yl = 800.0
    pi = int(np.argmin(np.abs(ds.yFRF.values - yl)))
    cols = plt.cm.viridis(np.linspace(0, 1, ds.time.size))
    for ti in range(ds.time.size):
        a1.plot(ds.xFRF, ds.elevation.isel(profile=pi, time=ti), color=cols[ti], lw=0.8)
    a1.set(xlim=(0, 500), ylim=(-8, 3), xlabel='xFRF [m]', ylabel='elevation [m]',
           title=f'Profile evolution — yFRF~{ds.yFRF.values[pi]:.0f} m ({ds.time.size} surveys)')
    a1.axhline(0, color='k', lw=0.5)

    fig.suptitle(f'Gridded FRF elevation-transect surveys — {ds.profile.size} profiles, '
                 f'{ds.time.size} dates', y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def contact_sheet(ds, path=OUT_SHEET, xmax=700, ncols=6):
    """Small-multiples plan-view of every survey date on a shared color scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n = ds.time.size
    nrows = int(np.ceil(n / ncols))
    vmin, vmax = np.nanpercentile(ds.elevation.values, [1, 99])

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 2.2 * nrows),
                             sharex=True, sharey=True)
    pc = None
    for ax, ti in zip(axes.flat, range(n)):
        pc = ax.pcolormesh(ds.xFRF, ds.yFRF, ds.elevation.isel(time=ti),
                           shading='nearest', cmap='terrain', vmin=vmin, vmax=vmax)
        ax.set_title(f'{pd.to_datetime(ds.time.values[ti]):%Y-%m-%d}', fontsize=8)
        ax.set_xlim(0, xmax)
    for ax in axes.flat[n:]:
        ax.axis('off')

    fig.supxlabel('xFRF [m]')
    fig.supylabel('yFRF [m]')
    fig.colorbar(pc, ax=axes, label='elevation [m]', shrink=0.6, pad=0.02)
    fig.suptitle(f'Gridded FRF elevation-transect surveys — {n} dates', y=1.0)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    ds = grid_surveys()
    ds.to_netcdf(OUT_NC)
    overview_plot(ds, OUT_PNG)
    contact_sheet(ds, OUT_SHEET)
    print(f'wrote {OUT_NC}  dims={dict(ds.sizes)}')
    print(f'wrote {OUT_PNG}')
    print(f'wrote {OUT_SHEET}')
    print(f'profiles yFRF: {np.round(ds.yFRF.values, 0)}')
    print(f'dates: {pd.to_datetime(ds.time.values[0]):%Y-%m-%d} .. '
          f'{pd.to_datetime(ds.time.values[-1]):%Y-%m-%d}')


if __name__ == '__main__':
    main()
