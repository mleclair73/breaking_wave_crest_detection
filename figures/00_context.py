"""
Campaign context: Hs + field coverage (top), frequency spectrogram (middle),
direction spectrogram (bottom). Coverage overlays on all panels.
"""

import pickle
from glob import glob
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import dunex_paths
from matplotlib.colors import LogNorm
from common.figure_style import (
    CMAP,
    COLOR_ARGUS,
    COLOR_SURVEY,
    COLOR_TRANSECT,
    PAGE_W,
    apply_style,
    panel_label,
    savefig,
)

WAVE_DIR = dunex_paths.WAVES_8M_DIR
BATHY_DIR = dunex_paths.BATHY_DIR
ARGUS_META_DIR = dunex_paths.DATA_ROOT / "argus" / "metadata"
OUTPUT_PATH = Path(__file__).parent / "00_context.png"

T0 = pd.to_datetime('2021-09-17')
T1 = pd.to_datetime('2021-10-31')

FRF_OFFSET   = -18.2 # OFFSET FROM WEST
SHORE_NORMAL = 90.0 + FRF_OFFSET   # 71.8° CW from North

SURVEY_COLOR   = COLOR_SURVEY
TRANSECT_COLOR = COLOR_TRANSECT
ARGUS_COLOR    = COLOR_ARGUS


def load_argus_times(meta_dir):
    times = []
    for f in sorted(glob(str(Path(meta_dir) / '*.pickle'))):
        with open(f, 'rb') as fh:
            md = pickle.load(fh)
        times.append((md['time'][0], md['time'][-1]))
    return times


def add_coverage(ax, argus_times, survey_times, transect_times):
    for start, end in argus_times:
        ax.axvspan(start, end, alpha=0.5, facecolor=ARGUS_COLOR, zorder=1)
    for t in survey_times:
        ax.axvline(t, c=SURVEY_COLOR, linewidth=0.8, zorder=5)
    for t in transect_times:
        ax.axvline(t, c=TRANSECT_COLOR, linewidth=0.8, zorder=1)


def main():
    apply_style()

    ds = xr.open_mfdataset(sorted(glob(str(Path(WAVE_DIR) / '*.nc'))),
                           combine='by_coords').sel(time=slice(T0, T1)).load()

    # Surveys (DEMs)
    dems = sorted(glob(str(Path(BATHY_DIR) / '*DEM*.nc')))
    surveys = xr.open_mfdataset(dems, combine='nested', concat_dim=['time'])
    survey_times = pd.to_datetime(surveys.time.values)
    survey_times = survey_times[(survey_times >= T0) & (survey_times <= T1)]

    # Transects
    trans = sorted(glob(str(Path(BATHY_DIR) / '*Transect*.nc')))
    transects = xr.open_mfdataset(trans, combine='nested', concat_dim=['time'])
    transect_times = np.unique(pd.to_datetime(transects.time.dt.date.values))
    transect_times = transect_times[(transect_times >= T0) & (transect_times <= T1)]

    # Argus
    argus_times = load_argus_times(ARGUS_META_DIR)

    # Wave data arrays
    freq      = ds.waveFrequency.values                  # (nf,)
    dirs      = ds.waveDirectionBins.values              # (ndir,)
    time_vals = ds.time.values                           # (nt,)
    E         = ds.waveEnergyDensity.values              # (nt, nf)
    dw_e      = ds.directionalWaveEnergyDensity.values   # (nt, nf, ndir)
    f_mean    = 1.0 / ds.waveTm.values                   # (nt,)
    direction = ds.waveMeanDirection.values              # (nt,)

    # Direction spectrogram: integrate over frequency
    df_vals = np.gradient(freq)
    E_dir = np.nansum(dw_e * df_vals[np.newaxis, :, np.newaxis], axis=1)  # (nt, ndir)

    # Wrap directions so 90° is centred: map (270, 360] → (-90, 0]
    dirs_wrap      = np.where(dirs > 270, dirs - 360, dirs)
    direction_wrap = np.where(direction > 270, direction - 360, direction)
    sort_idx       = np.argsort(dirs_wrap)
    dirs_wrap      = dirs_wrap[sort_idx]
    E_dir          = E_dir[:, sort_idx]

    # --- Figure: 3 equal-height panels ---
    fig_w = PAGE_W
    fig_h = fig_w / 1.4
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_w, fig_h),
                                         sharex=True,
                                         gridspec_kw={'hspace': 0.05})

    xfmt   = mdates.DateFormatter('%m/%d')
    xticks = pd.date_range(start=T0, end=T1, freq='1W')

    # --- Panel 1: Hs ---
    ax1.plot(time_vals, ds.waveHs.values, 'k-', linewidth=1, zorder=2)
    ax1.set_ylabel('$H_s$ [m]')
    ax1.set_ylim(0, 3.5)
    add_coverage(ax1, argus_times, survey_times, transect_times)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ARGUS_COLOR, alpha=0.8),
        plt.Line2D([0], [0], color=SURVEY_COLOR, linewidth=0.8),
        plt.Line2D([0], [0], color=TRANSECT_COLOR, linewidth=0.8),
    ]
    ax1.legend(handles, ['Argus', 'Survey', 'Transect'],
               loc='upper right', ncol=3, framealpha=0.95)

    # --- Panel 2: frequency spectrogram ---
    E_plot = np.where(E > 0, E, np.nan)
    ax2.pcolormesh(time_vals, freq, E_plot.T,
                   norm=LogNorm(vmin=1e-1, vmax=10),
                   cmap=CMAP['spectro'], shading='nearest')
    ax2.plot(time_vals, f_mean, '-', color='k', linewidth=0.8, alpha=1, zorder=2)
    ax2.set_ylabel('Mean Frequency [Hz]')
    ax2.set_ylim(0.04, 0.35)
    add_coverage(ax2, argus_times, survey_times, transect_times)

    # --- Panel 3: direction spectrogram (wrapped, 90° centred) ---
    E_dir_plot = np.where(E_dir > 0, E_dir, np.nan)
    ax3.pcolormesh(time_vals, dirs_wrap, E_dir_plot.T,
                   norm=LogNorm(vmin=1e-4, vmax=E_dir_plot[np.isfinite(E_dir_plot)].max()),
                   cmap=CMAP['spectro'], shading='nearest')
    ax3.plot(time_vals, direction_wrap, '-', color='k', linewidth=0.8, alpha=1, zorder=2)
    ax3.axhline(SHORE_NORMAL, color='k', linewidth=0.5, linestyle='--', zorder=2,
                label=f'Shore normal ({SHORE_NORMAL:.1f}°)')
    ax3.set_ylabel('Mean Direction [°]')
    ax3.set_ylim(-90, 270)
    ax3.set_yticks([-90, 0, 90, 180, 270])
    leg = ax3.legend(loc='upper right', facecolor='white', framealpha=0.95)
    leg.get_frame().set_boxstyle('round,pad=0.2')
    add_coverage(ax3, argus_times, survey_times, transect_times)

    # Panel labels
    for ax, label in zip((ax1, ax2, ax3), ('(a)', '(b)', '(c)')):
        panel_label(ax, label)

    # Align y-labels to a common x position (independent of ytick label width)
    fig.align_ylabels((ax1, ax2, ax3))

    # Shared x-a
    ax3.set_xlim([T0, T1])
    ax3.xaxis.set_major_formatter(xfmt)
    ax3.set_xticks(xticks)

    savefig(fig, OUTPUT_PATH)


if __name__ == '__main__':
    main()
