"""Supplemental: 2D histograms of breaker index gamma_s vs depth, Hs, and cross-shore bed slope,
colored by observation counts."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

from common.figure_style import apply_style, savefig, COL_W

DATA_PATH   = str(Path(__file__).parent.parent / 'combined_dunex_dataset_v2_lerp.nc')
OUTPUT_PATH = str(Path(__file__).parent / '01b_breaker_index_hist2d.png')


def main():
    apply_style()

    ds = xr.open_dataset(DATA_PATH)

    gamma_da = ds.waveHs / (ds.depth + ds.setup)
    slope_da = xr.apply_ufunc(lambda e: np.gradient(e, -ds.xFRF.values, axis=-1),
                              ds.elevation)  # dh/dx, cross-shore (signed)

    gamma_s = gamma_da.values.flatten()
    depth   = ds.depth.values.flatten()
    hs      = ds.waveHs.broadcast_like(gamma_da).values.flatten()
    slope   = slope_da.values.flatten()
    counts  = ds.observation_count.values.flatten()
    valid   = ~(np.isnan(gamma_s) | np.isnan(counts)) & (counts > 0) & (gamma_s > 0)

    gv, wt = gamma_s[valid], counts[valid]

    gamma_max = 6.0
    xbins = np.linspace(0, gamma_max, 60)

    # (y-values, y-bins, y-label) per panel
    panels = [
        (depth[valid], np.linspace(0, 5, 60),        r'Water depth $h$ (m)'),
        (hs[valid],    np.linspace(0, 5, 60),        r'$H_{s,offshore}$ (m)'),
        (slope[valid], np.linspace(-0.2, 0.2, 60), r'Bed slope $dh/dx$'),
    ]

    # Shared log color scale across panels
    hist2d = [np.histogram2d(gv, yv, bins=[xbins, yb], weights=wt)[0]
              for yv, yb, _ in panels]
    vmax = max(h.max() for h in hist2d)
    vmin = min(h[h > 0].min() for h in hist2d) 
    vmin=1e4
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 3, figsize=(2.1 * COL_W, 0.95 * COL_W),
                             sharex=True, constrained_layout=True)

    for ax, (yv, yb, ylabel) in zip(axes, panels):
        im = ax.hist2d(gv, yv, bins=[xbins, yb], weights=wt,
                       cmap='magma', norm=norm)[3]
        ax.set_xlabel(r'$\gamma_s = H_{s,offshore} / h$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, gamma_max)

    cbar = fig.colorbar(im, ax=axes, pad=0.02, aspect=30)
    cbar.set_label(r'Observations')

    savefig(fig, OUTPUT_PATH)


if __name__ == '__main__':
    main()
