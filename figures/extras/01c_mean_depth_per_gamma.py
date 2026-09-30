"""Supplemental: weighted mean water depth per breaker-index (gamma_s) bin, with +/- 1 sigma error bars."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

from common.figure_style import apply_style, savefig, COL_W

DATA_PATH   = str(Path(__file__).parent.parent / 'combined_dunex_dataset_v2_lerp.nc')
OUTPUT_PATH = str(Path(__file__).parent / '01c_mean_depth_per_gamma.png')


def main():
    apply_style()

    ds = xr.open_dataset(DATA_PATH)

    depth   = ds.depth.values.flatten()
    gamma_s = (ds.waveHs / (ds.depth + ds.setup)).values.flatten()
    counts  = ds.observation_count.values.flatten()
    valid   = ~(np.isnan(gamma_s) | np.isnan(counts) | np.isnan(depth)) & (counts > 0) & (gamma_s > 0)

    gv, dv, wt = gamma_s[valid], depth[valid], counts[valid]

    x_max = 3
    bins    = np.linspace(0, x_max, 30)
    centers = 0.5 * (bins[:-1] + bins[1:])

    cnt   = np.histogram(gv, bins, weights=wt)[0]
    sw_d  = np.histogram(gv, bins, weights=wt * dv)[0]
    sw_d2 = np.histogram(gv, bins, weights=wt * dv ** 2)[0]
    mean_d = np.divide(sw_d, cnt, out=np.full_like(sw_d, np.nan), where=cnt > 0)
    var_d  = np.divide(sw_d2, cnt, out=np.full_like(sw_d2, np.nan), where=cnt > 0) - mean_d ** 2
    std_d  = np.sqrt(np.clip(var_d, 0, None))

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))

    ax.errorbar(centers, mean_d, yerr=std_d, fmt='o', markersize=3,
                color=plt.cm.Blues(0.85), ecolor=plt.cm.Blues(0.6),
                elinewidth=0.8, capsize=2, zorder=3)

    ax.set_xlabel(r'$\gamma_s = H_{s,offshore} / h$')
    ax.set_ylabel(r'Mean water depth $\overline{h}$ (m)')
    ax.set_box_aspect(1)
    ax.set_xlim(0, x_max)
    ax.set_ylim(bottom=0)

    fig.tight_layout(pad=0.3)
    savefig(fig, OUTPUT_PATH)


if __name__ == '__main__':
    main()
