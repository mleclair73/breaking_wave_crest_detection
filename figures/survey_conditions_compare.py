"""Compare wave height and tide stage (water level) distributions across the 3 survey events."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import dunex_paths

from common.figure_style import apply_style, savefig, PAGE_W, panel_label

DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"
OUTPUT_PATH = str(Path(__file__).parent / 'survey_conditions_compare.png')


def box_strip(ax, groups, labels, ylabel):
    """Box plot with jittered points overlaid, one column per survey."""
    positions = np.arange(len(groups))
    ax.boxplot(groups, positions=positions, widths=0.5, showfliers=False,
               medianprops=dict(color='black'),
               boxprops=dict(color='0.4'), whiskerprops=dict(color='0.4'),
               capprops=dict(color='0.4'))
    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        x = i + rng.uniform(-0.12, 0.12, size=len(g))
        ax.plot(x, g, 'o', markersize=3, color='steelblue', alpha=0.8, zorder=3)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylabel)


def main():
    apply_style()

    ds = xr.open_dataset(DATASET)
    svn = pd.to_datetime(ds.survey_time.values).normalize()
    dates = sorted(pd.unique(svn))
    labels = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in dates]

    hs_groups = [ds.waveHs.values[svn == d]      for d in dates]
    wl_groups = [ds.water_level.values[svn == d] for d in dates]

    fig, (ax_hs, ax_wl) = plt.subplots(1, 2, figsize=(PAGE_W, PAGE_W / 2.4))

    box_strip(ax_hs, hs_groups, labels, r'$H_s$ (m)')
    box_strip(ax_wl, wl_groups, labels, 'Water level (m)')
    ax_wl.axhline(0, color='0.6', linewidth=0.6, linestyle=':', zorder=1)

    panel_label(ax_hs, 'a')
    panel_label(ax_wl, 'b')

    fig.tight_layout(pad=0.4)
    savefig(fig, OUTPUT_PATH)

    # Summary stats per survey
    for name, groups in [('Hs (m)', hs_groups), ('water_level (m)', wl_groups)]:
        print(f'\n=== {name} ===')
        for lab, g in zip(labels, groups):
            print(f'  {lab}  n={len(g):2d}  '
                  f'median={np.median(g):+.2f}  mean={np.mean(g):+.2f}  '
                  f'min={np.min(g):+.2f}  max={np.max(g):+.2f}')


if __name__ == '__main__':
    main()
