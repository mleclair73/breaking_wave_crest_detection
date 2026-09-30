"""Breaker index histogram (alt) saved once per survey event: depth-stacked bars with a labelled literature reference lane."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import dunex_paths
from common.figure_style import apply_style, savefig, COL_W

DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"
OUTPUT_STEM = str(Path(__file__).parent / '01_breaker_index_hist_alt')


def make_figure(dv, gv_all, wt_all):
    """Build one alt-style breaker-index figure from valid depth/gamma/count arrays."""
    H_rms_to_Hs = np.sqrt(2)  # Hs = sqrt(2) * Hrms for Rayleigh distribution

    # McCowan (1891)
    mccowan_gamma = 0.78

    # Thornton & Guza (1982) — reported as gamma_rms; convert to gamma_s for plotting
    thornton_guza_gamma_rms = 0.42
    thornton_guza_gamma_s   = thornton_guza_gamma_rms * H_rms_to_Hs

    # Sallenger & Holman (1985) — reported as gamma_rms range; convert to gamma_s for plotting
    sallenger_holman_gamma_rms_low  = 0.29
    sallenger_holman_gamma_rms_high = 0.55
    sallenger_holman_gamma_s_low    = sallenger_holman_gamma_rms_low  * H_rms_to_Hs
    sallenger_holman_gamma_s_high   = sallenger_holman_gamma_rms_high * H_rms_to_Hs

    # Raubenheimer (1996) — reported as gamma_s range
    raubenheimer_gamma_s_low  = 0.3
    raubenheimer_gamma_s_high = 0.7

    # Battjes (1974); see related discussion in Raubenheimer.
    battjes_gamma_low  = 0.7
    battjes_gamma_high = 1.2

    # Carini et al. (2021) — wave-by-wave gamma measured at the FRF, mean +/- 1 sigma
    carini_gamma_mean = 0.49
    carini_gamma_std  = 0.27
    carini_color      = 'mediumpurple'

    x_max = 3

    fig, (ax_ref, ax) = plt.subplots(
        2, 1, figsize=(COL_W, COL_W * 1.22), sharex=True,
        gridspec_kw={'height_ratios': [0.34, 1], 'hspace': 0.06})

    # Reference lane above the histogram: literature values as labelled, non-overlapping rows
    bar_h = 0.34
    lit_rows = [
        dict(kind='errbar', mean=carini_gamma_mean, std=carini_gamma_std,
             color=carini_color,
             label=r'Carini et al. (2021)$^\dagger$'),
        dict(kind='point', x=mccowan_gamma, marker='o', color='black',
             label=r'McCowan (1891)'),
        dict(kind='point', x=thornton_guza_gamma_s, marker='^', color='black',
             label=r'Thornton et al. (1983)'),
        dict(kind='range', lo=sallenger_holman_gamma_s_low, hi=sallenger_holman_gamma_s_high,
             color='indianred', hatch='\\\\\\\\\\\\',
             label=r'Sallenger et al. (1985)$^\dagger$'),
        dict(kind='range', lo=raubenheimer_gamma_s_low, hi=raubenheimer_gamma_s_high,
             color='forestgreen', hatch='//////',
             label=r'Raubenheimer et al. (1996)$^\dagger$'),
        dict(kind='range', lo=battjes_gamma_low, hi=battjes_gamma_high,
             color='darkgoldenrod', hatch='|||||',
             label=r'Battjes (1974)'),
    ]
    for row, e in enumerate(lit_rows):
        if e['kind'] == 'point':
            ax_ref.plot(e['x'], row, marker=e['marker'], color=e['color'],
                        markersize=4, linestyle='none', zorder=2)
        elif e['kind'] == 'range':
            ax_ref.add_patch(mpatches.Rectangle((e['lo'], row - bar_h / 2), e['hi'] - e['lo'],
                                                bar_h, facecolor='none', edgecolor=e['color'],
                                                hatch=e['hatch'], linewidth=1.0))
        elif e['kind'] == 'errbar':
            ax_ref.errorbar(e['mean'], row, xerr=e['std'], fmt='D', color=e['color'],
                            markersize=4, elinewidth=1.0, capsize=2, zorder=2)
        ax_ref.text(x_max - 0.05, row, e['label'], va='center', ha='right', fontsize=6)

    ax_ref.text(0.05, len(lit_rows) - 1, r'$\dagger$ Duck, NC', va='center', ha='left',
                fontsize=5.5, style='italic', color='0.35')

    ax_ref.set_ylim(len(lit_rows) - 0.5, -0.5)
    ax_ref.set_yticks([])
    ax_ref.tick_params(labelbottom=False)

    # Histogram, stacked by depth band
    bins = np.linspace(0.0, x_max, 40)

    depth_edges = np.arange(1, 6, 1)  # 1.0, 2.0, 3.0, 4.0, 5.0 m; bands ordered deep -> shallow
    depth_masks  = [dv > depth_edges[-1]]
    depth_labels = [rf'$h > {depth_edges[-1]:g}$ m']
    for hi, lo in zip(depth_edges[-1:0:-1], depth_edges[-2::-1]):
        depth_masks.append((dv > lo) & (dv <= hi))
        depth_labels.append(rf'${lo:g} < h \leq {hi:g}$ m')
    depth_masks.append(dv <= depth_edges[0])
    depth_labels.append(rf'$h \leq {depth_edges[0]:g}$ m')
    depth_colors = [plt.cm.Blues(v) for v in np.linspace(0.92, 0.20, len(depth_masks))]

    ax.hist([gv_all[m] for m in depth_masks], bins=bins,
            weights=[wt_all[m] for m in depth_masks], stacked=True,
            color=depth_colors, edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_xlabel(r'$\gamma_s = H_{s,offshore} / h$')
    ax.set_ylabel(r'Observations $(\times 10^6)$')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))
    ax.set_xlim(0, x_max)

    # Depth-band legend (describes the stacked bars); listed shallow -> deep, top to bottom.
    depth_handles = [mpatches.Patch(facecolor=c, edgecolor='black', linewidth=0.4)
                     for c in depth_colors[::-1]]
    leg = ax.legend(depth_handles, depth_labels[::-1], title='Water depth',
                    loc='upper right', fontsize=6, title_fontsize=6,
                    frameon=True, handlelength=1.1, handleheight=1.1,
                    labelspacing=0.4, borderaxespad=0.8)
    leg.get_frame().set_linewidth(0.4)
    leg.get_frame().set_edgecolor('black')

    fig.tight_layout(pad=0.3)

    # Summary stats
    gv, wt = gv_all, wt_all
    total  = wt.sum()
    pct    = wt[(gv >= raubenheimer_gamma_s_low) & (gv <= battjes_gamma_high)].sum() / total * 100
    pct_gt = wt[gv > 0.3].sum() / total * 100
    w_mean = np.average(gv, weights=wt)
    si     = np.argsort(gv)
    w_med  = gv[si][np.searchsorted(np.cumsum(wt[si]), 0.5 * total)]
    hist, _ = np.histogram(gv, bins=bins, weights=wt)
    w_mode = 0.5 * (bins[hist.argmax()] + bins[hist.argmax() + 1])

    mode_line = ax.axvline(w_mode, color='black', linestyle='--', linewidth=1.0, zorder=4)
    ax.add_artist(leg)  # keep the depth-band legend when adding the second one below
    mode_leg = ax.legend([mode_line], [rf'mode $\gamma_s = {w_mode:.2f}$'],
                         loc='upper right', fontsize=6, frameon=True,
                         handlelength=1.4, borderaxespad=0.8)
    mode_leg.get_frame().set_linewidth(0.4)
    mode_leg.get_frame().set_edgecolor('black')

    # Stack the depth-band legend directly beneath the mode legend, both upper right.
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    mbox = mode_leg.get_window_extent(rend).transformed(ax.transAxes.inverted())
    leg.set_bbox_to_anchor((1.0, mbox.y0 - 0.015), transform=ax.transAxes)

    print(f"% in [{raubenheimer_gamma_s_low}, {battjes_gamma_high}]:  {pct:.1f}%")
    print(f"% gamma_s > 0.3:   {pct_gt:.1f}%")
    print(f"Weighted mean:     {w_mean:.4f}")
    print(f"Weighted median:   {w_med:.4f}")
    print(f"Weighted mode:     {w_mode:.4f}  (bin center, width {bins[1]-bins[0]:.4f})")

    return fig


def main():
    apply_style()

    ds = xr.open_dataset(DATASET)

    # Group time steps by calendar date of the survey.
    survey_dates = pd.to_datetime(ds.survey_time.values).normalize()
    unique_dates = sorted(pd.unique(survey_dates))

    for date in unique_dates:
        sel = survey_dates == date
        ds_sub = ds.isel(time=np.flatnonzero(sel))
        max_time = ds_sub.time.values.max()
        lag_days = np.abs((max_time - np.datetime64(date)) / np.timedelta64(1, 'D'))
        print(f'  latest data: {pd.Timestamp(max_time)}  ({lag_days:.2f} d after survey)')
        depth   = ds_sub.depth.values.flatten()
        gamma_s = (ds_sub.waveHs / (ds_sub.depth + ds_sub.setup)).values.flatten()
        counts  = ds_sub.observation_count.values.flatten()
        valid   = ~(np.isnan(gamma_s) | np.isnan(counts) | np.isnan(depth)) & (counts > 0) & (gamma_s > 0)

        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        print(f"=== {date_str}  (n_time={int(sel.sum())}) ===")

        fig = make_figure(depth[valid], gamma_s[valid], counts[valid])
        savefig(fig, f'{OUTPUT_STEM}_{date_str}.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
