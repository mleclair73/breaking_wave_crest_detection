"""Breaker index histogram (alt) saved once per survey event: depth-stacked bars with a labelled literature reference lane."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

import dunex_paths
from common.figure_style import apply_style, savefig, COL_W

DATASET = dunex_paths.ANALYSIS_DATASET
OUTPUT_STEM = str(Path(__file__).parent / '01_breaker_index_hist_alt')


def weighted_half_sample_mode(values, weights, assume_sorted=False):
    """Return the half-sample mode of an integer-weighted sample."""
    values = np.asarray(values, dtype=float)
    weights = np.rint(np.asarray(weights)).astype(np.int64)
    positive = np.isfinite(values) & (weights > 0)
    x = values[positive]
    weight = weights[positive]
    if not assume_sorted:
        order = np.argsort(x, kind='mergesort')
        x = x[order]
        weight = weight[order]
    if x.size == 0:
        return np.nan

    while True:
        sample_size = int(weight.sum())
        if sample_size <= 3:
            sample = np.repeat(x, weight)
            if sample_size == 1:
                return float(sample[0])
            if sample_size == 2:
                return float(sample.mean())
            left_gap, right_gap = np.diff(sample)
            if left_gap < right_gap:
                return float(sample[:2].mean())
            if right_gap < left_gap:
                return float(sample[1:].mean())
            return float(sample.mean())

        half_size = (sample_size + 1) // 2
        cumulative = np.r_[0, np.cumsum(weight)]
        max_start = sample_size - half_size

        # Expanded-sample widths change only where an endpoint crosses a
        # cumulative-weight boundary, so expansion is unnecessary.
        candidates = np.concatenate((
            [0, max_start],
            cumulative[1:-1],
            cumulative[1:-1] - half_size + 1,
        ))
        candidates = np.unique(
            candidates[(candidates >= 0) & (candidates <= max_start)]
        )
        lower = np.searchsorted(cumulative, candidates, side='right') - 1
        ends = candidates + half_size - 1
        upper = np.searchsorted(cumulative, ends, side='right') - 1
        widths = x[upper] - x[lower]

        tied = np.flatnonzero(widths == widths.min())
        start = int(candidates[tied[np.argmin(candidates[tied])]])
        end = start + half_size - 1
        first = int(np.searchsorted(cumulative, start, side='right') - 1)
        last = int(np.searchsorted(cumulative, end, side='right') - 1)

        x = x[first:last + 1]
        weight = weight[first:last + 1].copy()
        if first == last:
            weight[0] = half_size
        else:
            weight[0] = cumulative[first + 1] - start
            weight[-1] = end - cumulative[last] + 1


def make_figure(dv, gv_all, wt_all, video_all):
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

    # Carini et al. (2021) — bulk significant-wave-height index across FRF
    # transects, calculated from local Hsig and mean-sea-level depth.
    carini_gamma_low = 0.67
    carini_gamma_high = 0.71
    x_max = 3
    bin_width = 0.1
    bins = np.arange(0.0, x_max + 0.5 * bin_width, bin_width)
    reference_color = '0.35'
    reference_marker = 'o'
    reference_markersize = 2.4

    # Histogram depth bands, ordered deep -> shallow.
    depth_edges = np.arange(1, 6, 1)
    depth_masks = [dv > depth_edges[-1]]
    for hi, lo in zip(depth_edges[-1:0:-1], depth_edges[-2::-1]):
        depth_masks.append((dv > lo) & (dv <= hi))
    depth_masks.append(dv <= depth_edges[0])
    depth_colors = [plt.cm.Blues(v) for v in np.linspace(0.92, 0.20, len(depth_masks))]

    # Calculate a crest-count-weighted half-sample mode for each video in
    # water deeper than 1 m, then summarize the videos by their median and range.
    mode_mask = dv > 1
    video_modes = np.array([
        weighted_half_sample_mode(
            gv_all[mode_mask & (video_all == video)],
            wt_all[mode_mask & (video_all == video)])
        for video in np.unique(video_all)
        if np.any(mode_mask & (video_all == video))
    ])
    our_mode = float(np.median(video_modes))
    our_mode_low = float(video_modes.min())
    our_mode_high = float(video_modes.max())
    our_color = plt.cm.Blues(0.75)
    our_range_color = plt.cm.Blues(0.45)

    fig, (ax_ref, ax) = plt.subplots(
        2, 1, figsize=(COL_W, COL_W * 1.22), sharex=True,
        gridspec_kw={'height_ratios': [0.34, 1], 'hspace': 0.06})

    # Reference lane above the histogram. Keep a common gray style while
    # distinguishing point estimates, uncertainty, and reported ranges.
    lit_rows = [
        dict(kind='range', lo=carini_gamma_low, hi=carini_gamma_high,
             label=r'Carini et al. (2021)', duck=True),
        dict(kind='point', x=mccowan_gamma,
             label=r'McCowan (1891)'),
        dict(kind='point', x=thornton_guza_gamma_s,
             label=r'Thornton and Guza (1983)'),
        dict(kind='range', lo=sallenger_holman_gamma_s_low,
             hi=sallenger_holman_gamma_s_high,
             label=r'Sallenger and Holman (1985)', duck=True),
        dict(kind='range', lo=raubenheimer_gamma_s_low,
             hi=raubenheimer_gamma_s_high,
             label=r'Raubenheimer et al. (1996)', duck=True),
        dict(kind='range', lo=battjes_gamma_low, hi=battjes_gamma_high,
             label=r'Battjes (1974)'),
        dict(kind='range_point', lo=our_mode_low, hi=our_mode_high,
             x=our_mode, color=our_color, range_color=our_range_color,
             label=r'Ours ($h > 1$ m)'),
    ]
    for row, e in enumerate(lit_rows):
        if e['kind'] == 'point':
            ax_ref.plot(e['x'], row, marker=reference_marker,
                        color=e.get('color', reference_color),
                        markersize=reference_markersize,
                        linestyle='none', zorder=2)
        elif e['kind'] == 'range':
            center = 0.5 * (e['lo'] + e['hi'])
            half_range = 0.5 * (e['hi'] - e['lo'])
            ax_ref.errorbar(center, row, xerr=half_range, fmt='none',
                            ecolor=reference_color, elinewidth=0.8,
                            capsize=2, capthick=0.8, zorder=2)
        elif e['kind'] == 'range_point':
            center = 0.5 * (e['lo'] + e['hi'])
            half_range = 0.5 * (e['hi'] - e['lo'])
            ax_ref.errorbar(center, row, xerr=half_range, fmt='none',
                            ecolor=e['range_color'], elinewidth=0.8,
                            capsize=2, capthick=0.8, zorder=2)
            ax_ref.plot(e['x'], row, marker=reference_marker,
                        color=e['color'], markersize=reference_markersize,
                        linestyle='none', zorder=3)
        ax_ref.text(x_max - 0.05, row, e['label'], va='center', ha='right', fontsize=6)
        if e.get('duck'):
            ax_ref.text(x_max - 0.05, row, r'$^\dagger$',
                        va='center', ha='left', fontsize=6)

    ax_ref.set_ylim(len(lit_rows) - 0.5, -0.5)
    ax_ref.set_yticks([])
    ax_ref.tick_params(labelbottom=False)

    # Histogram, stacked by depth band.
    ax.hist([gv_all[mask] for mask in depth_masks], bins=bins,
            weights=[wt_all[mask] for mask in depth_masks], stacked=True,
            color=depth_colors, edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_xlabel(r'$\gamma_s = H_{s,offshore} / h$')
    ax.set_ylabel('Crest detections (millions)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))
    ax.set_xlim(0, x_max)

    # Segmented depth colorbar; listed shallow -> deep, top to bottom.
    colorbar_colors = depth_colors[::-1]
    colorbar_bounds = np.arange(len(colorbar_colors) + 1)
    depth_cmap = mcolors.ListedColormap(colorbar_colors)
    depth_norm = mcolors.BoundaryNorm(colorbar_bounds, depth_cmap.N)
    depth_mappable = plt.cm.ScalarMappable(norm=depth_norm, cmap=depth_cmap)
    colorbar_box = ax.inset_axes([0.76, 0.52, 0.20, 0.44], zorder=5)
    colorbar_box.set_facecolor('none')
    colorbar_box.set_xticks([])
    colorbar_box.set_yticks([])
    for spine in colorbar_box.spines.values():
        spine.set_visible(False)

    cax = colorbar_box.inset_axes([0.18, 0.10, 0.28, 0.80])
    cbar = fig.colorbar(depth_mappable, cax=cax, boundaries=colorbar_bounds,
                        ticks=[0, 2, 4, 6], drawedges=True)
    cbar.ax.invert_yaxis()
    cbar.ax.set_yticklabels(['0', '2', '4', '6'])
    cbar.set_label('Water depth (m)', rotation=270, labelpad=8, fontsize=6)
    cbar.ax.tick_params(direction='out', length=2, width=0.4, pad=2, labelsize=6)
    cbar.outline.set_linewidth(0.4)
    cbar.dividers.set_color('black')
    cbar.dividers.set_linewidth(0.4)

    fig.tight_layout(pad=0.3)

    # Summary stats
    gv, wt = gv_all, wt_all
    total  = wt.sum()
    pct    = wt[(gv >= raubenheimer_gamma_s_low) & (gv <= battjes_gamma_high)].sum() / total * 100
    pct_gt = wt[gv > 0.3].sum() / total * 100
    w_mean = np.average(gv, weights=wt)
    si     = np.argsort(gv)
    w_med  = gv[si][np.searchsorted(np.cumsum(wt[si]), 0.5 * total)]
    print(f"% in [{raubenheimer_gamma_s_low}, {battjes_gamma_high}]:  {pct:.1f}%")
    print(f"% gamma_s > 0.3:   {pct_gt:.1f}%")
    print(f"Weighted mean:     {w_mean:.4f}")
    print(f"Weighted median:   {w_med:.4f}")
    print(f"Median video-level crest-count-weighted half-sample mode (h > 1 m): "
          f"{our_mode:.4f}")
    print(f"  Video-level range: [{our_mode_low:.4f}, {our_mode_high:.4f}]")
    print(f"  Video-level modes: {np.array2string(video_modes, precision=4)}")

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
        total_depth = (ds_sub.depth + ds_sub.setup).values.flatten()
        gamma_s = (ds_sub.waveHs / (ds_sub.depth + ds_sub.setup)).values.flatten()
        counts  = ds_sub.crest_count.values.flatten()
        pixels_per_video = ds_sub.sizes['yFRF'] * ds_sub.sizes['xFRF']
        video_index = np.repeat(np.arange(ds_sub.sizes['time']), pixels_per_video)
        valid = ~(np.isnan(gamma_s) | np.isnan(counts) | np.isnan(total_depth)) & \
            (counts > 0) & (gamma_s > 0)

        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        print(f"=== {date_str}  (n_time={int(sel.sum())}) ===")

        fig = make_figure(total_depth[valid], gamma_s[valid], counts[valid],
                          video_index[valid])
        savefig(fig, f'{OUTPUT_STEM}_{date_str}.png')
        plt.close(fig)


if __name__ == '__main__':
    main()
