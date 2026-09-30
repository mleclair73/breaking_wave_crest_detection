"""Breaker index histogram split by survey date: observed Hs/h vs. literature."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from common.figure_style import apply_style, savefig, PAGE_W, panel_label

DATA_PATH   = str(Path(__file__).parent.parent / 'combined_dunex_dataset_v2_lerp.nc')
OUTPUT_PATH = str(Path(__file__).parent / '01_breaker_index_hist_by_survey.png')

PIER_MIN = 490
PIER_MAX = 530


def _gamma_and_counts(ds_sub):
    """Flattened gamma_s and observation counts for a time-subset of the dataset."""
    gamma_s = (ds_sub.waveHs / (ds_sub.depth + ds_sub.setup)).values.flatten()
    counts  = ds_sub.observation_count.values.flatten()
    valid   = ~(np.isnan(gamma_s) | np.isnan(counts)) & (counts > 0) & (gamma_s > 0)
    return gamma_s[valid], counts[valid]


def main():
    apply_style()

    ds = xr.open_dataset(DATA_PATH)

    # Group time steps by calendar date of the survey.
    survey_dates = pd.to_datetime(ds.survey_time.values).normalize()
    unique_dates = sorted(pd.unique(survey_dates))

    H_rms_to_Hs = np.sqrt(2)  # Hs = sqrt(2) * Hrms for Rayleigh distribution

    # McCowan (1891)
    mccowan_gamma = 0.78

    # Thornton & Guza (1982) — reported as gamma_rms; convert to gamma_s for plotting
    thornton_guza_gamma_rms = 0.42
    thornton_guza_gamma_s   = thornton_guza_gamma_rms * H_rms_to_Hs

    # Sallenger & Holman (1985) — reported as gamma_rms range; convert to gamma_s for plotting
    sallenger_holman_gamma_rms_low  = 0.29
    sallenger_holman_gamma_rms_high = 0.50
    sallenger_holman_gamma_s_low    = sallenger_holman_gamma_rms_low  * H_rms_to_Hs
    sallenger_holman_gamma_s_high   = sallenger_holman_gamma_rms_high * H_rms_to_Hs

    # Raubenheimer (1996) — reported as gamma_s range
    raubenheimer_gamma_s_low  = 0.3
    raubenheimer_gamma_s_high = 0.6

    # Battjes (1974); see related discussion in Raubenheimer.
    battjes_gamma_low  = 0.7
    battjes_gamma_high = 1.2

    x_max = 4.5
    bins  = np.linspace(0, x_max, 30)

    n = len(unique_dates)
    fig, axes = plt.subplots(1, n, figsize=(PAGE_W, PAGE_W / n + 0.3),
                             sharey=True)
    axes = np.atleast_1d(axes)

    for i, (ax, date) in enumerate(zip(axes, unique_dates)):
        sel = survey_dates == date
        gv, wt = _gamma_and_counts(ds.isel(time=np.flatnonzero(sel)))

        # Background literature ranges
        ax.axvspan(raubenheimer_gamma_s_low, raubenheimer_gamma_s_high, facecolor='none',
                   edgecolor='forestgreen', hatch='//////', alpha=0.6, zorder=1)
        ax.axvspan(sallenger_holman_gamma_s_low, sallenger_holman_gamma_s_high, facecolor='none',
                   edgecolor='indianred', hatch='\\\\\\\\\\\\', alpha=0.6, zorder=1)
        ax.axvspan(battjes_gamma_low, battjes_gamma_high, facecolor='none',
                   edgecolor='darkgoldenrod', hatch='------', alpha=0.6, zorder=1)

        # Literature lines
        ax.axvline(mccowan_gamma,        color='black', linestyle='--', linewidth=1.0, zorder=2)
        ax.axvline(thornton_guza_gamma_s, color='black', linestyle='-',  linewidth=1.0, zorder=2)

        # Histogram
        ax.hist(gv, bins=bins, weights=wt,
                color='dimgray', edgecolor='black', linewidth=0.4, zorder=3)

        ax.set_xlabel(r'$\gamma_s = H_{s,offshore} / h$')
        if i == 0:
            ax.set_ylabel(r'Observations $(\times 10^6)$')
        ax.set_box_aspect(1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))
        ax.set_xlim(0, x_max)
        ax.set_title(pd.Timestamp(date).strftime('%Y-%m-%d'), fontsize=8)
        panel_label(ax, chr(ord('a') + i))

        # Summary stats
        total   = wt.sum()
        pct     = wt[(gv >= raubenheimer_gamma_s_low) & (gv <= battjes_gamma_high)].sum() / total * 100
        pct_gt  = wt[gv > 0.3].sum() / total * 100
        w_mean  = np.average(gv, weights=wt)
        si      = np.argsort(gv)
        w_med   = gv[si][np.searchsorted(np.cumsum(wt[si]), 0.5 * total)]
        hist, _ = np.histogram(gv, bins=bins, weights=wt)
        w_mode  = 0.5 * (bins[hist.argmax()] + bins[hist.argmax() + 1])
        print(f"=== {pd.Timestamp(date).date()}  (n_time={int(sel.sum())}) ===")
        print(f"% in [{raubenheimer_gamma_s_low}, {battjes_gamma_high}]:  {pct:.1f}%")
        print(f"% gamma_s > 0.3:   {pct_gt:.1f}%")
        print(f"Weighted mean:     {w_mean:.4f}")
        print(f"Weighted median:   {w_med:.4f}")
        print(f"Weighted mode:     {w_mode:.4f}  (bin center, width {bins[1]-bins[0]:.4f})")

    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.0,
                   label=rf'McCowan (1891): $\gamma = {mccowan_gamma}$'),
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.0,
                   label=rf'Thornton & Guza (1982): $\gamma_{{rms}} = {thornton_guza_gamma_rms}$'),
        mpatches.Patch(edgecolor='indianred', facecolor='none',
                       hatch='\\\\\\\\\\\\',
                       label=rf'Sallenger & Holman (1985): $\gamma_{{rms}} \in [{sallenger_holman_gamma_rms_low}, {sallenger_holman_gamma_rms_high}]$'),
        mpatches.Patch(edgecolor='forestgreen', facecolor='none', hatch='//////',
                       label=rf'Raubenheimer (1996): $\gamma_s \in [{raubenheimer_gamma_s_low}, {raubenheimer_gamma_s_high}]$'),
        mpatches.Patch(edgecolor='darkgoldenrod', facecolor='none', hatch='------',
                       label=rf'Battjes (1974): $\gamma \in [{battjes_gamma_low}, {battjes_gamma_high}]$'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', frameon=True,
                   framealpha=1.0, edgecolor='black', fontsize=6,
                   borderpad=0.3, labelspacing=0.2, handletextpad=0.3, handlelength=1.2)

    fig.tight_layout(pad=0.3)
    savefig(fig, OUTPUT_PATH)


if __name__ == '__main__':
    main()
