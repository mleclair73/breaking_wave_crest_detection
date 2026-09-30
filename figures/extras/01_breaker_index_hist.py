"""Breaker index histogram: observed Hs/h distribution vs. literature values."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from common.figure_style import apply_style, savefig, COL_W

DATA_PATH   = str(Path(__file__).parent.parent / 'combined_dunex_dataset_v2_lerp.nc')
OUTPUT_PATH = str(Path(__file__).parent / '01_breaker_index_hist.png')

PIER_MIN = 490
PIER_MAX = 530


def main():
    apply_style()

    ds = xr.open_dataset(DATA_PATH)

    gamma_s = (ds.waveHs / (ds.depth + ds.setup)).values.flatten()
    # gamma_s = (ds.waveHs / (ds.depth)).values.flatten()
    counts  = ds.observation_count.values.flatten()
    valid   = ~(np.isnan(gamma_s) | np.isnan(counts)) & (counts > 0) & (gamma_s > 0)

    H_rms_to_Hs = np.sqrt(2)  # Hs = sqrt(2) * Hrms for Rayleigh distribution

    # McCowan (1891)
    mccowan_gamma = 0.78

    # Thornton & Guza (1983) — reported as gamma_rms; convert to gamma_s for plotting
    thornton_guza_gamma_rms = 0.42
    thornton_guza_gamma_s   = thornton_guza_gamma_rms * H_rms_to_Hs

    # Sallenger & Holman (1985) — reported as gamma_rms range; convert to gamma_s for plotting
    sallenger_holman_gamma_rms_low  = 0.29
    sallenger_holman_gamma_rms_high = 0.55
    sallenger_holman_gamma_s_low    = sallenger_holman_gamma_rms_low  * H_rms_to_Hs
    sallenger_holman_gamma_s_high   = sallenger_holman_gamma_rms_high * H_rms_to_Hs
    print(f'{sallenger_holman_gamma_s_low=}')
    print(f'{sallenger_holman_gamma_s_high=}')

    # Raubenheimer (1996) — reported as gamma_s range
    raubenheimer_gamma_s_low  = 0.3
    raubenheimer_gamma_s_high = 0.7

    # Battjes (1974); see related discussion in Raubenheimer.
    battjes_gamma_low  = 0.7
    battjes_gamma_high = 1.2

    fig, ax = plt.subplots(figsize=(COL_W, COL_W))

    # Background literature ranges
    ax.axvspan(raubenheimer_gamma_s_low, raubenheimer_gamma_s_high, facecolor='none', edgecolor='forestgreen',
               hatch='//////', alpha=0.6, zorder=1)
    ax.axvspan(sallenger_holman_gamma_s_low, sallenger_holman_gamma_s_high, facecolor='none', edgecolor='indianred',
               hatch='\\\\\\\\\\\\', alpha=0.6, zorder=1)
    ax.axvspan(battjes_gamma_low, battjes_gamma_high, facecolor='none', edgecolor='darkgoldenrod',
               hatch='------', alpha=0.6, zorder=1)

    # Literature lines
    ax.axvline(mccowan_gamma,       color='black', linestyle='--', linewidth=1.0, zorder=2)
    ax.axvline(thornton_guza_gamma_s, color='black', linestyle='-',  linewidth=1.0, zorder=2)

    # Histogram
    x_max = 4.5
    bins = np.linspace(0, x_max, 30)
    ax.hist(gamma_s[valid], bins=bins, weights=counts[valid],
            color='dimgray', edgecolor='black', linewidth=0.4, zorder=3)

    ax.set_xlabel(r'$\gamma_s = H_{s,offshore} / h$')
    ax.set_ylabel(r'Observations $(\times 10^6)$')
    ax.set_box_aspect(1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}'))

    ax.set_xlim(0, x_max)

    legend_elements = [
        plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.0,
                   label=rf'McCowan (1891): $\gamma = {mccowan_gamma}$'),
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.0,
                   label=rf'Thornton & Guza (1983): $\gamma_{{rms}} = {thornton_guza_gamma_rms}$'),
        mpatches.Patch(edgecolor='indianred', facecolor='none',
                       hatch='\\\\\\\\\\\\',
                       label=rf'Sallenger & Holman (1985): $\gamma_{{rms}} \in [{sallenger_holman_gamma_rms_low}, {sallenger_holman_gamma_rms_high}]$'),
        mpatches.Patch(edgecolor='forestgreen', facecolor='none', hatch='//////',
                       label=rf'Raubenheimer (1996): $\gamma_s \in [{raubenheimer_gamma_s_low}, {raubenheimer_gamma_s_high}]$'),
        mpatches.Patch(edgecolor='darkgoldenrod', facecolor='none', hatch='------',
                       label=rf'Battjes (1974): $\gamma \in [{battjes_gamma_low}, {battjes_gamma_high}]$'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              framealpha=1.0, edgecolor='black', fontsize=6,
              borderpad=0.3, labelspacing=0.2, handletextpad=0.3, handlelength=1.2)

    fig.tight_layout(pad=0.3)
    savefig(fig, OUTPUT_PATH)

    # Summary stats
    gv, wt  = gamma_s[valid], counts[valid]
    total   = wt.sum()
    pct     = wt[(gv >= raubenheimer_gamma_s_low) & (gv <= battjes_gamma_high)].sum() / total * 100
    w_mean  = np.average(gv, weights=wt)
    si      = np.argsort(gv)
    w_med   = gv[si][np.searchsorted(np.cumsum(wt[si]), 0.5 * total)]
    hist, _ = np.histogram(gv, bins=bins, weights=wt)
    w_mode  = 0.5 * (bins[hist.argmax()] + bins[hist.argmax() + 1])
    print(f"% in [{raubenheimer_gamma_s_low}, {battjes_gamma_high}]:  {pct:.1f}%")
    print(f"Weighted mean:     {w_mean:.4f}")
    print(f"Weighted median:   {w_med:.4f}")
    print(f"Weighted mode:     {w_mode:.4f}  (bin center, width {bins[1]-bins[0]:.4f})")


if __name__ == '__main__':
    main()
