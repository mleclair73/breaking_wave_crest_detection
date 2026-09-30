"""Breaker index histogram (survey bathy): Hs/h using gridded FRF elevation-transect
surveys as the depth source, binned +/-5 m onto the survey lines. Depth = water_level -
survey_elevation (tide included). gamma_s = Hs / h (no setup). Nearest survey in time,
no fallback."""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from common.figure_style import apply_style, savefig, COL_W

DATA_PATH   = str(Path(__file__).parent.parent / 'combined_dunex_dataset_v2_lerp.nc')
SURVEY_PATH = str(Path(__file__).parent.parent / 'data' / 'bathy' / 'gridded_surveys.nc')
OUTPUT_PATH = str(Path(__file__).parent / '01_breaker_index_hist_survey.png')

BIN = 5.0  # m; +/-5 m alongshore band centred on each survey line (10 m bins)

PIER_MIN = 490  # m; exclude survey lines in the FRF pier alongshore band
PIER_MAX = 530


def survey_depth_binned(obs, surv):
    """Still-water depth from gridded survey transects, binned +/-BIN m onto the survey
    lines and mapped to each obs time by the nearest survey in time (no fallback).

    Returns gamma-ready arrays flattened over (time, line, xFRF):
    depth, obs_count, and per-time Hs broadcast to match.
    """
    obs_t  = pd.to_datetime(obs.time.values)
    surv_t = pd.to_datetime(surv.time.values)
    oy, ox = obs.yFRF.values, obs.xFRF.values
    wl     = obs.water_level.values
    hs     = obs.waveHs.values

    # survey lines inside the obs alongshore span, excluding the pier band
    yline = surv.yFRF.values
    keep  = ((yline >= oy.min()) & (yline <= oy.max())
             & ~((yline >= PIER_MIN) & (yline <= PIER_MAX)))
    yline = yline[keep]

    # survey elevation on the obs cross-shore grid: (survey_time, line, xFRF)
    elev = surv.elevation.interp(xFRF=ox).transpose('time', 'profile', 'xFRF').values[:, keep, :]

    # nearest survey time per obs time (strict; no fallback to non-empty slices)
    nearest = np.array([int(np.argmin(np.abs((surv_t - t).days))) for t in obs_t])

    depth  = np.full((len(obs_t), len(yline), len(ox)), np.nan)
    count  = np.full((len(obs_t), len(yline), len(ox)), np.nan)
    hs_b   = np.full((len(obs_t), len(yline), len(ox)), np.nan)
    for li, yl in enumerate(yline):
        band    = np.abs(oy - yl) <= BIN
        cnt_yx  = obs.observation_count.isel(yFRF=np.where(band)[0]).sum('yFRF').values  # (time, xFRF)
        for ti in range(len(obs_t)):
            depth[ti, li, :] = wl[ti] - elev[nearest[ti], li, :]
            count[ti, li, :] = cnt_yx[ti, :]
            hs_b[ti, li, :]  = hs[ti]

    return depth.ravel(), count.ravel(), hs_b.ravel()


def main():
    apply_style()

    obs  = xr.open_dataset(DATA_PATH)
    surv = xr.open_dataset(SURVEY_PATH)

    depth, counts, hs = survey_depth_binned(obs, surv)
    gamma_s = hs / depth
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
