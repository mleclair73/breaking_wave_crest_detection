import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
import pandas as pd
import dunex_paths

from common.figure_style import (apply_style, savefig, panel_label, styled_legend,
                   add_colorbar, PAGE_W, CMAP)

G = 9.81
GAMMA = 0.55

from scipy.optimize import curve_fit

def fit_booij_gamma(depth, speed, weights):
    # Filter valid data
    mask = (np.isfinite(depth) & np.isfinite(speed) & 
            (depth > 0) & (speed > 0) & (weights > 0))
    h, c, w = depth[mask], speed[mask], weights[mask]

    # 1. Define the model function
    # c_squared = G * h * (1 + gamma / 2)
    def model(h_val, gamma):
        return G * h_val * (1 + gamma / 2)

    # 2. Use curve_fit
    # Since we are fitting c^2, we pass speed**2
    # curve_fit weights residuals by 1 / sigma**2.  Poisson-like count
    # weights therefore require sigma = 1 / sqrt(count), not 1 / count
    # (which would inadvertently apply count-squared weighting).
    popt, _ = curve_fit(model, h, c**2, sigma=1/np.sqrt(w), p0=[0.5])

    return popt[0]


DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"

def main():
    apply_style()

    ds = xr.open_dataset(DATASET)
    survey_dates = sorted(pd.unique(pd.to_datetime(ds.survey_time.values).normalize()))
    ds = ds.where((ds.survey_time == survey_dates[0]) |
                  (ds.survey_time == survey_dates[1]))
    counts = ds.observation_count.values.flatten()
    depth  = ds.depth.values
    setup  = ds.setup.values
    rmss    = np.sqrt(ds.mean_squared_speed.values)
    hs = ds.waveHs.values[:, np.newaxis, np.newaxis] * np.ones_like(depth)

    # Global hist2d settings (bins and range handled per-plot)
    HIST_BASE_KW = dict(
        cmap=CMAP['density'],
        norm=LogNorm(vmin=10),
        weights=counts,
        cmin=10,
    )

    fig, axes = plt.subplot_mosaic(
        [['a', 'b']],
        figsize=(PAGE_W + 0.6, 4.6),
        gridspec_kw={'wspace': 0.5},
    )
    ax_a, ax_b = axes['a'], axes['b']

    # ── (a) speed vs depth + setup ────────────────────────────────────────────
    # Range: X is 12 units (-2 to 10), Y is 12 units (0 to 12)
    # To make them square and small, we use a high equal number for both
    x_a = (depth + setup).flatten()
    y_a = rmss.flatten()
    
    _, _, _, im_a = ax_a.hist2d(x_a, y_a,
                                range=[[-2, 9], [0, 14]],
                                bins=(60, 60), # square bins (square box)
                                **HIST_BASE_KW)

    d = np.linspace(0, 9, 200)
    ax_a.plot(d, np.sqrt(G * d),                 'k-',  alpha=0.75, lw=1.2, label='Linear')
    ax_a.plot(d, np.sqrt(G * d * (1 + GAMMA/2)), 'k--', alpha=0.75, lw=1.2, label='Booij')
    ax_a.plot(d, np.sqrt(G * d * (1 + GAMMA)),   'k:',  alpha=0.75, lw=1.2, label='Solitary')


    styled_legend(ax_a, loc='lower right')
    ax_a.set_box_aspect(1)
    ax_a.set_xlim(-2, 9)
    ax_a.set_ylim(0, 14)
    ax_a.set_xlabel(r'$h + \bar{\eta}$ [m]')
    ax_a.set_ylabel('Observed RMS \nBreaker Speed [m/s]')
    add_colorbar(fig, ax_a, im_a, 'Observations')

    # ── (b) normalised speed vs γ ─────────────────────────────────────────────
    # Range: X is 4 units (0 to 4), Y is 3 units (0 to 3)
    # Ratio is 4:3. To get square bins in a square plot, bins must be 4:3 ratio.
    x_b = (hs / (depth + setup)).flatten()
    y_b = (rmss / np.sqrt((depth + setup) * G)).flatten()

    # ── Fit best-fit GAMMA for Booij model: speed = sqrt(G*h*(1 + GAMMA/2)) ──
    gamma_fit = fit_booij_gamma(x_a, y_a, counts.flatten())
    print(f'Best-fit Booij GAMMA: {gamma_fit:.4f}  (standard = 0.5)')

    _, _, _, im_b = ax_b.hist2d(x_b, y_b,
                                range=[[0, 6],
                                       [0, 6]],
                                bins=(60, 60), # square bins (square box)
                                **HIST_BASE_KW)

    # Bin-averaged overlay
    mask = np.isfinite(x_b) & np.isfinite(y_b) & (counts > 0)
    x_c, y_c, w_c = x_b[mask], y_b[mask], counts[mask]
    edges = np.linspace(0, 6, 20) # Slightly more bins for the overlay trend
    bin_means = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x_c >= lo) & (x_c < hi)
        if sel.sum() > 0:
            bin_means.append((np.average(x_c[sel], weights=w_c[sel]),
                              np.average(y_c[sel], weights=w_c[sel])))
    bin_means = np.array(bin_means)
    BESTFIT_C = "#94eafa"  # bright cyan — high contrast on RdPu density
    ax_b.plot(bin_means[:, 0], bin_means[:, 1], color=BESTFIT_C, alpha=0.8, lw=1.5, zorder=9, label='Mean')
    ax_b.scatter(bin_means[:, 0], bin_means[:, 1], s=15, c=BESTFIT_C, edgecolors='w', lw=0.3, zorder=10)

    gam = np.linspace(0.01, 6, 200)
    ax_b.axhline(1,                       color='k', ls='-', alpha=0.75, lw=1.2)
    ax_b.plot(gam, np.sqrt(1 + gam/2),    'k--',     alpha=0.75, lw=1.2)
    ax_b.plot(gam, np.sqrt(1 + gam),      'k:',      alpha=0.75, lw=1.2)


    ax_b.axvspan(
        0.3, 1.2, label='Surf Zone', zorder=-1, alpha=0.25, color='gray',
    )

    styled_legend(ax_b, loc='upper right')
    ax_b.set_box_aspect(1)
    ax_b.set_xlim(0, 6)
    ax_b.set_ylim(0, 6)
    ax_b.set_xlabel(r'$\gamma = H_s / (h + \bar{\eta})$  [–]')
    ax_b.set_ylabel('Observed RMS Speed / Linear Speed\n[–]')
    add_colorbar(fig, ax_b, im_b, 'Observations')

    # ── panel labels ──────────────────────────────────────────────────────────
    for ax, label in zip([ax_a, ax_b], ['(a)', '(b)']):
        panel_label(ax, label)

    fig.tight_layout()
    savefig(fig, Path(__file__).parent / '03_speed_depth_gamma.png')

if __name__ == '__main__':
    main()
