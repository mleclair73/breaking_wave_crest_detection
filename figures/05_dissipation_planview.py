"""Dissipation planview: inverted depth, Qb, dissipation, and integrated D."""

import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import newton
import dunex_paths

from common.figure_style import (apply_style, savefig, panel_label, styled_legend,
                   draw_pier, draw_land, CMAP, COLOR_SURVEY, IBM, PIER_Y,
                   LABEL_DEPTH, DEPTH_VMIN, DEPTH_VMAX, DEPTH_TICKS)

PIER_HALF_WIDTH = 25
PIER_MIN = PIER_Y - PIER_HALF_WIDTH
PIER_MAX = PIER_Y + PIER_HALF_WIDTH

DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"


def nice_ceil(val):
    if val <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(val))
    norm = val / mag
    for nice in [1, 1.5, 2, 2.5, 3, 4, 5, 8, 10]:
        if norm <= nice:
            return nice * mag
    return 10 * mag


def main():
    apply_style()

    print("Loading datasets...")
    ds = xr.open_dataset(DATASET)

    ds_lerp = xr.open_dataset(DATASET)
    print(f"  {len(ds.time)} timesteps, xFRF: {ds.xFRF.values[0]:.0f}–{ds.xFRF.values[-1]:.0f} m")

    # Parameters
    GAMMA = 0.55
    RHO = 1025.0
    G = 9.81
    ALPHA = 1.0
    TIME_IDX = 3
    DX_CROSS = 10
    DY_ALONG = 10
    MIN_OBS = 1
    FRF_OFFSET = -18.2

    print(f'Good Minutes of Data: {(ds.isel(time=TIME_IDX).trimmed_frames / ds.isel(time=TIME_IDX).fps/60).item()}')

    ds['booij_depth'] = ds.mean_squared_speed / (G * (1 + GAMMA / 2))

    # Coarsen
    print("Coarsening and smoothing spatial fields...")
    spatial_vars = [v for v in ds.data_vars
                    if 'xFRF' in ds[v].dims and 'yFRF' in ds[v].dims]
    non_spatial_vars = [v for v in ds.data_vars if v not in spatial_vars]
    ds_sp = ds[spatial_vars]
    ds_nsp = ds[non_spatial_vars]

    # Match 04_bathy_planview's create_coarse_smooth: larger windows + boundary='pad'
    # so coarse cells survive on as little as one valid pixel in the wider footprint
    # and interp spans the full domain, avoiding the speckled NaN dropout that small
    # 'trim' windows produce.
    obs_mask = ds_sp.observation_count >= MIN_OBS
    ds_masked = ds_sp.where(obs_mask).sel(xFRF=slice(-100, 1600))
    obs_subset = ds_sp.observation_count.sel(xFRF=slice(-100, 1600))
    ds_coarse = ds_masked.coarsen(xFRF=DX_CROSS, yFRF=DY_ALONG, boundary='pad').mean()
    obs_coarse = obs_subset.coarsen(xFRF=DX_CROSS, yFRF=DY_ALONG, boundary='pad').sum()
    # Keep per-pixel normalization so Qb stays a pixel-level breaking fraction.
    ds_coarse['observation_count'] = obs_coarse / (DX_CROSS * DY_ALONG)

    ds_smooth = ds_coarse.interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='linear')
    ds_smooth = xr.merge([ds_smooth, ds_nsp])

    # Qb and dissipation
    print("Computing Qb and dissipation rate D...")
    # Qb = Nb / Ntotal = crest_count / (t_total / Tm), with t_total = trimmed_frames / fps.
    # crest_count is distinct crest tracks per pixel, so this is fps-invariant by construction.
    qb = (ds_smooth.crest_count * ds_smooth.fps / ds_smooth.trimmed_frames
          * ds_smooth.waveTm).clip(max=1)
    # observation_count is defined everywhere, so restrict Qb to where the coarsened
    # physical fields are actually valid (obs >= MIN_OBS survived the coarsen-mean).
    # booij_depth is interpolated to native res, so Qb stays at native resolution.
    qb = qb.where(ds_smooth.booij_depth.notnull())
    booij_depth = ds_smooth.booij_depth.clip(min=0)
    hmax = GAMMA * booij_depth
    fm = 1.0 / ds_smooth.waveTm
    D = (ALPHA / 4.0) * qb * fm * RHO * G * hmax**2

    # Quality masks
    bad_mask_2d = xr.open_dataset(
        dunex_paths.DATA_ROOT / 'video_dataset/good_data_mask.nc'
    ).bad_mask.astype(int).interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='nearest').fillna(True).astype(bool)
    bad_row_mask = bad_mask_2d.mean(dim='xFRF') > 0.10
    analysis_mask = (qb > 0.0) & ~bad_row_mask

    # min_count=50 -> rows with no valid cells become NaN (a gap in panel d) rather than summing to 0.
    integrated_D = D.where(analysis_mask).sum(dim='xFRF', min_count=50)

    # Incident energy flux via dispersion relation, solved only for the depth at
    # the displayed time index (freq/dir bins are static across time).
    print("Solving dispersion relation...")
    h_8m = float(ds.depth_8m_array.isel(time=TIME_IDX))
    omega_vals = (2 * np.pi * ds.waveFrequency).values
    k0 = omega_vals**2 / G
    k_vals = newton(
        func=lambda k: omega_vals**2 - G * k * np.tanh(k * h_8m),
        x0=k0,
        fprime=lambda k: -G * (np.tanh(k * h_8m) + k * h_8m / np.cosh(k * h_8m)**2),
    )
    k = xr.DataArray(k_vals, dims=['waveFrequency'],
                     coords={'waveFrequency': ds.waveFrequency})
    kh = k * h_8m
    omega_da = 2 * np.pi * ds.waveFrequency
    c_p = omega_da / k
    c_g = 0.5 * c_p * (1 + 2 * kh / np.sinh(2 * kh))

    E_fd = ds.directionalWaveEnergyDensity.isel(time=TIME_IDX)
    n_east  = -np.sin(np.deg2rad(ds.waveDirectionBins))
    n_north = -np.cos(np.deg2rad(ds.waveDirectionBins))
    F_east  = (RHO * G * E_fd * c_g * n_east ).integrate('waveFrequency').integrate('waveDirectionBins')
    F_north = (RHO * G * E_fd * c_g * n_north).integrate('waveFrequency').integrate('waveDirectionBins')
    shore_normal = np.deg2rad(90.0 + FRF_OFFSET)
    Fx_val = float(-(F_east * np.sin(shore_normal) + F_north * np.cos(shore_normal)))
    print(f"  Onshore flux Fx (t={TIME_IDX}): {Fx_val:.1f} W/m")

    # Shallow-water comparison: non-dispersive c_g = c_p = sqrt(g*h) for all freqs. The takeaway was that this substantially overestimates
    c_g_sw = np.sqrt(G * h_8m)
    F_east_sw  = (RHO * G * E_fd * c_g_sw * n_east ).integrate('waveFrequency').integrate('waveDirectionBins')
    F_north_sw = (RHO * G * E_fd * c_g_sw * n_north).integrate('waveFrequency').integrate('waveDirectionBins')
    Fx_sw = float(-(F_east_sw * np.sin(shore_normal) + F_north_sw * np.cos(shore_normal)))
    print(f"  Onshore flux Fx shallow water: {Fx_sw:.1f} W/m  "
          f"(shallow/full = {Fx_sw / Fx_val:.2f})")

    # avg_D and incident rate for selected time slice
    t = TIME_IDX
    yvals = ds.yFRF.values
    dx = float(ds.xFRF.diff('xFRF').mean())
    intD_t = integrated_D.isel(time=t).compute()

    n_valid = analysis_mask.isel(time=t).sum(dim='xFRF').astype(float)
    Xs_valid = n_valid * abs(dx)

    # Block-average into true DY_ALONG (10 m) alongshore bins for panel (d): the
    # native 1 m curve above inherits the fake smoothness of ds_smooth's linear
    # interp (line 75), so its "steps" are indistinguishable from a sloped line.
    # Summing D and the valid cross-shore length within each real bin before
    # dividing gives the properly weighted 10 m average to plot as genuine steps.
    n_bins = len(yvals) // DY_ALONG
    intD_10m = intD_t.values[:n_bins * DY_ALONG].reshape(n_bins, DY_ALONG).sum(axis=1)
    Xs_10m = Xs_valid.values[:n_bins * DY_ALONG].reshape(n_bins, DY_ALONG).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_D_10m = intD_10m / Xs_10m
    y_10m = yvals[:n_bins * DY_ALONG].reshape(n_bins, DY_ALONG).mean(axis=1)

    h_local = ds_lerp.depth.isel(time=t).clip(min=0.01)
    Hs_t = float(ds.waveHs.isel(time=t))
    breaker_index = Hs_t / h_local
    Xs_lo = ((breaker_index > 0.3).sum(dim='xFRF').astype(float) * abs(dx)).values
    Xs_hi = ((breaker_index > 0.7).sum(dim='xFRF').astype(float) * abs(dx)).values
    with np.errstate(divide='ignore', invalid='ignore'):
        inc_lo = np.where(Xs_lo > 0, Fx_val / Xs_lo, np.nan)
        inc_hi = np.where(Xs_hi > 0, Fx_val / Xs_hi, np.nan)

    # ---- Figure ----
    xlim = (50, 350)       # cross-shore extent (plot y-axis)
    ylim = (1100, -90)     # alongshore extent (plot x-axis), reversed
    # Force every map panel to the data aspect ratio via set_box_aspect; this lets
    # constrained_layout pack the short, wide panels tightly with no dead space —
    # no manual inch-by-inch repositioning.
    box_aspect = (xlim[1] - xlim[0]) / (ylim[0] - ylim[1])

    # Inverted total water depth: booij_depth is already the setup-inclusive
    # wave-derived depth, so it IS the inverted depth directly, consistent with fig 04.
    inv_depth = ds_smooth.booij_depth

    fig, axes = plt.subplot_mosaic(
        [['A', 'a'],
         ['B', 'b'],
         ['C', 'c'],
         ['D', 'd']],
        width_ratios=[1, 0.03],
        figsize=(5.5, 6.0),
        constrained_layout=True,
    )
    # Tighten vertical packing: the box-aspect maps are short, so trim the space
    # constrained_layout leaves between rows.
    fig.get_layout_engine().set(h_pad=0.02, hspace=0.0)
    # Same box aspect on all four panels (D included) so every panel renders at
    # identical height and constrained_layout spaces them evenly — matching the
    # original even-height stack without any manual inch positioning.
    for key in ['A', 'B', 'C', 'D']:
        axes[key].set_box_aspect(box_aspect)
    for key in ['B', 'C', 'D']:
        axes[key].sharex(axes['A'])
    for key in ['A', 'B', 'C']:
        axes[key].tick_params(labelbottom=False)
    axes['d'].set_visible(False)   # reserve colourbar width so (d) aligns with maps

    # Depth colormap: high (deep) = dark. CMAP['depth'] is reversed for elevation
    # space; in depth space we use the un-reversed map.
    depth_cmap = mpl.colormaps[CMAP['depth'].replace('_r', '')].copy()
    depth_cmap.set_bad(color=(0, 0, 0, 0))
    qb_cmap = mpl.colormaps[CMAP['Qb']].copy()
    qb_cmap.set_bad(color=(0, 0, 0, 0))
    diss_cmap = mpl.colormaps[CMAP['diss']].copy()
    diss_cmap.set_bad(color=(0, 0, 0, 0))

    depth_t = ds_lerp.depth.isel(time=t)

    def format_map(ax):
        ax.set_facecolor('lightgray')
        draw_land(ax, depth_t, x='yFRF', y='xFRF')
        draw_pier(ax, 'vertical')
        ax.set_xlim(ylim)
        ax.set_ylim(xlim)
        ax.set_title('')
        ax.set_ylabel('')
        ax.set_xlabel('')
        # Rasterize everything below zorder 2 (land fill at 0, data mesh at 1) into
        # one composited layer so land stays behind the mesh in the vector PDF;
        # shoreline (zorder 2), pier (6) and labels stay crisp vector.
        ax.set_rasterization_zorder(2)

    # (a) Qb
    im_a = qb.isel(time=t).plot.pcolormesh(
        ax=axes['A'], y='xFRF', x='yFRF', vmin=0, vmax=1,
        cmap=qb_cmap, add_colorbar=False, zorder=1)
    format_map(axes['A'])
    axes['A'].tick_params(labelleft=False)   # avoid y-tick overlap with (b)
    panel_label(axes['A'], '(a)')
    styled_legend(axes['A'], loc='upper right')
    cbar_a = fig.colorbar(im_a, cax=axes['a'])
    cbar_a.set_label('$Q_b$', labelpad=4)
    cbar_a.set_ticks([0, 0.5, 1.0])

    # (b) Inverted depth — shared depth scale with fig 04
    im_b = inv_depth.isel(time=t).plot.pcolormesh(
        ax=axes['B'], y='xFRF', x='yFRF', vmin=DEPTH_VMIN, vmax=DEPTH_VMAX,
        cmap=depth_cmap, add_colorbar=False, zorder=1)
    format_map(axes['B'])
    panel_label(axes['B'], '(b)')
    cbar_b = fig.colorbar(im_b, cax=axes['b'])
    cbar_b.set_label('Inverted ' + LABEL_DEPTH, labelpad=4)
    cbar_b.set_ticks(DEPTH_TICKS)

    # (c) Dissipation
    D_at_t = D.isel(time=t)
    D_vals = D_at_t.values[np.isfinite(D_at_t.values)]
    D_99 = float(np.nanpercentile(D_vals, 99)) if len(D_vals) > 0 else 1.0
    D_vmax = nice_ceil(D_99)
    im_c = D_at_t.plot.pcolormesh(
        ax=axes['C'], y='xFRF', x='yFRF', vmin=0, vmax=D_vmax,
        cmap=diss_cmap, add_colorbar=False, zorder=1)
    format_map(axes['C'])
    panel_label(axes['C'], '(c)')
    axes['B'].set_ylabel('Cross-shore distance [m]')   # centred over the map block
    cbar_c = fig.colorbar(im_c, cax=axes['c'])
    cbar_c.set_label('D [W/m²]', labelpad=4)
    cbar_c.set_ticks(np.linspace(0, D_vmax, 3))

    # (d) Average dissipation vs incident rate
    ax = axes['D']
    ax.axvline(PIER_Y, color='dimgray', linewidth=3.5, linestyle='-', zorder=3)
    ax.fill_between(yvals, inc_lo, inc_hi, alpha=0.3, color=IBM[3],
                    label='Incident Rate')
    ax.plot(y_10m, avg_D_10m, '-', drawstyle='steps-mid', color=COLOR_SURVEY, lw=1.5, label='Average Estimate')
    ax.set_xlim(ylim)
    ax.set_xlabel('Alongshore distance [m]')
    ax.set_ylabel('Dissipation Rate\n[W/m²]')
    panel_label(ax, '(d)')
    styled_legend(ax, loc='upper right', ncols=1, fontsize=6)
    ax.grid(alpha=0.3)

    # Freeze the constrained layout, then match each colourbar to its panel height.
    fig.canvas.draw()
    fig.set_layout_engine('none')
    # Use (c)'s x0 as the common left edge so all three bars line up vertically;
    # constrained_layout leaves (a)/(b) slightly left of (c), so bump them right.
    cbar_x0 = axes['c'].get_position().x0
    for pk, ck in [('A', 'a'), ('B', 'b'), ('C', 'c')]:
        p = axes[pk].get_position()
        c = axes[ck].get_position()
        axes[ck].set_position([cbar_x0, p.y0, c.width, p.height])

    # Right-align the three colourbar labels to a common figure x. Tick labels have
    # differing widths (e.g. "250" vs "4"), which otherwise pushes each rotated label
    # to a different x; place them all just right of the widest tick label.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    max_right = max(lbl.get_window_extent(renderer).x1
                    for ck in ['a', 'b', 'c']
                    for lbl in axes[ck].get_yticklabels())
    label_x = fig.transFigure.inverted().transform((max_right, 0))[0] + 0.012
    for ck in ['a', 'b', 'c']:
        pos = axes[ck].get_position()
        axes[ck].yaxis.set_label_coords(
            label_x, pos.y0 + pos.height / 2, transform=fig.transFigure)

    savefig(fig, Path(__file__).parent / '05_dissipation_planview.png')


if __name__ == '__main__':
    main()
