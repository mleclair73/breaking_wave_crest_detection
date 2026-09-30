"""Bathymetry planview (survey, inverted, difference) and cross-shore profiles."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import dunex_paths
from matplotlib import gridspec
from matplotlib.lines import Line2D
from common.figure_style import (
    CMAP,
    DEPTH_TICKS,
    DEPTH_VMAX,
    DEPTH_VMIN,
    FONTSIZE_LEGEND,
    LABEL_DEPTH,
    PAGE_W,
    apply_style,
    draw_pier,
    panel_label,
    savefig,
)
from common.figure_style import PROFILE_COLORS as _PROFILE_COLORS

DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"

GAMMA = 0.55
TIME_IDX = 3
XFRF_LIM = (60, 350)
YFRF_LIM = (-90, 1100)

PROFILE_Y = [1000, 0]
PROFILE_Y = [600, 200]
PROFILE_COLORS = _PROFILE_COLORS[1:3]   # two alongshore locations (skip blue)

XFRF_BIN = 12
YFRF_BIN = 24
MIN_OBS = 10
# Preserve the effective support gate used before observation counts were
# correctly kept as per-pixel means rather than 12×24-cell sums.
MIN_WAVE_COVERAGE = 0.15 / (XFRF_BIN * YFRF_BIN)

# Experimental: scale planview fill alpha by observation count (pcolormesh).
# Not ready for prime time — default off uses the original contourf rendering.
ALPHA_BY_OBS = False



def create_coarse_smooth(ds, dx_cross=12, dy_along=24, min_obs_count=15):
    spatial_vars = [v for v in ds.data_vars if 'xFRF' in ds[v].dims and 'yFRF' in ds[v].dims]
    non_spatial_vars = [v for v in ds.data_vars if v not in spatial_vars]
    ds_spatial = ds[spatial_vars]
    obs_mask = ds_spatial.observation_count >= min_obs_count
    ds_subset = ds_spatial.where(obs_mask).sel(xFRF=slice(-100, 1600))
    obs_subset = ds_spatial.observation_count.sel(xFRF=slice(-100, 1600))
    ds_coarse = ds_subset.coarsen(xFRF=dx_cross, yFRF=dy_along, boundary='pad').mean()
    # Coverage is a per-pixel fraction, so preserve the mean observation
    # count through coarsening.  Summing here would inflate coverage by the
    # 2-D bin area when it is converted below using count / duration * Tp.
    obs_count_coarse = obs_subset.coarsen(xFRF=dx_cross, yFRF=dy_along, boundary='pad').mean()
    ds_coarse['observation_count'] = obs_count_coarse
    ds_smoothed = ds_coarse.interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='linear')
    return xr.merge([ds_coarse, ds[non_spatial_vars]]), xr.merge([ds_smoothed, ds[non_spatial_vars]])


def create_coarse_profiles(ds, dx_cross=12, dy_along=24, min_obs_count=10):
    spatial_vars = [v for v in ds.data_vars if 'xFRF' in ds[v].dims and 'yFRF' in ds[v].dims]
    non_spatial_vars = [v for v in ds.data_vars if v not in spatial_vars]
    ds_spatial = ds[spatial_vars]
    obs_mask = ds_spatial.observation_count >= min_obs_count
    ds_subset = ds_spatial.where(obs_mask).sel(xFRF=slice(-100, 1600))
    obs_subset = ds_spatial.observation_count.sel(xFRF=slice(-100, 1600))
    ds_coarse = ds_subset.coarsen(xFRF=dx_cross, yFRF=dy_along, boundary='trim').mean()
    obs_coarse = obs_subset.coarsen(xFRF=dx_cross, yFRF=dy_along, boundary='trim').sum()
    ds_coarse['observation_count'] = obs_coarse
    return xr.merge([ds_coarse, ds[non_spatial_vars]])


def plot_planview(ax, cax, data, cmap, vmin, vmax, levels,
                  cbar_label, cbar_ticks=None, ylabel='', xlabel='',
                  waterline=None, land_over=False, hline_colors=None, hline_alpha=None, hline_style=None, hline_width=None,
                  obs_count=None, obs_ref=None, alpha_min=0.25):
    ax.set_facecolor('lightgray')
    if obs_count is None:
        im = data.plot.contourf(
            ax=ax, x='xFRF', y='yFRF', vmin=vmin, vmax=vmax,
            levels=levels, cmap=cmap, add_colorbar=False, extend='both', zorder=1)
    else:
        # Per-cell alpha scaled by observation count (requires pcolormesh).
        norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend='both')
        C = data.transpose('yFRF', 'xFRF')
        obs = obs_count.transpose('yFRF', 'xFRF')
        a = np.clip(obs.values / obs_ref, 0, 1) * (1 - alpha_min) + alpha_min
        a = np.where(np.isfinite(C.values), a, 0.0)
        a = np.nan_to_num(a, nan=0.0)
        im = ax.pcolormesh(C.xFRF.values, C.yFRF.values, C.values,
                           cmap=cmap, norm=norm, alpha=a,
                           shading='nearest', zorder=1)
    # Saddlebrown land mask: fill from the left axis to the waterline, i.e. where the
    # bed is above the total water surface (waterline field > 0). Drawn over the data
    # fill for panel (a) and under it for (b)/(c). The black waterline stays on top.
    if waterline is not None:
        land_z = 1.2 if land_over else 0.5
        land_cmap = mpl.colors.ListedColormap(['saddlebrown'])
        land_cmap.set_bad(alpha=0)
        xr.where(waterline > 0, 1.0, np.nan).plot.pcolormesh(
            ax=ax, x='xFRF', y='yFRF', cmap=land_cmap, add_colorbar=False, zorder=land_z)
        waterline.plot.contour(ax=ax, x='xFRF', y='yFRF', levels=[0],
                               colors=['black'], linewidths=1.0, zorder=4)
    line_colors = hline_colors if hline_colors is not None else PROFILE_COLORS
    line_alpha = hline_alpha if hline_alpha is not None else 1
    line_style = hline_style if hline_style is not None else '--'
    line_width = hline_width if hline_width is not None else 1.2
    for y_val, color in zip(PROFILE_Y, line_colors):
        ax.axhline(y_val, color=color, linewidth=line_width, linestyle=line_style, alpha=line_alpha, zorder=5)
    draw_pier(ax, 'horizontal')
    ax.set_xlim(XFRF_LIM)
    ax.set_ylim(YFRF_LIM)
    ax.set_title('')
    ax.set_xlabel(xlabel)      # set after plot.contour, which resets axis labels
    ax.set_ylabel(ylabel)
    # Rasterize the filled field (zorder 1) into one composited 300-dpi image so the
    # PDF stays small; waterline (4), profile lines (5) and pier (6) stay vector.
    ax.set_rasterization_zorder(1.5)
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(cbar_label, labelpad=2)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    return im


def main():
    apply_style()

    ds = xr.open_dataset(DATASET)
    # booij_depth is the total wave-derived water depth — it already includes setup.
    ds['booij_depth'] = ds.mean_squared_speed / (9.81 * (1 + GAMMA / 2))

    _, ds_smoothed = create_coarse_smooth(ds, XFRF_BIN, YFRF_BIN, MIN_OBS)
    depth_smooth = (ds.depth
                    .coarsen(xFRF=12, yFRF=24, boundary='pad').mean()
                    .interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='linear'))
    wave_coverage = ds_smoothed.observation_count / ds_smoothed.video_duration * ds_smoothed.waveTp
    obs_mask = (wave_coverage > MIN_WAVE_COVERAGE).compute()

    # Total water depth = water_level - elevation + setup = total_surface - elevation,
    # where total_surface = water_level + setup is the setup-raised water surface.
    # Survey bed elevation = water_level - depth, so the survey total depth is depth+setup.
    # booij_depth is already the total (setup-inclusive) wave-derived water depth, so it
    # IS the inverted total depth directly.
    # Unmasked total surface so the waterline contour has no gaps where obs were sparse
    # (ds_smoothed fields are obs-masked; ds fields are defined everywhere the survey is).
    total_surface = (ds.water_level + ds.setup).isel(time=TIME_IDX)   # total water-surface elevation
    survey_elev = (ds.water_level - depth_smooth).isel(time=TIME_IDX)
    survey_depth = total_surface - survey_elev
    inverted_depth = ds_smoothed.booij_depth.where(obs_mask).isel(time=TIME_IDX)
    # Waterline marker: bed elevation minus the total water surface; its 0-contour is
    # where the bed meets water level + setup (i.e. zero water depth, varies alongshore).
    waterline = survey_elev - total_surface
    diff = inverted_depth - survey_depth   # inverted_depth already obs-masked
    obs_pv = ds_smoothed.observation_count.where(obs_mask).isel(time=TIME_IDX)
    obs_pv_ref = float(np.nanpercentile(obs_pv.values, 75))
    obs_kw = {'obs_count': obs_pv, 'obs_ref': obs_pv_ref} if ALPHA_BY_OBS else {}

    # Profile datasets
    ds_p = create_coarse_profiles(ds, XFRF_BIN, YFRF_BIN, MIN_OBS)
    ds_p['booij_depth'] = ds_p.mean_squared_speed / (9.81 * (1 + GAMMA / 2))
    ds_p = ds_p.sel(xFRF=slice(0, 500))
    ds_t = ds_p.isel(time=TIME_IDX)
    # Ground-truth survey bed depth: coarsen the survey total depth directly (untrimmed,
    # NO obs mask) so the transect is continuous everywhere the survey covers.
    survey_depth_coarse = ((ds.water_level + ds.setup - ds.elevation)
                           .coarsen(xFRF=XFRF_BIN, yFRF=YFRF_BIN, boundary='pad').mean()
                           .isel(time=TIME_IDX))
    obs_max = float(ds_t.observation_count.max().values)

    # Size-legend levels
    obs_max_k = obs_max / 1000
    magnitude_k = 10 ** np.floor(np.log10(obs_max_k))
    nice_levels_k = np.array([1, 2, 5, 10, 20, 50, 100]) * magnitude_k
    obs_levels_k = nice_levels_k[(nice_levels_k >= obs_max_k * 0.1) & (nice_levels_k <= obs_max_k)]
    if len(obs_levels_k) < 2:
        obs_levels_k = np.array([round(obs_max_k * 0.25), round(obs_max_k)])
    obs_levels = obs_levels_k * 1000

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(PAGE_W, 8.0))

    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[5, 2],
                               hspace=0.18, left=0.10, right=0.97,
                               top=0.97, bottom=0.06)

    top = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[0],
        height_ratios=[18, 1], hspace=0.25, wspace=0.35)
    ax_a  = fig.add_subplot(top[0, 0])
    ax_b  = fig.add_subplot(top[0, 1])
    ax_c  = fig.add_subplot(top[0, 2])
    cax_a = fig.add_subplot(top[1, 0])
    cax_b = fig.add_subplot(top[1, 1])
    cax_c = fig.add_subplot(top[1, 2])

    bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[1],
        width_ratios=[1, 1, 0.05], wspace=0.12)
    ax_d    = fig.add_subplot(bot[0, 0])
    ax_e    = fig.add_subplot(bot[0, 1], sharey=ax_d)
    cax_err = fig.add_subplot(bot[0, 2])

    # ── Colormaps ──────────────────────────────────────────────────────────────
    # Depth colormap: high (deep) = dark. CMAP['depth'] is reversed for elevation space
    # (deep = most negative); in depth space we use the un-reversed map.
    cmap_depth = mpl.colormaps[CMAP['depth'].replace('_r', '')].copy()
    cmap_depth.set_bad(alpha=0)
    # Depth-space diff (inverted − survey) uses the reversed diverging map so the
    # colours read the same as before the sign flip: inverted reads deeper than
    # survey one way, shallower the other.
    cmap_diff = mpl.colormaps[CMAP['diverge']].reversed()
    cmap_diff.set_bad(alpha=0)

    # ── Planview panels ────────────────────────────────────────────────────────
    depth_levels = np.arange(DEPTH_VMIN, DEPTH_VMAX + 0.01, 0.5)
    plot_planview(ax_a, cax_a, survey_depth, cmap_depth,
                  DEPTH_VMIN, DEPTH_VMAX, depth_levels, 'Survey ' + LABEL_DEPTH,
                  cbar_ticks=DEPTH_TICKS, ylabel='Alongshore distance [m]',
                  waterline=waterline, land_over=True)

    plot_planview(ax_b, cax_b, inverted_depth, cmap_depth,
                  DEPTH_VMIN, DEPTH_VMAX, depth_levels, 'Inverted ' + LABEL_DEPTH,
                  cbar_ticks=DEPTH_TICKS, xlabel='Cross-shore distance [m]',
                  waterline=waterline, hline_colors=['black', 'black'], hline_alpha=0.4, hline_style='-', hline_width=1, **obs_kw)
    ax_b.set_title('')
    ax_b.set_xlabel('Cross-shore distance [m]')
    ax_b.set_ylabel('')
    ax_b.tick_params(labelleft=False)
    leg_b = ax_b.legend(
        handles=[
            Line2D([0], [0], color='black', lw=1.0, label='Survey waterline'),
            Line2D([0], [0], color='dimgray', lw=3.5, label='FRF Pier'),
        ],
        loc='lower right', fontsize=FONTSIZE_LEGEND, framealpha=0.9)
    leg_b.get_frame().set_edgecolor('none')

    DIFF_SCALE = 1
    plot_planview(ax_c, cax_c, diff, cmap_diff,
                  -DIFF_SCALE, DIFF_SCALE, np.linspace(-DIFF_SCALE, DIFF_SCALE, 9), 'Inverted \u2212 Survey [m]',
                  cbar_ticks=[-1, 0, 1], waterline=waterline,
                  hline_colors=['black', 'black'], hline_alpha=0.4, hline_style='-', hline_width=1, **obs_kw)
    ax_c.tick_params(labelleft=False)

    # ── Profile panels ─────────────────────────────────────────────────────────
    GAMMA_THRESHOLD = 0.01
    scatter_handles = []

    for (ax, y_val, show_ylabel), color in zip(
            [(ax_d, PROFILE_Y[0], True), (ax_e, PROFILE_Y[1], False)], PROFILE_COLORS):
        ds_at_y = ds_t.sel(yFRF=y_val, method='nearest')
        gamma_at_y = (ds_t.waveHs / ds_t.depth).sel(yFRF=y_val, method='nearest')
        mask = (gamma_at_y >= GAMMA_THRESHOLD).compute()

        # Ground-truth survey bed depth at the untrimmed coarse (pad) resolution, with
        # each node marked so the transect sampling is visible.
        ds_coarse_y = survey_depth_coarse.sel(yFRF=y_val, method='nearest')
        ax.plot(ds_coarse_y.xFRF, ds_coarse_y, linewidth=1.5, color=color,
                marker='o', markersize=3, markerfacecolor='white',
                markeredgecolor=color, markeredgewidth=0.6, zorder=6)
        valid = ds_at_y.where(mask, drop=True)
        sc = None
        if len(valid.xFRF) > 0:
            inv_depth_p = valid.booij_depth                       # inverted total depth
            surv_depth_p = valid.depth + valid.setup              # survey total depth
            error = (inv_depth_p - surv_depth_p).values
            obs_vals = valid.observation_count.values
            ax.plot(valid.xFRF, inv_depth_p,
                    color='gray', linewidth=0.8, alpha=0.6, zorder=9)
            sc = ax.scatter(valid.xFRF, inv_depth_p,
                            c=error, s=obs_vals / obs_max * 150,
                            cmap=cmap_diff, vmin=-1, vmax=1,
                            alpha=0.85, edgecolor='k', linewidth=0.1, zorder=10)
            scatter_handles.append(sc)

        # Depth increases downward (inverted y-axis) for a natural cross-section view.
        ax.set_ylim(DEPTH_VMAX, -1)
        ax.set_xlim(75, 350)
        ax.set_title(f'yFRF = {y_val} m')
        ax.set_xlabel('Cross-shore distance [m]')
        ax.grid(True, alpha=0.3)
        if show_ylabel:
            ax.set_ylabel(LABEL_DEPTH)
        else:
            ax.tick_params(labelleft=False)

    # Size legend on ax_d
    size_handles = [
        plt.scatter([], [], s=o / obs_max * 150, c='gray', alpha=0.9,
                    edgecolors='none', label=f'{o/1000:.0f}')
        for o in obs_levels]
    ax_d.legend(handles=size_handles, title='Obs (\u00d71000)',
                loc='lower left', framealpha=0.9,
                fontsize=FONTSIZE_LEGEND - 1,
                title_fontsize=FONTSIZE_LEGEND,
                labelspacing=0.5, borderpad=0.5, handletextpad=0.5)

    # Profile error colorbar
    if scatter_handles:
        cbar_err = fig.colorbar(scatter_handles[-1], cax=cax_err)
        cbar_err.set_label('Inverted \u2212 Survey [m]')
        cbar_err.set_ticks([-1, 0, 1])

    # ── Panel labels ───────────────────────────────────────────────────────────
    for ax, label in zip([ax_a, ax_b, ax_c, ax_d, ax_e],
                          ['(a)', '(b)', '(c)', '(d)', '(e)']):
        panel_label(ax, label)

    savefig(fig, Path(__file__).parent / '04_bathy_planview.png')


if __name__ == '__main__':
    main()
