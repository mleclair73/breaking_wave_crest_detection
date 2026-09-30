"""Dissipation ratio histogram: avg Qb dissipation / incident dissipation rate."""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import newton
from scipy.stats import skew
import pandas as pd
import dunex_paths

from common.figure_style import apply_style, savefig, styled_legend, COL_W, IBM

PIER_MIN = 490
PIER_MAX = 530
DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"


def main():
    apply_style()
    print("Loading combined dataset...")
    ds = xr.open_dataset(DATASET)
    survey_dates = sorted(pd.unique(pd.to_datetime(ds.survey_time.values).normalize()))
    ds = ds.where((ds.survey_time == survey_dates[0]) |
                    (ds.survey_time == survey_dates[1]))
    print(f"  {len(ds.time)} timesteps, xFRF: {ds.xFRF.values[0]:.0f}–{ds.xFRF.values[-1]:.0f} m")

    # Parameters
    GAMMA = 0.55
    RHO = 1025.0
    G = 9.81
    ALPHA = 1.0
    DX_CROSS = 5
    DY_ALONG = 5
    MIN_OBS = 5
    FRF_OFFSET = -18.2

    ds['booij_depth'] = ds.mean_squared_speed / (G * (1 + GAMMA / 2))

    # Coarsen
    print("Coarsening and smoothing spatial fields...")
    spatial_vars = [v for v in ds.data_vars
                    if 'xFRF' in ds[v].dims and 'yFRF' in ds[v].dims]
    non_spatial_vars = [v for v in ds.data_vars if v not in spatial_vars]
    ds_sp = ds[spatial_vars]
    ds_nsp = ds[non_spatial_vars]

    obs_mask = ds_sp.observation_count >= MIN_OBS
    ds_masked = ds_sp.where(obs_mask)
    ds_coarse = ds_masked.coarsen(xFRF=DX_CROSS, yFRF=DY_ALONG, boundary='trim').mean()
    obs_coarse = ds_sp.observation_count.coarsen(
        xFRF=DX_CROSS, yFRF=DY_ALONG, boundary='trim').sum()
    ds_coarse['observation_count'] = obs_coarse / (DX_CROSS * DY_ALONG)

    ds_smooth = ds_coarse.interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='linear')
    for dim in ['xFRF', 'yFRF']:
        ds_smooth = ds_smooth.ffill(dim).bfill(dim)
    ds_smooth = xr.merge([ds_smooth, ds_nsp])

    # Qb, dissipation
    print("Computing Qb and dissipation rate D...")
    # Qb = Nb / Ntotal = crest_count / (t_total / Tm), with t_total = trimmed_frames / fps.
    # crest_count is distinct crest tracks per pixel, so this is fps-invariant by construction.
    qb = (ds_smooth.crest_count * ds_smooth.fps / ds_smooth.trimmed_frames * ds_smooth.waveTm).clip(max=1)
    booij_depth = ds_smooth.booij_depth.clip(min=0)
    hmax = GAMMA * booij_depth
    fm = 1.0 / ds_smooth.waveTm
    D = (ALPHA / 4.0) * qb * fm * RHO * G * hmax**2 # * GAMMA (Dropped from original BJ78)

    bad_mask_2d = xr.open_dataset(
        dunex_paths.DATA_ROOT / 'video_dataset/good_data_mask.nc'
    ).bad_mask.astype(int).interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='nearest').fillna(True).astype(bool)

    # Treat pier as bad pixels so it feeds into bad_row_mask naturally
    pier_mask = (ds.yFRF >= PIER_MIN) & (ds.yFRF <= PIER_MAX)
    bad_mask_2d = bad_mask_2d | pier_mask
    # Exclude entire yFRF transects where >10% of cross-shore pixels are flagged bad
    bad_row_mask = bad_mask_2d.mean(dim='xFRF') > 0.10  # (time, yFRF)
    analysis_mask = (ds.depth > 0) & (qb > 0.0) & ~bad_row_mask
    D_masked = D.where(analysis_mask).fillna(0)
    integrated_D = D_masked.sum(dim='xFRF')

    # Survey-depth dissipation (uses ds.depth instead of booij_depth)
    hmax_survey = GAMMA * ds.depth.clip(min=0)
    D_survey = (ALPHA / 4.0) * qb * fm * RHO * G * hmax_survey**2
    integrated_D_survey = D_survey.where(analysis_mask).fillna(0).sum(dim='xFRF')

    # Incident energy flux — intermediate depth, integrated over all frequency bins
    # Dispersion: omega^2 = g*k*tanh(k*h), solved via Newton's method
    depth_sensor = ds.depth_8m_array

    # Solve dispersion relation for every frequency bin × time step
    print("Solving dispersion relation across all frequency bins and timesteps...")
    omega_vals = (2 * np.pi * ds.waveFrequency).values  # (n_freq,)
    depth_vals = depth_sensor.values                     # (n_time,)
    print(f"  {len(omega_vals)} frequency bins, {len(depth_vals)} timesteps")

    omega_2d = omega_vals[np.newaxis, :]                 # (1, n_freq)
    h_2d     = depth_vals[:, np.newaxis]                 # (n_time, 1)
    k0       = omega_2d**2 / G * np.ones((len(depth_vals), len(omega_vals)))

    k_vals = newton(
        func   = lambda k: omega_2d**2 - G * k * np.tanh(k * h_2d),
        x0     = k0,
        fprime = lambda k: -G * (np.tanh(k * h_2d) + k * h_2d / np.cosh(k * h_2d)**2),
    )
    k   = xr.DataArray(k_vals, dims=['time', 'waveFrequency'],
                       coords={'time': ds.time, 'waveFrequency': ds.waveFrequency})
    kh     = k * depth_sensor
    omega_da = 2 * np.pi * ds.waveFrequency
    c_p    = omega_da / k
    c_g    = 0.5 * c_p * (1 + 2 * kh / np.sinh(2 * kh))
    print(f"  c_g range: {float(c_g.min()):.2f}–{float(c_g.max()):.2f} m/s")

    # Integrate directional energy flux over all frequency and direction bins
    print("Integrating directional energy flux (freq × direction)...")
    E_fd           = ds.directionalWaveEnergyDensity   # (time, waveFrequency, waveDirectionBins)
    theta_bins_rad = np.deg2rad(ds.waveDirectionBins)

    # MET "coming from" convention — wave travels toward theta + 180°
    n_east  = -np.sin(theta_bins_rad)
    n_north = -np.cos(theta_bins_rad)

    F_east  = (RHO * G * E_fd * c_g * n_east ).integrate('waveFrequency').integrate('waveDirectionBins')
    F_north = (RHO * G * E_fd * c_g * n_north).integrate('waveFrequency').integrate('waveDirectionBins')

    # Project onto FRF cross-shore; positive = onshore (incident)
    phi = np.deg2rad(90.0 + FRF_OFFSET)   # 71.8° CW from North
    Fx  = -(F_east * np.sin(phi) + F_north * np.cos(phi))
    print(f"  Mean onshore flux Fx: {float(Fx.mean()):.1f} W/m")

    dx = float(ds.xFRF.diff('xFRF').mean())
    Hs = ds.waveHs  # (time,)
    h_local = ds.depth.clip(min=0.01)

    # avg_D = integrated_D / surfzone_width (from analysis mask)
    print("Computing average dissipation rate and surf zone widths...")
    n_valid = analysis_mask.sum(dim='xFRF').astype(float)
    Xs_valid = n_valid * abs(dx)
    avg_D = (integrated_D / Xs_valid).compute()
    avg_D_survey = (integrated_D_survey / Xs_valid).compute()

    # Incident rate Fx/Xs for two gamma_crit bounds — simple cell count, no rolling
    breaker_index = Hs / h_local  # (time, xFRF, yFRF)
    gamma_min = 0.3
    gamma_max = 1.2
    Xs_lo = (breaker_index > gamma_min).sum(dim='xFRF').astype(float) * abs(dx)
    Xs_hi = (breaker_index > gamma_max).sum(dim='xFRF').astype(float) * abs(dx)
    print(f"  Surf zone width (γ=0.3): {float(Xs_lo.mean()):.0f} m mean")
    print(f"  Surf zone width (γ=1.2): {float(Xs_hi.mean()):.0f} m mean")
    with np.errstate(divide='ignore', invalid='ignore'):
        inc_hi = (Fx / Xs_hi).compute()
        inc_lo = (Fx / Xs_lo).compute()

    # Two bounds on the ratio from the surf-zone-width assumption:
    #   inc_lo uses gamma=0.3 (wide surf zone)   -> smaller incident rate -> larger ratio (upper bound)
    #   inc_hi uses gamma=1.2 (narrow surf zone) -> larger incident rate  -> smaller ratio (lower bound)
    ratio_upper = (avg_D/inc_lo).values.flatten()  # gamma=0.3 -> upper ratio bound
    ratio_lower = (avg_D/inc_hi).values.flatten()  # gamma=1.2 -> lower ratio bound

    valid_upper = np.isfinite(ratio_upper) & (ratio_upper > 0)
    valid_lower = np.isfinite(ratio_lower) & (ratio_lower > 0)
    ratio_upper_clean = ratio_upper[valid_upper]
    ratio_lower_clean = ratio_lower[valid_lower]
    print(f"  Booij upper (γ=0.3) — {valid_upper.sum()} valid, median={np.median(ratio_upper_clean):.2f}, mean={np.mean(ratio_upper_clean):.2f}, skew={skew(ratio_upper_clean):.2f}")
    print(f"  Booij lower (γ=1.2) — {valid_lower.sum()} valid, median={np.median(ratio_lower_clean):.2f}, mean={np.mean(ratio_lower_clean):.2f}, skew={skew(ratio_lower_clean):.2f}")

    ratio_lo_survey = (avg_D_survey/inc_lo).values.flatten()
    ratio_hi_survey = (avg_D_survey/inc_hi).values.flatten()
    min_ratio_survey = np.fmin(ratio_lo_survey, ratio_hi_survey)
    valid_survey = np.isfinite(min_ratio_survey) & (min_ratio_survey > 0)
    ratio_clean_survey = min_ratio_survey[valid_survey]
    print(f"  Survey — {valid_survey.sum()} valid, median={np.median(ratio_clean_survey):.2f}, mean={np.mean(ratio_clean_survey):.2f}, skew={skew(ratio_clean_survey):.2f}")

    BINS = np.arange(0, 6.25, 0.25)
    BINS = np.arange(0, 10, 0.25)

    # Square single-column figure
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))

    ax.hist(ratio_lower_clean,
            bins=BINS, density=True, color=IBM[1], edgecolor='black',
            lw=0.4, alpha=0.5,
            label='Lower bound\n' + r'(narrow surf zone, $\gamma=1.2$)')
    ax.hist(ratio_upper_clean,
            bins=BINS, density=True, color=IBM[3], edgecolor='black',
            lw=0.4, alpha=0.5,
            label='Upper bound\n' + r'(wide surf zone, $\gamma=0.3$)')

    ax.set_xlabel(r'$\alpha = \frac{\mathrm{Average\;Estimated\;Dissipation}}{\mathrm{Incident\;Energy\;Dissipation}}$')
    ax.set_ylabel('Density')
    ax.axvline(1, c='k', ls='--', label='Unity')
    styled_legend(ax)
    ax.set_box_aspect(1)
    fig.tight_layout(pad=0.3)

    savefig(fig, Path(__file__).parent / '06_dissipation_ratio.png')



if __name__ == '__main__':
    main()
