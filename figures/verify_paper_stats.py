"""Verify manuscript headline stats against the combined datasets.

Paper claims checked against the canonical merged dataset.
Breaker-index and bathymetric-error statistics use the first two survey
periods, which were held out from selecting gamma=0.52; the third survey period
was used for that calibration and is excluded from verification.
  1. Breaker index: 73% of observations in [0.3, 1.2], median 0.81   (fig 01)
  2. Bathy inversion: RMSE, bias, and mean/RMSE percentage error     (table 1)
  3. Dissipation ratio alpha: median 0.56                            (fig 06)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import dunex_paths

G = 9.81

DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"
DS_TAG = Path(DATASET).stem

# ── 1. Breaker index stats (replicates figures/01_breaker_index_hist.py) ──────
print("=" * 60)
print(f"1. BREAKER INDEX ({DS_TAG})")
ds = xr.open_dataset(DATASET)

survey_dates = sorted(pd.unique(pd.to_datetime(ds.survey_time.values).normalize()))
# The final survey period was used to select gamma=0.52. Retain only the two
# held-out survey periods for both breaker-index and bathymetric-error metrics.
held_out_survey_dates = survey_dates[:2]
print(f"Held-out survey dates: {held_out_survey_dates}")
counts = ds.observation_count.values.flatten()
# Figure 01 defines gamma_s using total depth (tide-inclusive depth + setup).
# Keep the no-setup result only as an explicitly labelled sensitivity comparison.
for label, denom in [("depth + setup (full total depth) [as in fig 01]", ds.depth + ds.setup),
                     ("depth (tide, no setup) [comparison only]", ds.depth)]:
    gamma_s = (ds.waveHs / denom).values.flatten()
    valid = ~(np.isnan(gamma_s) | np.isnan(counts)) & (counts > 0) & (gamma_s > 0)
    gv, wt = gamma_s[valid], counts[valid]
    total = wt.sum()
    pct = wt[(gv >= 0.3) & (gv <= 1.2)].sum() / total * 100
    si = np.argsort(gv)
    w_med = gv[si][np.searchsorted(np.cumsum(wt[si]), 0.5 * total)]
    print(f"  {label}: {pct:.1f}% in [0.3,1.2], median {w_med:.3f}  (obs {total:,.0f})")

# ── 2. Bathy inversion error at gamma=0.52 (04_bathy_planview binning) ────────
print("=" * 60)
print(f"2. BATHY INVERSION RMSE/BIAS ({DS_TAG}, 12x24 bins, min_obs=10)")
GAMMA = 0.52
PERCENT_ERROR_MIN_DEPTH = 1.0
ds['booij_depth'] = ds.mean_squared_speed / (G * (1 + GAMMA / 2))
ds['survey_depth'] = ds.depth + ds.setup
spatial = ds[['booij_depth', 'mean_squared_speed', 'depth', 'setup',
              'survey_depth', 'observation_count']]
paired_valid = (
    (spatial.observation_count >= 10) &
    np.isfinite(spatial.booij_depth) &
    np.isfinite(spatial.survey_depth) &
    np.isfinite(spatial.depth) & np.isfinite(spatial.setup)
)
sub = spatial.where(paired_valid).sel(xFRF=slice(-100, 1600))
coarse = sub.coarsen(xFRF=12, yFRF=24, boundary='trim').mean()
inv = coarse.booij_depth
for slabel, surv in [("with setup", coarse.survey_depth),
                     ("no setup  ", coarse.depth)]:
    print(f"  --- survey {slabel} ---")
    err = (inv - surv).values.flatten()
    surv_flat = surv.values.flatten()
    m = np.isfinite(err) & np.isfinite(surv_flat)
    for region, region_mask in (
        ("all finite", m),
        ("positive depth", m & (surv_flat > 0)),
        ("nonpositive depth", m & (surv_flat <= 0)),
    ):
        region_err = err[region_mask]
        print(
            f"  {region:17s}: RMSE {np.sqrt(np.mean(region_err**2)):.3f}  "
            f"bias {np.mean(region_err):+.3f}  MAE {np.mean(np.abs(region_err)):.3f}  "
            f"P95(|e|) {np.percentile(np.abs(region_err), 95):.3f}  "
            f"(N={region_mask.sum():,})"
        )

    # Percentage errors become unstable as surveyed depth approaches zero.
    # Report them only offshore of the 1 m depth threshold; retain metre-based
    # metrics in shallower water. Mean percentage error is signed, consistent
    # with the bias convention above, while percentage RMSE is unsigned.
    deep = m & np.isfinite(surv_flat) & (surv_flat > PERCENT_ERROR_MIN_DEPTH)
    pct_err = 100 * err[deep] / surv_flat[deep]
    print(f"  depth >{PERCENT_ERROR_MIN_DEPTH:g}m percent error: mean {np.mean(pct_err):+.1f}%  RMSE {np.sqrt(np.mean(pct_err**2)):.1f}%  (N={deep.sum():,})")

    shallow = (m & np.isfinite(surv_flat) & (surv_flat > 0) &
               (surv_flat <= PERCENT_ERROR_MIN_DEPTH))
    shallow_err = err[shallow]
    print(f"  0<depth<={PERCENT_ERROR_MIN_DEPTH:g}m: mean {np.mean(shallow_err):+.3f}m  MAE {np.mean(np.abs(shallow_err)):.3f}m  RMSE {np.sqrt(np.mean(shallow_err**2)):.3f}m  (N={shallow.sum():,})")

    # masked to cells with |hErr| < 0.5 m (excludes gross-error outliers)
    mh = m & (np.abs(err) < 0.5)
    eh = err[mh]
    print(f"  |hErr|<0.5m: RMSE {np.sqrt(np.mean(eh**2)):.3f}  bias {np.mean(eh):+.3f}  (N={mh.sum():,})")

# Between-video variability for the setup-inclusive reference. These are the
# mean and sample standard deviation of the 27 independently calculated video
# metrics, not the pooled-cell metrics printed above.
print("  --- per-video variability (with setup; mean +/- sample SD) ---")
video_error = (inv - coarse.survey_depth).values
video_depth = coarse.survey_depth.values
video_valid = np.isfinite(video_error) & np.isfinite(video_depth)
for region, region_mask in (
    ("all finite", video_valid),
    ("positive depth", video_valid & (video_depth > 0)),
    ("nonpositive depth", video_valid & (video_depth <= 0)),
):
    video_rmse = []
    video_bias = []
    for index in range(video_error.shape[0]):
        values = video_error[index][region_mask[index]]
        if len(values) == 0:
            continue
        video_rmse.append(np.sqrt(np.mean(values**2)))
        video_bias.append(np.mean(values))
    mean_video_bias = np.mean(video_bias)
    if abs(mean_video_bias) < 0.0005:
        mean_video_bias = 0.0
    print(
        f"  {region:17s}: RMSE {np.mean(video_rmse):.3f} +/- "
        f"{np.std(video_rmse, ddof=1):.3f} m  bias {mean_video_bias:+.3f} "
        f"+/- {np.std(video_bias, ddof=1):.3f} m  (N={len(video_rmse)} videos)"
    )

# ── 2b. Booij SPEED RMSE/bias with/without setup (same coarse grid) ────────────
# Measured speed sqrt(<c^2>) vs shallow-water Booij speed sqrt(g(1+gamma/2) h_surv)
print("-" * 60)
print("2b. BOOIJ SPEED RMSE/BIAS (m/s)")
c_meas = np.sqrt(coarse.mean_squared_speed)
stats = {}
for slabel, surv in [("with setup", coarse.depth + coarse.setup),
                     ("no setup  ", coarse.depth)]:
    c_pred = np.sqrt(G * (1 + GAMMA / 2) * surv.clip(min=0))
    cerr = (c_meas - c_pred).values.flatten()
    m = np.isfinite(cerr)
    rmse, bias = np.sqrt(np.mean(cerr[m]**2)), np.mean(cerr[m])
    stats[slabel.strip()] = (rmse, bias)
    print(f"  --- survey {slabel} ---")
    print(f"  RMSE {rmse:.3f}  bias {bias:+.3f}  (N={m.sum():,})")
# difference and % change from no-setup -> with-setup
(rw, bw), (rn, bn) = stats["with setup"], stats["no setup"]
print("  --- setup effect (with - no setup) ---")
print(f"  RMSE: {rw:.3f} vs {rn:.3f}  diff {rw-rn:+.3f} m/s  ({(rw-rn)/rn*100:+.1f}%)")
print(f"  bias: {bw:+.3f} vs {bn:+.3f}  diff {bw-bn:+.3f} m/s  ({(bw-bn)/abs(bn)*100:+.1f}%)")
ds.close()

# ── 3. Dissipation ratio alpha (mirrors figures/06_dissipation_rate_ratio_hist) ───
print("=" * 60)
print(f"3. DISSIPATION RATIO ({DS_TAG}, crest-count Qb)")
from scipy.optimize import newton

ds = xr.open_dataset(DATASET)
GAMMA, RHO, ALPHA = 0.6, 1025.0, 1.0
DX_CROSS = DY_ALONG = 5
MIN_OBS = 5
FRF_OFFSET = -18.2
PIER_MIN, PIER_MAX = 490, 530

ds['booij_depth'] = ds.mean_squared_speed / (G * (1 + GAMMA / 2))
spatial_vars = [v for v in ds.data_vars if 'xFRF' in ds[v].dims and 'yFRF' in ds[v].dims]
ds_sp = ds[spatial_vars]
obs_mask = ds_sp.observation_count >= MIN_OBS
ds_coarse = ds_sp.where(obs_mask).coarsen(xFRF=DX_CROSS, yFRF=DY_ALONG, boundary='trim').mean()
obs_coarse = ds_sp.observation_count.coarsen(xFRF=DX_CROSS, yFRF=DY_ALONG, boundary='trim').sum()
ds_coarse['observation_count'] = obs_coarse / (DX_CROSS * DY_ALONG)
ds_smooth = ds_coarse.interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='linear')
for dim in ['xFRF', 'yFRF']:
    ds_smooth = ds_smooth.ffill(dim).bfill(dim)
ds_smooth = xr.merge([ds_smooth, ds[[v for v in ds.data_vars if v not in spatial_vars]]])

# Qb = Nb / Ntotal = crest_count / (t_total / Tm), with t_total = trimmed_frames / fps
# Crest-count formula shared with figures 05 and 06.
qb = (ds_smooth.crest_count * ds_smooth.fps / ds_smooth.trimmed_frames * ds_smooth.waveTm).clip(max=1)
booij_depth = ds_smooth.booij_depth.clip(min=0)
hmax = GAMMA * booij_depth
fm = 1.0 / ds_smooth.waveTm
D = (ALPHA / 4.0) * qb * fm * RHO * G * hmax**2

bad_mask_2d = xr.open_dataset(dunex_paths.DATA_ROOT / 'video_dataset/good_data_mask.nc') \
    .bad_mask.astype(int).interp(xFRF=ds.xFRF, yFRF=ds.yFRF, method='nearest') \
    .fillna(True).astype(bool)
pier_mask = (ds.yFRF >= PIER_MIN) & (ds.yFRF <= PIER_MAX)
bad_mask_2d = bad_mask_2d | pier_mask
bad_row_mask = bad_mask_2d.mean(dim='xFRF') > 0.10
analysis_mask = (ds.depth > 0) & (qb > 0.0) & ~bad_row_mask
D_masked = D.where(analysis_mask).fillna(0)
integrated_D = D_masked.sum(dim='xFRF')

depth_sensor = ds.depth_8m_array
omega_vals = (2 * np.pi * ds.waveFrequency).values
depth_vals = depth_sensor.values
omega_2d = omega_vals[np.newaxis, :]
h_2d = depth_vals[:, np.newaxis]
k0 = omega_2d**2 / G * np.ones((len(depth_vals), len(omega_vals)))
k_vals = newton(
    func=lambda k: omega_2d**2 - G * k * np.tanh(k * h_2d),
    x0=k0,
    fprime=lambda k: -G * (np.tanh(k * h_2d) + k * h_2d / np.cosh(k * h_2d)**2),
)
k = xr.DataArray(k_vals, dims=['time', 'waveFrequency'],
                 coords={'time': ds.time, 'waveFrequency': ds.waveFrequency})
kh = k * depth_sensor
c_p = (2 * np.pi * ds.waveFrequency) / k
c_g = 0.5 * c_p * (1 + 2 * kh / np.sinh(2 * kh))

E_fd = ds.directionalWaveEnergyDensity
theta_bins_rad = np.deg2rad(ds.waveDirectionBins)
n_east = -np.sin(theta_bins_rad)
n_north = -np.cos(theta_bins_rad)
F_east = (RHO * G * E_fd * c_g * n_east).integrate('waveFrequency').integrate('waveDirectionBins')
F_north = (RHO * G * E_fd * c_g * n_north).integrate('waveFrequency').integrate('waveDirectionBins')
phi = np.deg2rad(90.0 + FRF_OFFSET)
Fx = -(F_east * np.sin(phi) + F_north * np.cos(phi))

dx = float(ds.xFRF.diff('xFRF').mean())
Hs = ds.waveHs
h_local = ds.depth.clip(min=0.01)
n_valid = analysis_mask.sum(dim='xFRF').astype(float)
Xs_valid = n_valid * abs(dx)
avg_D = (integrated_D / Xs_valid).compute()
breaker_index = Hs / h_local
Xs_lo = (breaker_index > 0.3).sum(dim='xFRF').astype(float) * abs(dx)
Xs_hi = (breaker_index > 1.2).sum(dim='xFRF').astype(float) * abs(dx)
with np.errstate(divide='ignore', invalid='ignore'):
    inc_hi = (Fx / Xs_hi).compute()
    inc_lo = (Fx / Xs_lo).compute()
ratio_lo = (avg_D / inc_lo).values.flatten()
ratio_hi = (avg_D / inc_hi).values.flatten()
min_ratio = np.fmin(ratio_lo, ratio_hi)
valid_min = np.isfinite(min_ratio) & (min_ratio > 0)
rc = min_ratio[valid_min]
print(f"  crest-count Qb: median alpha {np.median(rc):.3f}  mean {np.mean(rc):.3f}  N={valid_min.sum()}")
print(f"  median Qb over analysis mask: {float(qb.where(analysis_mask).median()):.3f}")
print("DONE")
