"""Reproduce the held-out bathymetric-error tables used in the manuscript.

The latest survey period is treated as the gamma-calibration survey and is
excluded. All reported error statistics therefore use the first two survey
periods. Signed error is

    inverted total depth - surveyed total depth,

so positive bias means that the inversion is too deep.

By default the tables are printed to stdout. Pass ``--output-dir`` to also
write one CSV per table plus a text report.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import dunex_paths


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"
DEFAULT_GLARE_CSV = (
    ROOT / "data/video_dataset/figures/argus_glare_error_daily.csv"
)

G = 9.81
GAMMA = 0.52
MIN_OBS = 10
X_BIN = 12
Y_BIN = 24
PERCENT_ERROR_MIN_DEPTH = 1.0
DEPTH_BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0),
              (3.0, 4.0), (4.0, 5.0))


def rmse(values):
    values = np.asarray(values)
    return float(np.sqrt(np.mean(values**2)))


def bias(values):
    values = np.asarray(values)
    return float(np.mean(values))


def ols(y, predictors, names, survey_indicator=None):
    """OLS statistics with optional two-period fixed intercept and LODO PRESS."""
    y = np.asarray(y, dtype=float)
    columns = [np.ones(len(y))]
    column_names = ["intercept"]
    for predictor, name in zip(predictors, names):
        columns.append(np.asarray(predictor, dtype=float))
        column_names.append(name)
    if survey_indicator is not None:
        columns.append(np.asarray(survey_indicator, dtype=float))
        column_names.append("survey_period")

    X = np.column_stack(columns)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ beta
    dof = len(y) - X.shape[1]
    sse = float(residual @ residual)
    sst = float(np.sum((y - y.mean()) ** 2))
    covariance = (sse / dof) * np.linalg.pinv(X.T @ X)
    standard_error = np.sqrt(np.diag(covariance))
    p_value = 2 * stats.t.sf(np.abs(beta / standard_error), dof)
    critical = stats.t.ppf(0.975, dof)

    press = 0.0
    for index in range(len(y)):
        train = np.arange(len(y)) != index
        beta_i = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
        press += float((y[index] - X[index] @ beta_i) ** 2)

    coefficients = pd.DataFrame({
        "term": column_names,
        "coefficient": beta,
        "standard_error": standard_error,
        "ci95_low": beta - critical * standard_error,
        "ci95_high": beta + critical * standard_error,
        "p_value": p_value,
    })
    model = {
        "n_days": len(y),
        "r2": 1.0 - sse / sst,
        "press_r2": 1.0 - press / sst,
    }
    return coefficients, model


def select_held_out(ds, calibration_survey_date=None):
    normalized = pd.to_datetime(ds.survey_time.values).normalize()
    survey_dates = list(sorted(pd.unique(normalized)))
    if len(survey_dates) < 2:
        raise ValueError("at least two survey periods are required")

    if calibration_survey_date is None:
        calibration = pd.Timestamp(survey_dates[-1]).normalize()
    else:
        calibration = pd.Timestamp(calibration_survey_date).normalize()
        if calibration.to_datetime64() not in survey_dates:
            choices = ", ".join(str(pd.Timestamp(x).date()) for x in survey_dates)
            raise ValueError(
                f"calibration survey {calibration.date()} not found; choose {choices}"
            )

    keep = normalized != calibration
    held_out = ds.isel(time=keep)
    held_dates = list(sorted(pd.unique(normalized[keep])))
    return held_out, held_dates, calibration


def prepare_spatial(ds):
    inverted = ds.mean_squared_speed / (G * (1.0 + GAMMA / 2.0))
    total_depth = ds.depth + ds.setup
    variables = xr.Dataset({
        "inverted_depth": inverted,
        "still_depth": ds.depth,
        "setup": ds.setup,
        "survey_depth": total_depth,
    })
    # Use one native-pixel support for every field in a coarse cell. Without
    # this paired mask, xarray's skipna mean can average inverted and surveyed
    # depths from different native pixels when the survey has missing values.
    paired_valid = (
        (ds.observation_count >= MIN_OBS) &
        np.isfinite(inverted) & np.isfinite(total_depth) &
        np.isfinite(ds.depth) & np.isfinite(ds.setup)
    )
    masked = variables.where(paired_valid)
    coarse = masked.coarsen(
        xFRF=X_BIN, yFRF=Y_BIN, boundary="trim"
    ).mean()
    return inverted, total_depth, coarse


def population_table(ds, held_dates, calibration):
    video_days = pd.to_datetime(ds.time.values).normalize()
    return pd.DataFrame([{
        "gamma": GAMMA,
        "calibration_survey_excluded": str(calibration.date()),
        "held_out_surveys": ", ".join(str(pd.Timestamp(x).date()) for x in held_dates),
        "videos": ds.sizes["time"],
        "days": len(pd.unique(video_days)),
        "Hs_min_m": float(ds.waveHs.min()),
        "Hs_max_m": float(ds.waveHs.max()),
        "native_min_observations": MIN_OBS,
        "cross_shore_bin_m": X_BIN,
        "alongshore_bin_m": Y_BIN,
    }])


def headline_table(coarse):
    inverted = coarse.inverted_depth.values.ravel()
    rows = []
    for label, surveyed in (
        ("with_setup", coarse.survey_depth.values.ravel()),
        ("no_setup", coarse.still_depth.values.ravel()),
    ):
        error = inverted - surveyed
        valid = np.isfinite(error) & np.isfinite(surveyed)
        regions = {
            "all_finite": valid,
            "positive_depth": valid & (surveyed > 0),
            "nonpositive_depth": valid & (surveyed <= 0),
        }
        for region, mask in regions.items():
            absolute = np.abs(error[mask])
            rows.append({
                "reference": label,
                "region": region,
                "n_video_cells": int(mask.sum()),
                "bias_m": bias(error[mask]),
                "mae_m": float(np.mean(absolute)),
                "rmse_m": rmse(error[mask]),
                "p95_abs_error_m": float(np.percentile(absolute, 95)),
            })
    return pd.DataFrame(rows)


def video_domain_variability_table(coarse):
    """Summarize between-video variability for the reported depth domains."""
    surveyed = coarse.survey_depth.values
    error = (coarse.inverted_depth - coarse.survey_depth).values
    valid = np.isfinite(error) & np.isfinite(surveyed)
    regions = {
        "all_finite": valid,
        "positive_depth": valid & (surveyed > 0),
        "nonpositive_depth": valid & (surveyed <= 0),
    }
    rows = []
    for region, mask in regions.items():
        video_rmse = []
        video_bias = []
        for index in range(error.shape[0]):
            values = error[index][mask[index]]
            if len(values) == 0:
                continue
            video_rmse.append(rmse(values))
            video_bias.append(bias(values))
        rows.append({
            "region": region,
            "n_videos": len(video_rmse),
            "mean_video_rmse_m": float(np.mean(video_rmse)),
            "std_video_rmse_m": float(np.std(video_rmse, ddof=1)),
            "mean_video_bias_m": float(np.mean(video_bias)),
            "std_video_bias_m": float(np.std(video_bias, ddof=1)),
        })
    return pd.DataFrame(rows)


def shallow_deep_metric_table(coarse):
    inverted = coarse.inverted_depth.values.ravel()
    rows = []
    for label, surveyed in (
        ("with_setup", coarse.survey_depth.values.ravel()),
        ("no_setup", coarse.still_depth.values.ravel()),
    ):
        error = inverted - surveyed
        valid = np.isfinite(error) & np.isfinite(surveyed)
        regions = {
            "0<depth<=1m": valid & (surveyed > 0) &
                            (surveyed <= PERCENT_ERROR_MIN_DEPTH),
            "depth>1m": valid & (surveyed > PERCENT_ERROR_MIN_DEPTH),
        }
        for region, mask in regions.items():
            row = {
                "reference": label,
                "region": region,
                "n_video_cells": int(mask.sum()),
            }
            if region == "depth>1m":
                percent_error = 100.0 * error[mask] / surveyed[mask]
                row.update({
                    "bias_m": np.nan,
                    "mae_m": np.nan,
                    "rmse_m": np.nan,
                    "mean_percent_error": bias(percent_error),
                    "rms_percent_error": rmse(percent_error),
                })
            else:
                row.update({
                    "bias_m": bias(error[mask]),
                    "mae_m": float(np.mean(np.abs(error[mask]))),
                    "rmse_m": rmse(error[mask]),
                    "mean_percent_error": np.nan,
                    "rms_percent_error": np.nan,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def depth_and_setup_table(coarse):
    inverted = coarse.inverted_depth.values.ravel()
    total_depth = coarse.survey_depth.values.ravel()
    still_depth = coarse.still_depth.values.ravel()
    setup = coarse.setup.values.ravel()
    with_setup_error = inverted - total_depth
    no_setup_error = inverted - still_depth
    time_shape = (coarse.sizes["time"], -1)
    rows = []

    for low, high in DEPTH_BINS:
        mask = (
            np.isfinite(with_setup_error) & np.isfinite(no_setup_error) &
            np.isfinite(total_depth) & (total_depth > low) & (total_depth <= high)
        )
        per_time = mask.reshape(time_shape)
        mean_depth = float(np.mean(total_depth[mask]))
        rows.append({
            "depth_bin_m": f"{low:g}-{high:g}",
            "n_video_cells": int(mask.sum()),
            "n_videos": int(np.count_nonzero(np.any(per_time, axis=1))),
            "mean_depth_m": mean_depth,
            "mean_setup_m": float(np.mean(setup[mask])),
            "bias_no_setup_m": bias(no_setup_error[mask]),
            "bias_with_setup_m": bias(with_setup_error[mask]),
            "mae_with_setup_m": float(np.mean(np.abs(with_setup_error[mask]))),
            "rmse_no_setup_m": rmse(no_setup_error[mask]),
            "rmse_with_setup_m": rmse(with_setup_error[mask]),
            "p95_abs_error_with_setup_m": float(
                np.percentile(np.abs(with_setup_error[mask]), 95)
            ),
            "rmse_over_mean_depth_percent": (
                np.nan if low == 0 else 100.0 * rmse(with_setup_error[mask]) / mean_depth
            ),
        })
    return pd.DataFrame(rows)


def fixed_footprint(ds, inverted, total_depth):
    error = inverted - total_depth
    valid = (
        (ds.observation_count >= MIN_OBS) &
        np.isfinite(error) & np.isfinite(total_depth)
    )
    common_native = valid.all("time")
    error_coarse = error.where(common_native).coarsen(
        xFRF=X_BIN, yFRF=Y_BIN, boundary="trim"
    ).mean()
    depth_coarse = total_depth.where(common_native).coarsen(
        xFRF=X_BIN, yFRF=Y_BIN, boundary="trim"
    ).mean()
    wet_bins = (depth_coarse > 0).all("time")

    video = pd.DataFrame({
        "day": pd.to_datetime(ds.time.values).normalize(),
        "survey": pd.to_datetime(ds.survey_time.values).normalize(),
        "rmse": np.sqrt((error_coarse**2).mean(("yFRF", "xFRF"))).values,
        "bias": error_coarse.mean(("yFRF", "xFRF")).values,
        "mean_depth": depth_coarse.mean(("yFRF", "xFRF")).values,
        "relative_mean_depth": depth_coarse.where(wet_bins)
        .mean(("yFRF", "xFRF")).values,
        "relative_rmse": np.sqrt(
            ((error_coarse / depth_coarse).where(wet_bins) ** 2)
            .mean(("yFRF", "xFRF"))
        ).values,
        "Hs": ds.waveHs.values,
        "water_level": ds.water_level.values,
        "Tp": ds.waveTp.values,
        "Tm": ds.waveTm.values,
        "wave_direction": ds.waveMeanDirection.values,
    })
    daily = video.groupby("day", as_index=False).agg({
        "survey": "first",
        "rmse": "mean",
        "bias": "mean",
        "mean_depth": "mean",
        "relative_mean_depth": "mean",
        "relative_rmse": "mean",
        "Hs": "mean",
        "water_level": "mean",
        "Tp": "mean",
        "Tm": "mean",
        "wave_direction": "mean",
    })
    metadata = pd.DataFrame([{
        "common_native_pixels": int(common_native.sum()),
        "common_coarse_bins": int(np.isfinite(error_coarse.isel(time=0)).sum()),
        "always_wet_coarse_bins": int(wet_bins.sum()),
        "daily_conditions": len(daily),
    }])
    return daily, metadata


def simple_model_row(daily, response, predictor, survey_adjusted=False):
    survey_indicator = None
    if survey_adjusted:
        survey_indicator = (daily.survey == daily.survey.iloc[-1]).astype(float)
    coefficients, model = ols(
        daily[response].values,
        [daily[predictor].values],
        [predictor],
        survey_indicator=survey_indicator,
    )
    coefficient = coefficients.loc[coefficients.term == predictor].iloc[0]
    pearson_r = stats.pearsonr(daily[predictor], daily[response]).statistic
    return {
        "response": response,
        "predictor": predictor,
        "survey_adjusted": survey_adjusted,
        "n_days": model["n_days"],
        "pearson_r": pearson_r,
        "slope": coefficient.coefficient,
        "ci95_low": coefficient.ci95_low,
        "ci95_high": coefficient.ci95_high,
        "p_value": coefficient.p_value,
        "model_r2": model["r2"],
        "press_r2": model["press_r2"],
    }


def temporal_model_tables(daily):
    rows = []
    for predictor in ("Hs", "mean_depth", "water_level", "Tp", "Tm",
                      "wave_direction"):
        rows.append(simple_model_row(daily, "rmse", predictor, False))
    for predictor in ("Hs", "relative_mean_depth", "water_level"):
        rows.append(simple_model_row(daily, "relative_rmse", predictor, False))
        rows.append(simple_model_row(daily, "relative_rmse", predictor, True))
    rows.append(simple_model_row(daily, "bias", "Hs", False))
    rows.append(simple_model_row(daily, "bias", "Hs", True))
    simple = pd.DataFrame(rows)

    survey_indicator = (daily.survey == daily.survey.iloc[-1]).astype(float)
    multiple_rows = []
    for model_name, predictors, names in (
        ("depth_plus_survey", [daily.mean_depth], ["mean_depth"]),
        ("Hs_depth_plus_survey", [daily.Hs, daily.mean_depth], ["Hs", "mean_depth"]),
    ):
        coefficients, model = ols(
            daily.rmse, predictors, names, survey_indicator=survey_indicator
        )
        for row in coefficients.to_dict("records"):
            row.update({
                "model": model_name,
                "n_days": model["n_days"],
                "model_r2": model["r2"],
                "press_r2": model["press_r2"],
            })
            multiple_rows.append(row)
    multiple = pd.DataFrame(multiple_rows)
    return simple, multiple


def within_video_table(ds, inverted, total_depth):
    error = inverted - total_depth
    coarse = xr.Dataset({"error": error, "depth": total_depth}).where(
        ds.observation_count >= MIN_OBS
    ).coarsen(xFRF=X_BIN, yFRF=Y_BIN, boundary="trim").mean()
    x_grid = np.broadcast_to(
        coarse.xFRF.values[None, :], coarse.error.isel(time=0).shape
    ).ravel()
    values = {
        "abs_error_vs_depth": [],
        "fractional_abs_error_vs_depth": [],
        "abs_error_vs_cross_shore": [],
        "signed_error_vs_depth": [],
    }
    for index in range(coarse.sizes["time"]):
        error_i = coarse.error.isel(time=index).values.ravel()
        depth_i = coarse.depth.isel(time=index).values.ravel()
        mask = np.isfinite(error_i) & np.isfinite(depth_i) & (depth_i > 0)
        values["abs_error_vs_depth"].append(
            stats.spearmanr(depth_i[mask], np.abs(error_i[mask])).statistic
        )
        values["fractional_abs_error_vs_depth"].append(
            stats.spearmanr(
                depth_i[mask], np.abs(error_i[mask]) / depth_i[mask]
            ).statistic
        )
        values["abs_error_vs_cross_shore"].append(
            stats.spearmanr(x_grid[mask], np.abs(error_i[mask])).statistic
        )
        values["signed_error_vs_depth"].append(
            stats.spearmanr(depth_i[mask], error_i[mask]).statistic
        )

    rows = []
    for relationship, correlations in values.items():
        correlations = np.asarray(correlations)
        rows.append({
            "relationship": relationship,
            "n_videos": len(correlations),
            "median_spearman_rho": float(np.nanmedian(correlations)),
            "mean_spearman_rho": float(np.nanmean(correlations)),
            "fraction_positive": float(np.mean(correlations > 0)),
            "one_sample_ttest_p": float(
                stats.ttest_1samp(correlations, 0, nan_policy="omit").pvalue
            ),
        })
    return pd.DataFrame(rows)


def geometry_tables(ds, inverted, total_depth):
    """Camera-transition bins and within-video geometric correlations."""
    # argus_error also contains plotting helpers and therefore imports
    # matplotlib. Give those imports writable caches even though this script
    # only uses its geometry functions.
    cache_root = Path(tempfile.gettempdir()) / "breaking_wave_error_table_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        module_path = ROOT / "data/argus/argus_error.py"
        module_name = "_breaking_wave_argus_error"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load geometry helpers from {module_path}")
        argus_error = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = argus_error
        try:
            spec.loader.exec_module(argus_error)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        station = argus_error.load_camera_yaml(argus_error.CAMERA_YAML, "Duck")
        X, Y = np.meshgrid(ds.xFRF.values, ds.yFRF.values)
        transition_distance = argus_error.nominal_camera_transition_distance(
            station, X, Y
        )[0]
        resolution_x, resolution_y = argus_error.station_resolution(
            station, X, Y, method="exact"
        )
        footprint = np.sqrt(resolution_x * resolution_y)

        camera_scores = []
        for camera in station.cameras:
            bearing = np.arctan2(X - camera.x, Y - camera.y)
            off_axis = np.abs(np.angle(
                np.exp(1j * (bearing - camera.azimuth))
            )) / (camera.fov / 2.0)
            camera_scores.append(
                np.where(argus_error.in_view(camera, X, Y), off_axis, np.nan)
            )
        off_axis = np.nanmin(np.stack(camera_scores), axis=0)

    coordinates = {"yFRF": ds.yFRF, "xFRF": ds.xFRF}

    def coarsen_geometry(values):
        return xr.DataArray(
            values, dims=("yFRF", "xFRF"), coords=coordinates
        ).coarsen(xFRF=X_BIN, yFRF=Y_BIN, boundary="trim").mean().values

    geometry = {
        "camera_transition_distance": coarsen_geometry(transition_distance),
        "nominal_pixel_footprint": coarsen_geometry(footprint),
        "normalized_off_axis_angle": coarsen_geometry(off_axis),
    }

    error = inverted - total_depth
    coarse = xr.Dataset({"error": error, "depth": total_depth}).where(
        ds.observation_count >= MIN_OBS
    ).coarsen(xFRF=X_BIN, yFRF=Y_BIN, boundary="trim").mean()

    transition_rows = []
    distance = geometry["camera_transition_distance"]
    for low, high in ((0, 15), (15, 30), (30, 60), (60, 120)):
        mask = (
            np.isfinite(coarse.error.values) & (coarse.depth.values > 0) &
            np.isfinite(distance)[None, :, :] &
            (distance[None, :, :] >= low) & (distance[None, :, :] < high)
        )
        errors = coarse.error.values[mask]
        transition_rows.append({
            "distance_bin_m": f"{low}-{high}",
            "n_video_cells": int(mask.sum()),
            "bias_m": bias(errors),
            "rmse_m": rmse(errors),
        })

    correlation_rows = []
    for name, field in geometry.items():
        correlations = []
        for index in range(coarse.sizes["time"]):
            error_i = coarse.error.isel(time=index).values.ravel()
            depth_i = coarse.depth.isel(time=index).values.ravel()
            field_i = field.ravel()
            mask = (
                np.isfinite(error_i) & np.isfinite(depth_i) & (depth_i > 0) &
                np.isfinite(field_i)
            )
            correlations.append(
                stats.spearmanr(field_i[mask], np.abs(error_i[mask])).statistic
            )
        correlations = np.asarray(correlations)
        correlation_rows.append({
            "geometry_metric": name,
            "n_videos": len(correlations),
            "median_spearman_rho": float(np.nanmedian(correlations)),
            "one_sample_ttest_p": float(
                stats.ttest_1samp(correlations, 0, nan_policy="omit").pvalue
            ),
        })
    return pd.DataFrame(transition_rows), pd.DataFrame(correlation_rows)


def glare_table(daily, glare_csv):
    if not glare_csv.exists():
        return pd.DataFrame([{
            "response": "glare table unavailable",
            "predictor": str(glare_csv),
            "n_days": 0,
            "pearson_r": np.nan,
            "p_value": np.nan,
        }])

    glare = pd.read_csv(glare_csv, parse_dates=["day"])
    merged = daily.merge(
        glare[["day", "glare", "alignment_mean", "alignment_max"]],
        on="day", how="inner", validate="one_to_one"
    )
    rows = []
    for response in ("rmse", "bias"):
        for predictor in ("glare", "alignment_mean", "alignment_max"):
            result = stats.pearsonr(merged[predictor], merged[response])
            rows.append({
                "response": response,
                "predictor": predictor,
                "n_days": len(merged),
                "pearson_r": result.statistic,
                "p_value": result.pvalue,
            })
    return pd.DataFrame(rows)


def format_value(value):
    if pd.isna(value):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def print_table(title, frame, stream=sys.stdout):
    print(f"\n## {title}", file=stream)
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(format_value)
    print(display.to_string(index=False), file=stream)


def build_tables(dataset, calibration_survey_date=None,
                 glare_csv=DEFAULT_GLARE_CSV, include_geometry=True):
    ds_all = xr.open_dataset(dataset)
    try:
        ds, held_dates, calibration = select_held_out(
            ds_all, calibration_survey_date
        )
        inverted, total_depth, coarse = prepare_spatial(ds)
        daily, footprint_metadata = fixed_footprint(ds, inverted, total_depth)
        temporal_simple, temporal_multiple = temporal_model_tables(daily)

        tables = {
            "population": population_table(ds, held_dates, calibration),
            "headline_error": headline_table(coarse),
            "video_domain_variability": video_domain_variability_table(coarse),
            "shallow_and_percentage_error": shallow_deep_metric_table(coarse),
            "depth_bins_and_setup": depth_and_setup_table(coarse),
            "fixed_footprint": footprint_metadata,
            "daily_fixed_footprint_values": daily,
            "temporal_simple_models": temporal_simple,
            "temporal_multiple_models": temporal_multiple,
            "within_video_spatial_correlations": within_video_table(
                ds, inverted, total_depth
            ),
            "glare_and_sun_alignment": glare_table(daily, glare_csv),
        }
        if include_geometry:
            transitions, correlations = geometry_tables(ds, inverted, total_depth)
            tables["camera_transition_bins"] = transitions
            tables["camera_geometry_correlations"] = correlations
        return tables
    finally:
        ds_all.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--calibration-survey-date",
        help="Survey date excluded as gamma calibration (default: latest survey)",
    )
    parser.add_argument("--glare-csv", type=Path, default=DEFAULT_GLARE_CSV)
    parser.add_argument(
        "--skip-geometry", action="store_true",
        help="Skip camera-transition, footprint, and off-axis tables",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Also write CSV tables and error_trend_tables.txt here",
    )
    args = parser.parse_args()

    tables = build_tables(
        args.dataset,
        calibration_survey_date=args.calibration_survey_date,
        glare_csv=args.glare_csv,
        include_geometry=not args.skip_geometry,
    )

    report = io.StringIO()
    print(f"Dataset: {args.dataset}", file=report)
    for name, frame in tables.items():
        print_table(name.replace("_", " ").title(), frame, stream=report)
    rendered = report.getvalue()
    print(rendered, end="")

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        (args.output_dir / "error_trend_tables.txt").write_text(rendered)


if __name__ == "__main__":
    main()
