#!/usr/bin/env python3
"""Remove spatially isolated detections from wave-speed NetCDFs and maps.

The speed pipeline maps every tracked crest back onto its detected pixels.  A
false track therefore appears as a small, disconnected island in the spatial
support, often with only one observation per pixel.  This post-processor keeps
only 8-connected regions that are large enough to represent a coherent camera
footprint and masks the corresponding values in every speed/support variable.

By default, cleaned products retain the pipeline's normal filenames and are
written under a ``postprocessed/`` subdirectory.  This prevents directory
globs from ingesting both raw and cleaned NetCDFs.  Use ``--in-place`` to
replace the original NetCDF and PNG; recoverable ``*.pre_postprocess`` backups
are made first.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.ndimage import label


SPEED_VARIABLES = ("mean_squared_speed", "mean_speed", "median_speed")
SUPPORT_VARIABLES = ("observation_count", "crest_count")


def resolve_netcdf(path: Path) -> Path:
    """Accept either a speed NetCDF or its ``*_pixel_speed_map.png``."""
    if path.suffix.lower() == ".nc":
        return path
    suffix = "_pixel_speed_map.png"
    if path.name.endswith(suffix):
        candidate = path.with_name(path.name[: -len(suffix)] + "_speeds.nc")
        if candidate.is_file():
            return candidate
    raise ValueError(f"Cannot resolve a *_speeds.nc file from {path}")


def component_filter(mask: np.ndarray, min_component_pixels: int):
    """Return the retained mask and component-size diagnostics."""
    labels, component_count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel())
    keep_label = sizes >= min_component_pixels
    keep_label[0] = False
    retained = keep_label[labels]
    removed_sizes = sizes[1:][sizes[1:] < min_component_pixels]
    kept_sizes = sizes[1:][sizes[1:] >= min_component_pixels]
    return retained, component_count, kept_sizes, removed_sizes


def coordinate_exclusion_mask(
    ds: xr.Dataset,
    boxes: list[tuple[float, float, float, float]] | None,
) -> np.ndarray:
    """Return the union of explicit ``x_min,x_max,y_min,y_max`` boxes."""
    mask = np.zeros((ds.sizes["yFRF"], ds.sizes["xFRF"]), dtype=bool)
    x = np.asarray(ds["xFRF"].values)
    y = np.asarray(ds["yFRF"].values)
    for x_min, x_max, y_min, y_max in boxes or []:
        if x_min > x_max or y_min > y_max:
            raise ValueError(
                "Exclusion bounds must be ordered x_min <= x_max and "
                f"y_min <= y_max, got {(x_min, x_max, y_min, y_max)}"
            )
        mask |= ((y[:, None] >= y_min) & (y[:, None] <= y_max)
                 & (x[None, :] >= x_min) & (x[None, :] <= x_max))
    return mask


def filter_dataset(
    ds: xr.Dataset,
    min_component_pixels: int,
    exclusion_boxes: list[tuple[float, float, float, float]] | None = None,
):
    """Apply optional component and explicit-coordinate masks consistently."""
    if "mean_speed" not in ds:
        raise ValueError("Dataset does not contain mean_speed")

    mean_speed = np.asarray(ds["mean_speed"].squeeze(drop=True).values)
    if mean_speed.ndim != 2:
        raise ValueError(f"Expected a 2-D mean_speed map after squeeze, got {mean_speed.shape}")

    original_support = np.isfinite(mean_speed)
    if min_component_pixels > 0:
        retained, component_count, kept_sizes, removed_sizes = component_filter(
            original_support, min_component_pixels
        )
        component_removed = original_support & ~retained
    else:
        _, component_count = label(
            original_support, structure=np.ones((3, 3), dtype=np.uint8)
        )
        retained = original_support.copy()
        kept_sizes = np.array([], dtype=int)
        removed_sizes = np.array([], dtype=int)
        component_removed = np.zeros_like(original_support)

    explicit_exclusion = coordinate_exclusion_mask(ds, exclusion_boxes)
    removed = component_removed | explicit_exclusion

    filtered = ds.copy(deep=True)
    # Apply changes only at rejected speed pixels.  In particular, preserve any
    # crest_count-only pixels whose tracks had no finite local-speed estimate.
    keep_da = xr.DataArray(
        ~removed,
        dims=("yFRF", "xFRF"),
        coords={"yFRF": ds["yFRF"], "xFRF": ds["xFRF"]},
    )
    for name in SPEED_VARIABLES:
        if name in filtered:
            filtered[name] = filtered[name].where(keep_da)
    for name in SUPPORT_VARIABLES:
        if name in filtered:
            filtered[name] = filtered[name].where(keep_da, 0)

    now = datetime.now(timezone.utc).isoformat()
    history = filtered.attrs.get("history", "")
    methods = []
    if min_component_pixels > 0:
        methods.append(
            f"removed 8-connected speed-map regions smaller than "
            f"{min_component_pixels} pixels"
        )
    if exclusion_boxes:
        methods.append(f"masked {len(exclusion_boxes)} explicit FRF-coordinate box(es)")
    entry = f"{now}: " + "; ".join(methods)
    attributes = {
            "history": f"{history}; {entry}" if history else entry,
            "postprocessing": "; ".join(methods),
            "postprocessing_components_before": int(component_count),
            "postprocessing_components_retained": int(kept_sizes.size),
            "postprocessing_components_removed": int(removed_sizes.size),
            "postprocessing_component_pixels_removed": int(component_removed.sum()),
            "postprocessing_explicit_cells_masked": int(explicit_exclusion.sum()),
            "postprocessing_finite_speed_pixels_removed": int(
                (original_support & removed).sum()
            ),
    }
    if min_component_pixels > 0:
        attributes["postprocessing_min_component_pixels"] = int(min_component_pixels)
    if exclusion_boxes:
        attributes["postprocessing_exclusion_boxes_xxyy"] = json.dumps(
            [list(map(float, box)) for box in exclusion_boxes]
        )
    filtered.attrs.update(attributes)
    diagnostics = {
        "components_before": int(component_count),
        "components_retained": int(kept_sizes.size),
        "components_removed": int(removed_sizes.size),
        "pixels_before": int(original_support.sum()),
        "pixels_retained": int((original_support & ~removed).sum()),
        "pixels_removed": int((original_support & removed).sum()),
        "component_pixels_removed": int(component_removed.sum()),
        "explicit_cells_masked": int(explicit_exclusion.sum()),
        "explicit_finite_pixels_removed": int(
            (original_support & explicit_exclusion).sum()
        ),
        "largest_removed": int(removed_sizes.max()) if removed_sizes.size else 0,
        "retained_sizes": sorted((int(v) for v in kept_sizes), reverse=True),
    }
    return filtered, diagnostics


def plot_mean_speed(ds: xr.Dataset, output_path: Path) -> None:
    """Re-create the pipeline's per-pixel mean-speed PNG."""
    speed = np.asarray(ds["mean_speed"].squeeze(drop=True).values)
    finite = speed[np.isfinite(speed)]
    vmin, vmax = (np.percentile(finite, [1, 99]) if finite.size else (0.0, 10.0))
    height, width = speed.shape

    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color="white")
    image = ax.pcolormesh(
        np.arange(width), np.arange(height), speed,
        cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
    )
    ax.set_aspect(1)
    ax.set_xlim(0, width - 1)
    pixel_ticks = np.arange(0, height + 1, 200)
    ax.set_yticks(pixel_ticks)
    ax.set_yticklabels(1500 - pixel_ticks)
    fig.colorbar(image, ax=ax, label="Local Wave Speed (m/s)", shrink=0.6)
    ax.set_title("Per-Pixel Mean Wave Speed Map (Post-processed)")
    ax.set_xlabel("Cross-shore Position (xFRF, m)")
    ax.set_ylabel("Alongshore Position (yFRF, m)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def output_paths(source_nc: Path, in_place: bool, output_dir: Path | None = None):
    base = source_nc.name.removesuffix("_speeds.nc")
    source_png = source_nc.with_name(f"{base}_pixel_speed_map.png")
    if in_place:
        return source_nc, source_png
    output_dir = output_dir or source_nc.parent / "postprocessed"
    return (
        output_dir / f"{base}_speeds.nc",
        output_dir / f"{base}_pixel_speed_map.png",
    )


def write_products(
    source_nc: Path,
    filtered: xr.Dataset,
    *,
    in_place: bool,
    force: bool,
    output_dir: Path | None = None,
) -> tuple[Path, Path]:
    output_nc, output_png = output_paths(source_nc, in_place, output_dir)
    output_nc.parent.mkdir(parents=True, exist_ok=True)

    if in_place:
        source_png = output_png
        for source in (source_nc, source_png):
            if not source.exists():
                continue
            backup = source.with_name(source.name + ".pre_postprocess")
            if backup.exists() and not force:
                raise FileExistsError(f"Backup already exists: {backup} (use --force to refresh)")
            shutil.copy2(source, backup)
    elif not force:
        existing = [path for path in (output_nc, output_png) if path.exists()]
        if existing:
            raise FileExistsError(f"Output already exists: {existing[0]} (use --force)")

    temporary_nc = output_nc.with_name(output_nc.name + ".tmp")
    compression = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name, variable in filtered.data_vars.items()
        if "yFRF" in variable.dims and "xFRF" in variable.dims
    }
    filtered.to_netcdf(temporary_nc, encoding=compression)
    temporary_nc.replace(output_nc)
    plot_mean_speed(filtered, output_png)
    return output_nc, output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", type=Path,
        help="One or more *_speeds.nc or *_pixel_speed_map.png files",
    )
    parser.add_argument(
        "--min-component-pixels", type=int, default=2000,
        help=("Minimum retained 8-connected spatial area; use 0 to disable "
              "component filtering (default: 2000 pixels)"),
    )
    parser.add_argument(
        "--exclude-box", action="append", nargs=4, type=float, default=[],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        help=("Mask an explicit inclusive FRF-coordinate box. Repeat for "
              "multiple boxes."),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Write products to this experimental directory",
    )
    parser.add_argument(
        "--in-place", action="store_true",
        help="Replace original products after creating *.pre_postprocess backups",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing outputs/backups")
    args = parser.parse_args()
    if args.min_component_pixels < 0:
        parser.error("--min-component-pixels must be non-negative")
    if args.in_place and args.output_dir is not None:
        parser.error("--output-dir cannot be combined with --in-place")
    if args.min_component_pixels == 0 and not args.exclude_box:
        parser.error("No operation requested: add --exclude-box or a positive component threshold")
    return args


def main() -> None:
    args = parse_args()
    for supplied in args.inputs:
        source_nc = resolve_netcdf(supplied)
        with xr.open_dataset(source_nc) as source:
            filtered, diagnostics = filter_dataset(
                source.load(), args.min_component_pixels, args.exclude_box
            )
        output_nc, output_png = write_products(
            source_nc, filtered, in_place=args.in_place, force=args.force,
            output_dir=args.output_dir,
        )
        print(source_nc.name)
        print(
            "  components: "
            f"{diagnostics['components_before']} -> {diagnostics['components_retained']} "
            f"({diagnostics['components_removed']} removed; "
            f"largest removed={diagnostics['largest_removed']} px)"
        )
        print(
            f"  supported pixels: {diagnostics['pixels_before']} -> "
            f"{diagnostics['pixels_retained']} ({diagnostics['pixels_removed']} removed)"
        )
        if args.exclude_box:
            print(
                "  explicit boxes: "
                f"{diagnostics['explicit_cells_masked']} grid cells; "
                f"{diagnostics['explicit_finite_pixels_removed']} finite speed pixels removed"
            )
        print(f"  NetCDF: {output_nc}")
        print(f"  PNG: {output_png}")


if __name__ == "__main__":
    main()
