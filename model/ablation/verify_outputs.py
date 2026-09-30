#!/usr/bin/env python3
"""Fail unless the canonical study is complete and internally consistent."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .checkpoint import latest_complete_run
from .config import load_all_configs, resolve_path


METRICS = (
    "threshold",
    "dice",
    "iou_fg",
    "boundary_f1",
    "cldice",
    "crest_precision",
    "crest_recall",
    "crest_f1",
)
PRIORITY_MODEL_ID = "segnext_t_learned_up_skip_4_2"
WINNER_METRICS = ("iou_fg", "cldice", "crest_f1")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def require_ids(path: Path, expected: set[str]) -> list[dict[str, str]]:
    rows = read_rows(path)
    actual = {row["model_id"] for row in rows}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"{path}: expected model ids {sorted(expected)}, got {sorted(actual)}"
        )
    return rows


def require_finite(rows: list[dict[str, str]], path: Path) -> None:
    for row in rows:
        for key in METRICS:
            if key in row and not math.isfinite(float(row[key])):
                raise RuntimeError(f"{path}: non-finite {key} for {row.get('model_id')}")


def priority_margins(
    rows: list[dict[str, str]], table_ids: set[str]
) -> dict[str, float]:
    """Report priority-model margins without assuming the desired outcome."""
    indexed = {row["model_id"]: row for row in rows}
    missing = table_ids - set(indexed)
    if missing:
        raise RuntimeError(f"Missing validation rows for {sorted(missing)}")
    if PRIORITY_MODEL_ID not in table_ids:
        raise RuntimeError(f"Priority model {PRIORITY_MODEL_ID} is not in the table")

    competitors = table_ids - {PRIORITY_MODEL_ID}
    margins = {}
    for metric in WINNER_METRICS:
        selected = float(indexed[PRIORITY_MODEL_ID][metric])
        runner_up = max(float(indexed[model_id][metric]) for model_id in competitors)
        margins[metric] = selected - runner_up
    return margins


def verify_outputs() -> dict[str, int | float]:
    manifest, entries = load_all_configs()
    output = resolve_path(manifest["output_root"])
    metrics = output / "metrics"
    tables = output / "tables"

    for _, config in entries:
        latest_complete_run(config)

    all_ids = {config["id"] for _, config in entries}
    table_ids = {config["id"] for entry, config in entries if entry.get("table")}
    missing_ids = {
        config["id"] for entry, config in entries if entry.get("missing_data")
    }
    ood_ids = {config["id"] for entry, config in entries if entry.get("ood")}
    test_ids = {
        config["id"] for entry, config in entries if entry.get("holdout_test")
    }

    threshold_rows = require_ids(metrics / "thresholds.csv", all_ids)
    validation_rows = require_ids(metrics / "validation" / "summary.csv", all_ids)
    missing_rows = require_ids(metrics / "missing_data" / "summary.csv", missing_ids)
    ood_rows = require_ids(metrics / "ood" / "summary.csv", ood_ids)
    test_rows = require_ids(metrics / "test" / "summary.csv", test_ids)
    require_ids(metrics / "costs.csv", table_ids)
    for path, rows in (
        (metrics / "thresholds.csv", threshold_rows),
        (metrics / "validation" / "summary.csv", validation_rows),
        (metrics / "missing_data" / "summary.csv", missing_rows),
        (metrics / "ood" / "summary.csv", ood_rows),
        (metrics / "test" / "summary.csv", test_rows),
    ):
        require_finite(rows, path)
    margins = priority_margins(validation_rows, table_ids)

    expected_tables = {
        "model_comparison.csv": len(table_ids),
        "augmentation.csv": 4,
        "missing_data.csv": 4,
        "ood_swin_segnext.csv": 2,
        "holdout_test.csv": 1,
    }
    for name, count in expected_tables.items():
        rows = read_rows(tables / name)
        if len(rows) != count:
            raise RuntimeError(f"{tables / name}: expected {count} rows, got {len(rows)}")
        tex = tables / name.replace(".csv", ".tex")
        if not tex.is_file() or not tex.read_text().strip():
            raise RuntimeError(f"Missing or empty {tex}")

    for path in (
        metrics / "operating_points" / "pixel_precision_recall.csv",
        metrics / "operating_points" / "crest_precision_recall.csv",
        metrics / "operating_points" / "operating_points.pdf",
        metrics / "operating_points" / "operating_points.png",
        metrics / "examples" / "crest_and_pixel_metrics.csv",
        metrics / "examples" / "crest_and_pixel_metrics.pdf",
        metrics / "examples" / "crest_and_pixel_metrics.png",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty {path}")

    return {
        "runs": len(entries),
        "table_rows": len(table_ids),
        "missing_data_rows": len(missing_ids),
        "ood_rows": len(ood_ids),
        "test_rows": len(test_ids),
        **{f"priority_{key}_margin": value for key, value in margins.items()},
    }


def main() -> None:
    counts = verify_outputs()
    print("Final output verification passed:", counts)


if __name__ == "__main__":
    main()
