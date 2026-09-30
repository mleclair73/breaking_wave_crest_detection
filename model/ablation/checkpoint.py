"""Run discovery and strict checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import yaml

from .config import resolve_path


def complete_run(path: Path) -> bool:
    return (
        (path / "best_model.pth").is_file()
        and (path / "metrics" / "history.json").is_file()
        and (path / "metrics" / "summary.json").is_file()
        and (path / "provenance.json").is_file()
        and (path / "complete.json").is_file()
    )


def run_matches_config(path: Path, config: dict) -> bool:
    """Reject stale runs when a config id is reused for a revised model."""
    config_path = path / "config.yaml"
    if not config_path.is_file():
        return False
    saved = yaml.safe_load(config_path.read_text())
    if not isinstance(saved, dict):
        return False
    # These fields do not affect learned weights. In particular, tile_overlap
    # is a complete-image inference policy and may change for an evaluation
    # rerun without invalidating a trained checkpoint.
    ignored = {
        "label",
        "output_dir",
        "tile_overlap",
        # Execution and logging settings do not alter the indexed sample RNG,
        # optimizer, schedule, or learned model.
        "num_workers",
        "torch_num_threads",
        "persistent_workers",
        "prefetch_factor",
        "cache_images",
        "visualize_every",
    }
    expected = {key: value for key, value in config.items() if key not in ignored}
    actual = {key: value for key, value in saved.items() if key not in ignored}
    return actual == expected


def latest_complete_run(config: dict) -> Path:
    output = resolve_path(config["output_dir"])
    runs = sorted(
        path
        for path in output.glob("run_*")
        if complete_run(path) and run_matches_config(path, config)
    )
    if not runs:
        raise FileNotFoundError(f"No complete run for {config['id']} under {output}")
    return runs[-1]


def latest_incomplete_run(config: dict) -> Path | None:
    output = resolve_path(config["output_dir"])
    runs = sorted(
        path
        for path in output.glob("run_*")
        if (
            (path / "last_model.pth").is_file()
            and not complete_run(path)
            and run_matches_config(path, config)
        )
    )
    return runs[-1] if runs else None
