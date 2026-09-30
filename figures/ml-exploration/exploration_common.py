"""Shared runtime configuration for the ML-exploration figure scripts.

The analyses intentionally remain runnable as standalone scripts.  This module
keeps their repository paths, production checkpoint loading, device selection,
and common command-line options in one place.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


ML_DIR = Path(__file__).resolve().parent
FIGURES_DIR = ML_DIR.parent
REPO_ROOT = FIGURES_DIR.parent
MODEL_ROOT = REPO_ROOT / "model"

import dunex_paths
from common.module_loading import load_module_from_path
from segmentation.predict_video import (
    PRODUCTION_THRESHOLD as PRODUCTION_THRESHOLD,
    get_device,
    load_model,
)


DEFAULT_CHECKPOINT = (
    dunex_paths.OUTPUTS_DIR
    / "promoted/segnext_t_learned_up_skip_4_2/best_model.pth"
)
DEFAULT_DATASET = MODEL_ROOT / "data/full_split"
DEFAULT_IMAGE = (
    DEFAULT_DATASET / "images/ArgusFF_20211007T210100Z_y0600_f1624.png"
)
DEFAULT_OOD_IMAGES = MODEL_ROOT / "data/ood_test/images"


@dataclass(frozen=True)
class Runtime:
    """Resolved model, device, and output directory for one figure run."""

    model: object
    device: object
    output_dir: Path
    checkpoint: Path


def import_figure(module_name: str) -> ModuleType:
    """Import a repository figure module whose filename begins with a number."""
    candidates = (
        ML_DIR / f"{module_name}.py",
        FIGURES_DIR / f"{module_name}.py",
    )
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise ImportError(
            f"Expected one figure module named {module_name!r}, found {matches}"
        )
    return load_module_from_path(
        f"_breaking_wave_figure_{module_name}", matches[0]
    )


def add_runtime_arguments(
    parser: argparse.ArgumentParser,
    *,
    image: bool = False,
) -> None:
    """Add consistent, portable runtime flags to a figure parser."""
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="production checkpoint (default: canonical promoted checkpoint)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="inference device (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ML_DIR,
        help="directory for PNG output and its pdf/ sidecar directory",
    )
    if image:
        parser.add_argument(
            "--image",
            type=Path,
            default=DEFAULT_IMAGE,
            help="input timestack image",
        )


def load_runtime(args: argparse.Namespace) -> Runtime:
    """Strictly load the production model and resolve the output directory."""
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Production checkpoint not found: {checkpoint}\n"
            "Pass --checkpoint or set OUTPUTS_DIR to the external artifact root."
        )
    device = get_device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    model, _ = load_model(checkpoint, device=device)
    return Runtime(model, device, output_dir, checkpoint)


def require_file(path: Path, label: str = "input") -> Path:
    """Resolve a required external/input file with a useful error."""
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label.capitalize()} not found: {resolved}")
    return resolved
