"""Configuration loading and validation for the canonical ablation study."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


MODEL_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifest.yaml"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else MODEL_ROOT / path


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    return yaml.safe_load(resolved.read_text())


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return load_yaml(path)


def load_model_config(entry: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return _merge(load_yaml(manifest["common_config"]), load_yaml(entry["config"]))


def load_training_config(
    model_config: str | Path,
    common_config: str | Path = "ablation/configs/common.yaml",
) -> dict[str, Any]:
    """Resolve one model config against the frozen common training recipe."""
    return _merge(load_yaml(common_config), load_yaml(model_config))


def load_all_configs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    manifest = load_manifest(manifest_path)
    resolved = [(entry, load_model_config(entry, manifest)) for entry in manifest["models"]]
    return manifest, resolved
