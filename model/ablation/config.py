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
    value = yaml.safe_load(resolved.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {resolved}")
    return value


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    manifest = load_yaml(path)
    entries = manifest.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The manifest must contain at least one training run")
    return manifest


def load_model_config(entry: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    config = _merge(load_yaml(manifest["common_config"]), load_yaml(entry["config"]))
    validate_config(config)
    return config


def load_training_config(
    model_config: str | Path,
    common_config: str | Path = "ablation/configs/common.yaml",
) -> dict[str, Any]:
    """Resolve one model config against the frozen common training recipe."""
    config = _merge(load_yaml(common_config), load_yaml(model_config))
    validate_config(config)
    return config


def load_all_configs(
    manifest_path: str | Path = DEFAULT_MANIFEST,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], dict[str, Any]]]]:
    manifest = load_manifest(manifest_path)
    resolved = [(entry, load_model_config(entry, manifest)) for entry in manifest["models"]]
    ids = [config["id"] for _, config in resolved]
    labels = [config["label"] for _, config in resolved]
    if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
        raise ValueError("Model ids and labels must be unique")
    return manifest, resolved


def validate_config(config: dict[str, Any]) -> None:
    retired = {
        "augment_strength", "aug_overrides", "aux_loss_weights",
        "aux_supervision", "cldice_iterations", "cldice_loss_weight",
        "decoder_fusion_type", "decoder_upsample_type",
        "detail_skip_dropout", "disable_amp", "encoder_norm",
        "freeze_backbone_stages", "init_from_checkpoint", "loss_type",
        "model_type",
    }
    present_retired = sorted(retired & set(config))
    if present_retired:
        raise ValueError(
            f"{config.get('id', '<unknown>')}: retired options {present_retired}"
        )
    common_keys = {
        "id", "label", "arch", "dataset_root", "image_size", "in_channels",
        "num_classes", "patches_per_image", "augment_train", "epochs",
        "batch_size", "num_workers", "lr", "weight_decay", "seed",
        "output_dir", "loss_params", "tile_overlap",
        "study_version", "augmentation_version", "ood_dataset_root",
        "torch_num_threads", "early_stop_patience", "select_metric",
        "persistent_workers", "prefetch_factor", "cache_images",
        "visualize_every",
    }
    required = common_keys
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"{config.get('id', '<unknown>')}: missing keys {missing}")
    architecture_keys = {
        "unet": {"encoder_pretrained"},
        "attention_unet": {"encoder_pretrained"},
        "deeplabv3plus": {
            "encoder_name", "encoder_output_stride", "encoder_pretrained",
        },
        "swin_unet": {
            "swin_encoder", "encoder_pretrained", "drop_path_rate",
            "dropout_ratio",
        },
        "segnext": {
            "decoder_type", "decoder_skips", "decoder_upsample_channels",
            "decoder_tap_half_channels", "drop_path_rate", "dropout_ratio",
            "padding_mode", "ham_kwargs", "pretrained_path",
        },
    }
    architecture = config["arch"]
    if architecture not in architecture_keys:
        raise ValueError(f"{config['id']}: unsupported architecture {architecture}")
    unknown = sorted(set(config) - common_keys - architecture_keys[architecture])
    if unknown:
        raise ValueError(f"{config['id']}: unknown or unused options {unknown}")
    if int(config["seed"]) != 42:
        raise ValueError(f"{config['id']}: study seed must be 42")
    if int(config["image_size"]) != 224:
        raise ValueError(f"{config['id']}: study image size must be 224")
    if not 0 <= int(config["tile_overlap"]) < int(config["image_size"]):
        raise ValueError(
            f"{config['id']}: tile_overlap must be in [0, image_size)"
        )
    if int(config["batch_size"]) != 32:
        raise ValueError(f"{config['id']}: study batch size must be 32")
    positive_integers = (
        "patches_per_image", "epochs", "torch_num_threads",
        "early_stop_patience", "prefetch_factor", "visualize_every",
    )
    if any(int(config[key]) <= 0 for key in positive_integers):
        raise ValueError(f"{config['id']}: positive integer option is not positive")
    if int(config["num_workers"]) < 0:
        raise ValueError(f"{config['id']}: num_workers must be non-negative")
    if float(config["lr"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError(f"{config['id']}: invalid optimizer settings")
    if set(config["loss_params"]) != {
        "weight", "delta", "gamma", "gamma_focal"
    }:
        raise ValueError(f"{config['id']}: unexpected loss_params schema")
    if config.get("select_metric", "iou_fg") != "iou_fg":
        raise ValueError(f"{config['id']}: study selects foreground IoU")
    if config["arch"] == "segnext":
        allowed = {"ham", "learned_up"}
        if config.get("decoder_type") not in allowed:
            raise ValueError(
                f"{config['id']}: decoder_type must be one of {sorted(allowed)}"
            )
        upsample_channels = config.get("decoder_upsample_channels")
        if config.get("decoder_type") == "learned_up":
            if (
                not isinstance(upsample_channels, list)
                or len(upsample_channels) != 3
                or any(int(value) <= 0 for value in upsample_channels)
            ):
                raise ValueError(
                    f"{config['id']}: decoder_upsample_channels must contain "
                    "three positive integers"
                )
            decoder_skips = config.get("decoder_skips")
            if (
                not isinstance(decoder_skips, list)
                or len(decoder_skips) != len(set(decoder_skips))
                or decoder_skips not in ([], [4], [4, 2])
            ):
                raise ValueError(
                    f"{config['id']}: decoder_skips must be [], [4], or [4, 2]"
                )
            tap_channels = config.get("decoder_tap_half_channels")
            if 2 in decoder_skips:
                if tap_channels is None or int(tap_channels) <= 0:
                    raise ValueError(
                        f"{config['id']}: an H/2 skip requires a positive "
                        "decoder_tap_half_channels"
                    )
            elif "decoder_tap_half_channels" in config:
                raise ValueError(
                    f"{config['id']}: decoder_tap_half_channels is unused "
                    "without an H/2 skip"
                )
        elif any(
            key in config
            for key in (
                "decoder_skips", "decoder_upsample_channels",
                "decoder_tap_half_channels",
            )
        ):
            raise ValueError(
                f"{config['id']}: learned-up options require decoder_type=learned_up"
            )
