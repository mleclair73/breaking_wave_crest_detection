"""Validate model-study configuration and required local inputs."""

from __future__ import annotations

from .config import load_all_configs, resolve_path


def verify_inputs() -> None:
    """Check that datasets and pretrained weights exist."""
    _, entries = load_all_configs()
    for _, config in entries:
        dataset = resolve_path(config["dataset_root"])
        for required in ("image_mask_mapping.csv", "images", "masks"):
            if not (dataset / required).exists():
                raise FileNotFoundError(dataset / required)
        if config["arch"] == "segnext":
            pretrained = resolve_path(config["pretrained_path"])
            if not pretrained.is_file():
                raise FileNotFoundError(pretrained)


def main() -> None:
    verify_inputs()
    print("Dataset and weight verification passed.")


if __name__ == "__main__":
    main()
