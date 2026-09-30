"""Deterministic fingerprints for ablation training runs."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from .config import MODEL_ROOT, resolve_path


TRAINING_SOURCES = (
    "ablation/checkpoint.py",
    "ablation/config.py",
    "ablation/provenance.py",
    "augmentations.py",
    "common/normalization.py",
    "dataset.py",
    "losses.py",
    "train.py",
    "models/__init__.py",
    "models/attention_unet.py",
    "models/deeplabv3plus.py",
    "models/segnext.py",
    "models/swin_unet.py",
    "models/unet.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(model: torch.nn.Module) -> str:
    """Hash tensor names, metadata, and bytes in deterministic key order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _distribution_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def build_provenance(config: dict, model: torch.nn.Module) -> dict:
    sources = {
        relative: sha256_file(MODEL_ROOT / relative)
        for relative in TRAINING_SOURCES
    }
    inputs = {}
    for key in ("dataset_root", "ood_dataset_root"):
        root = resolve_path(config[key])
        mapping = root / "image_mask_mapping.csv"
        inputs[str(mapping.relative_to(MODEL_ROOT))] = sha256_file(mapping)
    if config.get("pretrained_path"):
        pretrained = resolve_path(config["pretrained_path"])
        inputs[str(pretrained.relative_to(MODEL_ROOT))] = sha256_file(pretrained)
    return {
        "schema_version": 1,
        "study_version": config["study_version"],
        "augmentation_version": config["augmentation_version"],
        "model_id": config["id"],
        "sources_sha256": sources,
        "inputs_sha256": inputs,
        "initial_model_state_sha256": state_dict_sha256(model),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "packages": {
                name: _distribution_version(name)
                for name in (
                    "torchvision",
                    "scipy",
                    "scikit-image",
                    "opencv-python-headless",
                    "Pillow",
                    "timm",
                    "segmentation-models-pytorch",
                )
            },
        },
    }


def write_or_validate_provenance(
    run_dir: Path, config: dict, model: torch.nn.Module
) -> dict:
    current = build_provenance(config, model)
    path = run_dir / "provenance.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        stable_fields = (
            "study_version",
            "augmentation_version",
            "model_id",
            "sources_sha256",
            "inputs_sha256",
            "initial_model_state_sha256",
        )
        changed = [key for key in stable_fields if saved.get(key) != current[key]]
        if changed:
            raise RuntimeError(
                "Refusing to resume with changed training provenance: "
                + ", ".join(changed)
            )
        return saved
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    return current
