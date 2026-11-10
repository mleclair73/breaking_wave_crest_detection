"""Device selection, strict model-checkpoint loading, and inference seeding."""

from __future__ import annotations

from pathlib import Path

import torch

from models import build_model


def get_device(device_name: str | None = None) -> torch.device:
    """Resolve an explicit device or auto-detect CUDA, MPS, then CPU."""
    if device_name and device_name.lower() != "auto":
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def model_from_payload(
    payload: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from a checkpoint's stored config and load it strictly."""
    config = dict(payload["config"])
    # The checkpoint holds the full encoder; never fetch pretrained weights.
    config["encoder_pretrained"] = False
    model = build_model(config)
    model.load_state_dict(clean_state_dict(payload["model_state_dict"]), strict=True)
    return model.to(device).eval(), config


def load_checkpoint_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return model_from_payload(payload, device)


def seed_inference(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
