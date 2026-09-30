"""Strict model-checkpoint loading and inference seeding."""

from __future__ import annotations

from pathlib import Path

import torch

from models import build_model


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def load_checkpoint_model(
    run: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict]:
    payload = torch.load(
        run / "best_model.pth", map_location=device, weights_only=False
    )
    config = dict(payload["config"])
    config["encoder_pretrained"] = False
    model = build_model(config)
    model.load_state_dict(clean_state_dict(payload["model_state_dict"]), strict=True)
    return model.to(device).eval(), config


def seed_inference(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
