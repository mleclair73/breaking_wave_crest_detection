"""DeepLabV3+ adapter with the common logits interface."""

from __future__ import annotations

from typing import Mapping, Any

from torch import nn


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int, classes: int, config: Mapping[str, Any]):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "DeepLabV3+ requires segmentation-models-pytorch. Install the "
                "dependencies with `uv sync --locked` from the repository root."
            ) from exc
        self.model = smp.DeepLabV3Plus(
            # Matches the established wave-crest DeepLab baseline.
            encoder_name=config.get("encoder_name", "mobilenet_v2"),
            encoder_weights="imagenet" if config.get("encoder_pretrained", True) else None,
            encoder_output_stride=int(config.get("encoder_output_stride", 8)),
            in_channels=in_channels,
            classes=classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)
