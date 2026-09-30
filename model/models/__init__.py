"""Segmentation model registry."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn


def build_model(config: Mapping[str, Any]) -> nn.Module:
    """Build one of the architectures retained in the study."""
    architecture = str(config["arch"]).lower()
    in_channels = int(config.get("in_channels", 3))
    classes = int(config.get("num_classes", 2))

    if architecture == "segnext":
        from .segnext import SegNeXt

        return SegNeXt(
            in_chans=in_channels,
            num_classes=classes,
            decoder_type=str(config["decoder_type"]),
            drop_path_rate=float(config.get("drop_path_rate", 0.1)),
            dropout_ratio=float(config.get("dropout_ratio", 0.1)),
            ham_kwargs=config.get("ham_kwargs"),
            padding_mode=str(config.get("padding_mode", "reflect")),
            decoder_upsample_channels=config.get("decoder_upsample_channels"),
            decoder_tap_half_channels=(
                int(config["decoder_tap_half_channels"])
                if "decoder_tap_half_channels" in config
                else None
            ),
            decoder_skips=config.get("decoder_skips"),
        )
    if architecture == "unet":
        from .unet import ResNetUNet

        return ResNetUNet(
            n_class=classes,
            in_channels=in_channels,
            pretrained=bool(config.get("encoder_pretrained", True)),
        )
    if architecture == "attention_unet":
        from .attention_unet import AttentionUNet

        return AttentionUNet(
            in_channels=in_channels,
            classes=classes,
            pretrained=bool(config.get("encoder_pretrained", True)),
        )
    if architecture == "deeplabv3plus":
        from .deeplabv3plus import DeepLabV3Plus

        return DeepLabV3Plus(in_channels=in_channels, classes=classes, config=config)
    if architecture == "swin_unet":
        from .swin_unet import SwinUNet

        return SwinUNet(
            in_channels=in_channels,
            classes=classes,
            encoder_name=str(
                config.get("swin_encoder", "swin_tiny_patch4_window7_224")
            ),
            pretrained=bool(config.get("encoder_pretrained", True)),
            dropout_ratio=float(config.get("dropout_ratio", 0.0)),
            drop_path_rate=float(config.get("drop_path_rate", 0.0)),
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


__all__ = ["build_model"]
