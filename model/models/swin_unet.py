"""Swin encoder with a convolutional U-Net decoder.

The decoder is local to this project; timm supplies only the optional ImageNet
pretrained Swin feature extractor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DecoderBlock(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin + skip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU(),
                                 nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.GELU())
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.net(torch.cat((x, skip), dim=1))


class SwinUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        classes=2,
        encoder_name="swin_tiny_patch4_window7_224",
        pretrained=True,
        dropout_ratio=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Swin-UNet requires timm; install it before selecting arch: swin_unet.") from exc
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=float(drop_path_rate),
        )
        channels = self.encoder.feature_info.channels()
        self.d3 = DecoderBlock(channels[3], channels[2], channels[2])
        self.d2 = DecoderBlock(channels[2], channels[1], channels[1])
        self.d1 = DecoderBlock(channels[1], channels[0], channels[0])
        self.stem = nn.Sequential(nn.Conv2d(channels[0], channels[0] // 2, 3, padding=1), nn.GELU())
        self.dropout = nn.Dropout2d(float(dropout_ratio)) if dropout_ratio > 0 else nn.Identity()
        self.head = nn.Conv2d(channels[0] // 2, classes, 1)

    def _decode(self, x):
        features = [f.permute(0, 3, 1, 2).contiguous() if f.ndim == 4 and f.shape[-1] != f.shape[1] else f for f in self.encoder(x)]
        y = self.d3(features[3], features[2])
        y = self.d2(y, features[1])
        return self.d1(y, features[0])

    def forward(self, x):
        out_size = x.shape[-2:]
        fine = self._decode(x)
        y = F.interpolate(fine, size=out_size, mode="bilinear", align_corners=False)
        return self.head(self.dropout(self.stem(y)))
