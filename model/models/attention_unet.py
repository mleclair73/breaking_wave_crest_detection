"""Attention-gated U-Net for binary and multiclass segmentation."""

from __future__ import annotations

import torch.nn.functional as F
import torch
from torch import nn
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.layers(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, intermediate: int):
        super().__init__()
        self.skip = nn.Sequential(nn.Conv2d(skip_channels, intermediate, 1, bias=False), nn.BatchNorm2d(intermediate))
        self.gate = nn.Sequential(nn.Conv2d(gate_channels, intermediate, 1, bias=False), nn.BatchNorm2d(intermediate))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(intermediate, 1, 1), nn.Sigmoid())

    def forward(self, x, g):
        g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * self.psi(self.skip(x) + self.gate(g))


class AttentionUNet(nn.Module):
    """Oktay attention gates over an ImageNet-pretrained ResNet-18 encoder."""
    def __init__(self, in_channels=3, classes=2, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        encoder = models.resnet18(weights=weights)
        if in_channels != 3:
            old_conv = encoder.conv1
            encoder.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    encoder.conv1.weight.copy_(old_conv.weight.mean(1, keepdim=True).repeat(1, in_channels, 1, 1) / in_channels)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2 = encoder.maxpool, encoder.layer1, encoder.layer2
        self.layer3, self.layer4 = encoder.layer3, encoder.layer4
        self.g4, self.g3 = AttentionGate(256, 512, 128), AttentionGate(128, 256, 64)
        self.g2, self.g1 = AttentionGate(64, 128, 32), AttentionGate(64, 64, 32)
        self.dec4, self.dec3 = ConvBlock(512 + 256, 256), ConvBlock(256 + 128, 128)
        self.dec2, self.dec1 = ConvBlock(128 + 64, 64), ConvBlock(64 + 64, 64)
        self.head = nn.Conv2d(64, classes, 1)

    @staticmethod
    def _up(x, ref): return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        out_size = x.shape[-2:]
        e1 = self.stem(x); e2 = self.layer1(self.pool(e1)); e3 = self.layer2(e2)
        e4 = self.layer3(e3); b = self.layer4(e4)
        d4 = self.dec4(torch.cat([self._up(b, e4), self.g4(e4, b)], 1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), self.g3(e3, d4)], 1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), self.g2(e2, d3)], 1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), self.g1(e1, d2)], 1))
        return F.interpolate(self.head(d1), size=out_size, mode="bilinear", align_corners=False)
