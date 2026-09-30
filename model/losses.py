"""Segmentation losses.

This is the minimal binary-segmentation subset of the project's historical
``unified_focal_loss_pytorch.py`` implementation.  Keeping it here makes the
ablation package self-contained and freezes the exact loss equations
used by every run in the comparison.
"""

import torch
from torch import nn


def _spatial_axes(shape: torch.Size) -> tuple[int, ...]:
    if len(shape) == 4:
        return (2, 3)
    if len(shape) == 5:
        return (2, 3, 4)
    raise ValueError("Expected a 2-D or 3-D segmentation tensor")


class AsymmetricFocalLoss(nn.Module):
    def __init__(self, delta: float = 0.7, gamma: float = 2.0, epsilon: float = 1e-7):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.clamp(prediction, self.epsilon, 1.0 - self.epsilon)
        cross_entropy = -target * torch.log(prediction)
        background = (1.0 - prediction[:, 0]) ** self.gamma * cross_entropy[:, 0]
        background = (1.0 - self.delta) * background
        foreground = self.delta * cross_entropy[:, 1]
        return torch.mean(torch.sum(torch.stack((background, foreground), dim=-1), dim=-1))


class AsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta: float = 0.7, gamma: float = 0.75, epsilon: float = 1e-7):
        super().__init__()
        self.delta = delta
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = torch.clamp(prediction, self.epsilon, 1.0 - self.epsilon)
        axes = _spatial_axes(target.size())
        true_positive = torch.sum(target * prediction, dim=axes)
        false_negative = torch.sum(target * (1.0 - prediction), dim=axes)
        false_positive = torch.sum((1.0 - target) * prediction, dim=axes)
        score = (true_positive + self.epsilon) / (
            true_positive
            + self.delta * false_negative
            + (1.0 - self.delta) * false_positive
            + self.epsilon
        )
        background = 1.0 - score[:, 0]
        foreground = (1.0 - score[:, 1]) * (1.0 - score[:, 1]) ** (-self.gamma)
        return torch.mean(torch.stack((background, foreground), dim=-1))


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Weighted asymmetric focal-Tversky and focal cross-entropy loss."""

    def __init__(
        self,
        weight: float | None = 0.5,
        delta: float = 0.6,
        gamma: float = 0.75,
        gamma_focal: float = 2.0,
    ):
        super().__init__()
        self.weight = weight
        self.tversky = AsymmetricFocalTverskyLoss(delta=delta, gamma=gamma)
        self.focal = AsymmetricFocalLoss(delta=delta, gamma=gamma_focal)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky = self.tversky(prediction, target)
        focal = self.focal(prediction, target)
        if self.weight is None:
            return tversky + focal
        return self.weight * tversky + (1.0 - self.weight) * focal
