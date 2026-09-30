"""ImageNet normalization constants and conversion helpers."""

import numpy as np
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEAN_NP = np.asarray(IMAGENET_MEAN, dtype=np.float32)
STD_NP = np.asarray(IMAGENET_STD, dtype=np.float32)
MEAN = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(3, 1, 1)


def denorm(x):
    """Normalized tensor -> [0, 1] image space (clamped)."""
    return (x * STD.to(x) + MEAN.to(x)).clamp(0, 1)


def renorm(x):
    """[0, 1] image space -> ImageNet-normalized tensor."""
    return (x - MEAN.to(x)) / STD.to(x)


def minmax(x, eps=1e-8):
    """Rescale to [0, 1] by its own min/max. Works for torch tensors or numpy arrays.

    Used for visualization (attention/activation maps), not for model input.
    """
    return (x - x.min()) / (x.max() - x.min() + eps)
