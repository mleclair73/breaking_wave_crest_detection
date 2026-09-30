"""Hanning-blended tiled segmentation inference."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from common.normalization import MEAN_NP, STD_NP


_BLEND_WEIGHT_CACHE = {}
_BLEND_DENOMINATOR_CACHE = OrderedDict()
_MAX_DENOMINATOR_CACHE_ENTRIES = 16


def _cached_blend_tensors(
    patch_size,
    padded_height,
    padded_width,
    positions,
    use_hanning,
    device,
):
    """Return immutable device-side blend weight and denominator tensors."""
    device = torch.device(device)
    weight_key = (str(device), int(patch_size), bool(use_hanning))
    weight = _BLEND_WEIGHT_CACHE.get(weight_key)
    if weight is None:
        if use_hanning:
            window_1d = np.hanning(patch_size + 2)[1:-1]
            weight = torch.from_numpy(np.outer(window_1d, window_1d)).float()
            weight = weight.to(device)
        else:
            weight = torch.ones(
                (patch_size, patch_size), dtype=torch.float32, device=device
            )
        _BLEND_WEIGHT_CACHE[weight_key] = weight

    positions = tuple(positions)
    denominator_key = (
        weight_key,
        int(padded_height),
        int(padded_width),
        positions,
    )
    denominator = _BLEND_DENOMINATOR_CACHE.get(denominator_key)
    if denominator is None:
        denominator = torch.zeros(
            (padded_height, padded_width), dtype=torch.float32, device=device
        )
        for h_start, w_start in positions:
            denominator[
                h_start : h_start + patch_size,
                w_start : w_start + patch_size,
            ] += weight
        _BLEND_DENOMINATOR_CACHE[denominator_key] = denominator
        if len(_BLEND_DENOMINATOR_CACHE) > _MAX_DENOMINATOR_CACHE_ENTRIES:
            _BLEND_DENOMINATOR_CACHE.popitem(last=False)
    else:
        _BLEND_DENOMINATOR_CACHE.move_to_end(denominator_key)
    return weight, denominator


def predict_tiled(
    model,
    image,
    *,
    patch_size=128,
    overlap_y,
    overlap_x,
    batch_size=32,
    device="cuda",
    skip_zero_patches=False,
):
    """Predict one image with Hanning-blended tiles."""
    return predict_tiled_many(
        model,
        [image],
        patch_size=patch_size,
        overlap_y=overlap_y,
        overlap_x=overlap_x,
        batch_size=batch_size,
        device=device,
        skip_zero_patches=skip_zero_patches,
    )[0]


def predict_tiled_many(
    model,
    images,
    *,
    patch_size=128,
    overlap_y,
    overlap_x,
    batch_size=32,
    device="cuda",
    skip_zero_patches=False,
):
    """Predict equally sized images while batching tiles across images."""
    if model.training:
        model.eval()
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if not 0 <= overlap_y < patch_size:
        raise ValueError(
            f"overlap_y must be in [0, patch_size), got {overlap_y}"
        )
    if not 0 <= overlap_x < patch_size:
        raise ValueError(
            f"overlap_x must be in [0, patch_size), got {overlap_x}"
        )

    prepared = []
    for image in images:
        image = np.asarray(image)
        if image.ndim == 2:
            image = image[..., None]
        if image.ndim != 3 or image.shape[2] not in (1, 3):
            raise ValueError(
                "each image must have shape (H, W), (H, W, 1), or (H, W, 3)"
            )
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        prepared.append(image)
    if not prepared:
        return np.empty((0, 0, 0), dtype=np.float32)
    if len({image.shape for image in prepared}) != 1:
        raise ValueError("all images must have the same shape")

    images = np.stack(prepared).astype(np.float32, copy=False)
    raw_is_zero = (images == 0).all(axis=3) if skip_zero_patches else None
    images = (images - MEAN_NP) / STD_NP

    count, height, width, _ = images.shape
    stride_y = patch_size - overlap_y
    stride_x = patch_size - overlap_x
    if height <= patch_size:
        h_steps = 1
        pad_h = patch_size - height
    else:
        h_steps = int(np.ceil((height - patch_size) / stride_y) + 1)
        pad_h = (h_steps - 1) * stride_y + patch_size - height
    if width <= patch_size:
        w_steps = 1
        pad_w = patch_size - width
    else:
        w_steps = int(np.ceil((width - patch_size) / stride_x) + 1)
        pad_w = (w_steps - 1) * stride_x + patch_size - width
    if pad_h > 0 or pad_w > 0:
        images = np.pad(
            images,
            ((0, 0), (0, pad_h), (0, pad_w), (0, 0)),
            mode="reflect",
        )
        if raw_is_zero is not None:
            raw_is_zero = np.pad(
                raw_is_zero,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )

    padded_height = height + pad_h
    padded_width = width + pad_w
    positions = [
        (i * stride_y, j * stride_x)
        for i in range(h_steps)
        for j in range(w_steps)
    ]
    weight, denominator = _cached_blend_tensors(
        patch_size,
        padded_height,
        padded_width,
        positions,
        overlap_y > 0 or overlap_x > 0,
        device,
    )
    prediction = torch.zeros(
        (count, padded_height, padded_width), dtype=torch.float32, device=device
    )

    patches = []
    destinations = []

    def run_batch():
        batch = torch.stack(patches).to(device)
        with torch.inference_mode():
            logits = model(batch)
            probabilities = (
                torch.sigmoid(logits)[:, 0]
                if logits.shape[1] == 1
                else torch.softmax(logits, dim=1)[:, 1]
            )
        for probability, (image_idx, h_start, w_start) in zip(
            probabilities, destinations
        ):
            prediction[
                image_idx,
                h_start : h_start + patch_size,
                w_start : w_start + patch_size,
            ] += probability * weight

    # Image-major ordering keeps stochastic model state aligned with consecutive
    # single-image calls.
    for image_idx in range(count):
        for h_start, w_start in positions:
            if raw_is_zero is not None and raw_is_zero[
                image_idx,
                h_start : h_start + patch_size,
                w_start : w_start + patch_size,
            ].all():
                continue
            patch = images[
                image_idx,
                h_start : h_start + patch_size,
                w_start : w_start + patch_size,
            ]
            patches.append(torch.from_numpy(patch.transpose(2, 0, 1)).float())
            destinations.append((image_idx, h_start, w_start))
            if len(patches) == batch_size:
                run_batch()
                patches = []
                destinations = []
    if patches:
        run_batch()

    prediction = torch.where(
        denominator[None] > 0,
        prediction / denominator[None],
        torch.zeros_like(prediction),
    )
    return prediction[:, :height, :width].cpu().numpy()
