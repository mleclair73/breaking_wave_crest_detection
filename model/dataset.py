"""Wave-crest segmentation dataset."""

import csv
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from numpy.random import RandomState
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from augmentations import (
    HEAVY_AUGMENTATION,
    add_gaussian_noise,
    add_water_droplets,
    apply_angled_gradient_transform,
    apply_brightness_contrast,
    apply_directional_motion_blur,
    apply_elastic_transform,
    apply_gaussian_blur,
    apply_glare_transform,
    apply_glint_speckle_transform,
    apply_horizontal_scale_transform,
    apply_local_gamma_transform,
    apply_missing_data_transform,
    apply_patch_gaussian_blur,
    apply_random_lines,
    apply_sharpen,
    apply_specular_streak_transform,
    apply_stationary_band_transform,
    apply_vertical_gradient_transform,
)
from common.normalization import IMAGENET_MEAN, IMAGENET_STD


class WaveBreakingDataset(Dataset):
    """
    Indexed crop dataset for wave-crest segmentation.

    Features:
    - Loads RGB images and corresponding binary masks
    - Filters the mapping by split
    - Uses deterministic per-index, per-epoch crop and augmentation streams
    - Optionally caches decoded source images per worker

    Args:
        dataset_root: Path to dataset directory containing images/, masks/, and image_mask_mapping.csv
        split: 'train', 'val', or 'all'
        transform_size: Target size for cropping (default 224)
        patches_per_image: Number of indexed crops per image (default 1)
    """

    def __init__(self, dataset_root, split='train', transform_size=224, patches_per_image=1,
                 augment=True, seed=42,
                 cache_images=False):
        """Create the fixed study training or deterministic validation dataset."""
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / 'images'
        self.masks_dir = self.dataset_root / 'masks'
        self.mapping_file = self.dataset_root / 'image_mask_mapping.csv'
        self.transform_size = transform_size
        self.split = split
        self.patches_per_image = patches_per_image
        self.augment = augment and (split == 'train')  # Only augment training data
        self.seed = seed
        # Shared memory keeps persistent DataLoader workers synchronized with
        # the epoch selected by the training process.
        self._epoch = mp.Value('q', 0, lock=True)
        self.cache_images = bool(cache_images)

        for name, value in HEAVY_AUGMENTATION.items():
            setattr(self, name, value)

        if not self.augment:
            self.missing_data_prob = 0.0

        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")
        if not self.mapping_file.exists():
            raise ValueError(f"Mapping file not found: {self.mapping_file}")

        self.samples = self._load_mapping()
        # Valid non-padding bounds depend only on the source image, not on the
        # sampled crop. Cache them per worker instead of scanning the full
        # source image for every patch.
        self._valid_bounds = [None] * len(self.samples)

        # The dataset is small relative to RunPod memory. Decoding each source
        # image once before DataLoader workers fork avoids repeated PNG opens
        # across the 64 random crops drawn from every image.
        self._cached_pairs = None
        if self.cache_images:
            self._cached_pairs = []
            for image_idx, sample in enumerate(self.samples):
                with Image.open(self.images_dir / sample['image_name']) as image:
                    cached_image = image.convert('RGB').copy()
                with Image.open(self.masks_dir / sample['mask_name']) as mask:
                    cached_mask = mask.copy()
                self._cached_pairs.append((cached_image, cached_mask))
                self._valid_bounds[image_idx] = self._find_valid_bounds(cached_image)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        num_images = len(self.samples)
        num_patches = num_images * patches_per_image
        aug_status = "with augmentation" if self.augment else "no augmentation"
        print(f"Loaded WaveBreakingDataset: {num_images} images × {patches_per_image} patches = {num_patches} samples ({split} split, {aug_status})")

    def _load_mapping(self):
        """Load image-mask mapping from CSV and filter by split"""
        samples = []

        with open(self.mapping_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.split == 'all' or row['split'] == self.split:
                    samples.append({
                        'image_name': row['image_name'],
                        'mask_name': row['mask_name'],
                        'split': row['split']
                    })

        if len(samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}'")

        return samples

    def __len__(self):
        return len(self.samples) * self.patches_per_image

    def set_epoch(self, epoch):
        """Select the deterministic augmentation stream for an epoch."""
        with self._epoch.get_lock():
            self._epoch.value = int(epoch)

    def _sample_seed(self, idx, stream=0):
        with self._epoch.get_lock():
            epoch = int(self._epoch.value)
        # Epoch zero reproduces the original sample seed; stream offsets keep
        # crop, NumPy, and Torch randomness independent.
        return int(
            (int(self.seed) + int(idx) + epoch * 1_000_003 + int(stream) * 100_000_007)
            % (2**32 - 1)
        )

    @staticmethod
    def _find_valid_bounds(image):
        """Return the non-black source extent as ``top, bottom, left, right``."""
        width, height = image.size
        array = np.asarray(image)
        if array.ndim == 2:
            valid = array != 0
        else:
            valid = np.any(array != 0, axis=2)
        rows = np.flatnonzero(np.any(valid, axis=1))
        cols = np.flatnonzero(np.any(valid, axis=0))
        if rows.size and cols.size:
            return int(rows[0]), int(rows[-1] + 1), int(cols[0]), int(cols[-1] + 1)
        return 0, height, 0, width

    def _apply_synchronized_transform(self, image, mask, image_idx, patch_idx, idx):
        """
        Apply synchronized transforms to image and mask

        Args:
            image: PIL Image (grayscale)
            mask: PIL Image (binary mask)
            image_idx: Image index (which original image)
            patch_idx: Patch index (which patch from this image)
            idx: Global sample index (for augmentation seeding)

        Returns:
            image_tensor: torch.Tensor (3, H, W) - RGB, normalized
            mask_tensor: torch.Tensor (1, H, W) - binary {0, 1}
        """
        image = image.convert('RGB')

        # Get original image dimensions and reuse the source's valid region.
        orig_w, orig_h = image.size
        bounds = self._valid_bounds[image_idx]
        if bounds is None:
            bounds = self._find_valid_bounds(image)
            self._valid_bounds[image_idx] = bounds
        valid_top, valid_bottom, valid_left, valid_right = bounds

        valid_h = valid_bottom - valid_top
        valid_w = valid_right - valid_left

        # Decide whether to use centered crop (no black edges) or edge crop (with black)
        # Val/test: always centered (no black edges), deterministic for reproducible eval
        # Train: 90% centered, 10% edge crops
        if self.split in ('val', 'test'):
            use_centered_crop = True
            # Deterministic seed for validation/test crops
            crop_rng = random.Random(42 + int(image_idx) * 1000 + int(patch_idx))
        else:
            crop_rng = random.Random(self._sample_seed(idx, stream=1))
            use_centered_crop = crop_rng.random() < 0.9  # 90% centered

        if use_centered_crop and valid_h >= self.transform_size and valid_w >= self.transform_size:
            # Centered crop: sample only from valid region (no black edges)
            max_top = valid_bottom - self.transform_size
            max_left = valid_right - self.transform_size
            i = crop_rng.randint(valid_top, max(valid_top, max_top))
            j = crop_rng.randint(valid_left, max(valid_left, max_left))
            image = TF.crop(image, i, j, self.transform_size, self.transform_size)
            mask = TF.crop(mask, i, j, self.transform_size, self.transform_size)
        else:
            # Edge crop: add padding and allow sampling past edges
            pad_size = self.transform_size // 2
            image = TF.pad(image, padding=pad_size, fill=0)
            mask = TF.pad(mask, padding=pad_size, fill=0)
            padded_w, padded_h = image.size
            i = crop_rng.randint(0, padded_h - self.transform_size)
            j = crop_rng.randint(0, padded_w - self.transform_size)
            image = TF.crop(image, i, j, self.transform_size, self.transform_size)
            mask = TF.crop(mask, i, j, self.transform_size, self.transform_size)

        image_tensor = self.to_tensor(image)  # (3, H, W) in [0, 1]
        mask_tensor = self.to_tensor(mask)     # (1, H, W) in [0, 1]

        # Apply augmentations BEFORE normalization (on [0, 1] range images)
        if self.augment:
            random_state = RandomState(self._sample_seed(idx, stream=0))
            noise_generator = np.random.default_rng(self._sample_seed(idx, stream=2))
            torch_generator = torch.Generator(device='cpu')
            torch_generator.manual_seed(self._sample_seed(idx, stream=3))

            # Create a mask for valid (non-padding) pixels - pixels that have any non-zero RGB value
            # This prevents augmentations from affecting the black padding regions
            valid_mask = (image_tensor.sum(dim=0, keepdim=True) > 0).float()  # (1, H, W)

            # Add gaussian noise to ALL training samples (only to valid regions)
            noise_augmented = add_gaussian_noise(
                image_tensor,
                noise_std=random_state.uniform(0.005, 0.03),
                numpy_generator=noise_generator,
            )
            image_tensor = noise_augmented * valid_mask  # Padding stays zero

            if random_state.random() < self.p_augmentation:
                # Keep the order and random draws fixed: training reproducibility relies on it.
                image_augmentations = (
                    (self.p_brightness, apply_brightness_contrast, {}),
                    (self.p_gradient, apply_vertical_gradient_transform, {}),
                    (self.p_gradient, apply_angled_gradient_transform, {}),
                    (self.p_glare, apply_glare_transform, {}),
                    (self.p_extreme_glare, apply_glare_transform, {'extreme': True}),
                    (self.p_water_droplet, add_water_droplets, {}),
                    (self.p_local_gamma, apply_local_gamma_transform, {}),
                    (self.p_blur, apply_gaussian_blur, {}),
                    (self.p_motion_blur, apply_directional_motion_blur, {}),
                    (self.p_blur, apply_sharpen, {}),
                    (self.p_patch_blur, apply_patch_gaussian_blur, {}),
                    (self.p_random_lines, apply_random_lines, {}),
                    (self.missing_data_prob, apply_missing_data_transform, {}),
                    (self.p_specular_streak, apply_specular_streak_transform, {}),
                    (self.p_stationary_band, apply_stationary_band_transform, {}),
                    (self.p_glint_speckle, apply_glint_speckle_transform, {}),
                )
                for probability, transform, kwargs in image_augmentations:
                    # Disabled transforms do not consume a random number.
                    if probability > 0 and random_state.random() < probability:
                        image_tensor = transform(image_tensor, random_state, p=1, **kwargs) * valid_mask

                # Preserve these disabled draws: later augmentation RNG values
                # are part of the authoritative policy's numerical lineage.
                if random_state.random() < self.p_flip:
                    image_tensor = TF.hflip(image_tensor)
                    mask_tensor = TF.hflip(mask_tensor)
                if random_state.random() < self.p_flip:
                    image_tensor = TF.vflip(image_tensor)
                    mask_tensor = TF.vflip(mask_tensor)

                # Right-angle rotations are disabled but retain their RNG draw.
                if random_state.random() < self.p_rotate:
                    k = random_state.choice([1, 2, 3])  # 90, 180, or 270 degrees
                    image_tensor = torch.rot90(image_tensor, k=k, dims=(1, 2))
                    mask_tensor = torch.rot90(mask_tensor, k=k, dims=(1, 2))


                if random_state.random() < self.p_elastic:
                    image_tensor, mask_tensor = apply_elastic_transform(
                        image_tensor, mask_tensor, random_state, p=1,
                        torch_generator=torch_generator,
                    )

                # Horizontal scale: simulate different wave periods
                # Compression (< 1.0) = shorter periods, closer spacing
                # Stretch (> 1.0) = longer periods, wider spacing
                if random_state.random() < self.p_horizontal_scale:
                    image_tensor, mask_tensor = apply_horizontal_scale_transform(
                        image_tensor, mask_tensor, random_state, p=1, scale_range=(0.7, 1.4)
                    )

        image_tensor = self.normalize(image_tensor)

        # Binarize mask (threshold at 0.5 to handle interpolation artifacts)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor

    def __getitem__(self, idx):
        """
        Get image and mask pair

        Returns:
            image: torch.Tensor (3, H, W) - RGB image, normalized
            mask: torch.Tensor (1, H, W) - Binary mask {0, 1}
        """
        image_idx = idx // self.patches_per_image
        patch_idx = idx % self.patches_per_image

        sample = self.samples[image_idx]

        if self._cached_pairs is not None:
            image, mask = self._cached_pairs[image_idx]
        else:
            image_path = self.images_dir / sample['image_name']
            mask_path = self.masks_dir / sample['mask_name']
            image = Image.open(image_path)
            mask = Image.open(mask_path)

        return self._apply_synchronized_transform(image, mask, image_idx, patch_idx, idx)
