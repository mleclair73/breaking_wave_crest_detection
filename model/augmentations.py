"""Image and mask augmentations for wave-crest segmentation."""

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.transforms.v2.functional as transforms_v2_functional
from PIL import Image, ImageEnhance
from torchvision.transforms import InterpolationMode

# DataLoader already assigns one Torch thread per worker. Prevent OpenCV from
# starting another nested pool in every worker process.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

def _pair(value, cast):
    if isinstance(value, (int, float)):
        item = cast(value)
        return item, item
    values = tuple(cast(item) for item in value)
    if len(values) == 1:
        return values[0], values[0]
    if len(values) != 2:
        raise ValueError(f"Expected one or two values, got {value!r}")
    return values

def _gaussian_blur(image, kernel_size, sigma=None):
    """Fast CPU Gaussian blur matching Torchvision's padding and sigma rules.

    OpenCV's ``BORDER_REFLECT_101`` is the same reflection convention used by
    Torchvision. Unsupported tensors fall back to Torchvision so this helper is
    safe if an augmentation is later moved off CPU.
    """
    kernel_x, kernel_y = _pair(kernel_size, int)
    if kernel_x <= 0 or kernel_y <= 0 or kernel_x % 2 == 0 or kernel_y % 2 == 0:
        raise ValueError(f"Gaussian kernel sizes must be positive odd values: {kernel_size!r}")
    if sigma is None:
        sigma_x = kernel_x * 0.15 + 0.35
        sigma_y = kernel_y * 0.15 + 0.35
    else:
        sigma_x, sigma_y = _pair(sigma, float)

    if isinstance(image, torch.Tensor):
        if image.device.type != 'cpu' or image.dtype not in (torch.float32, torch.float64, torch.uint8):
            return TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)
        shape = image.shape
        if image.ndim < 2:
            raise ValueError(f"Expected an image tensor with at least two dimensions, got {shape}")
        height, width = shape[-2:]
        array = image.detach().contiguous().numpy().reshape(-1, height, width).transpose(1, 2, 0)
        blurred = cv2.GaussianBlur(
            array,
            ksize=(kernel_x, kernel_y),
            sigmaX=sigma_x,
            sigmaY=sigma_y,
            borderType=cv2.BORDER_REFLECT_101,
        )
        if blurred.ndim == 2:
            blurred = blurred[..., None]
        output = np.ascontiguousarray(
            blurred.transpose(2, 0, 1).reshape(*shape)
        )
        return torch.from_numpy(output).to(dtype=image.dtype)

    if isinstance(image, Image.Image):
        array = np.asarray(image)
        blurred = cv2.GaussianBlur(
            array,
            ksize=(kernel_x, kernel_y),
            sigmaX=sigma_x,
            sigmaY=sigma_y,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return Image.fromarray(blurred, mode=image.mode)

    return TF.gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

def _adjust_gamma(image, gamma, gain=1.0):
    """Fast CPU gamma correction with Torchvision-compatible clipping."""
    if (
        isinstance(image, torch.Tensor)
        and image.device.type == 'cpu'
        and image.dtype in (torch.float32, torch.float64)
        and image.ndim == 3
    ):
        array = image.detach().contiguous().numpy().transpose(1, 2, 0)
        corrected = cv2.pow(array, float(gamma))
        if corrected.ndim == 2:
            corrected = corrected[..., None]
        if gain != 1.0:
            corrected *= float(gain)
        np.clip(corrected, 0.0, 1.0, out=corrected)
        output = np.ascontiguousarray(corrected.transpose(2, 0, 1))
        return torch.from_numpy(output).to(dtype=image.dtype)
    return TF.adjust_gamma(image, gamma=gamma, gain=gain)

def _blur_unit_mask(mask, kernel_size):
    """Blur a [0, 1] CPU mask with the same uint8 path as PIL conversion."""
    if (
        isinstance(mask, torch.Tensor)
        and mask.device.type == 'cpu'
        and mask.dtype == torch.float32
        and mask.ndim in (2, 3)
    ):
        array = mask.detach().squeeze(0).mul(255).to(torch.uint8).numpy()
        kernel_x, kernel_y = _pair(kernel_size, int)
        blurred = cv2.GaussianBlur(
            array,
            ksize=(kernel_x, kernel_y),
            sigmaX=kernel_x * 0.15 + 0.35,
            sigmaY=kernel_y * 0.15 + 0.35,
            borderType=cv2.BORDER_REFLECT_101,
        )
        return torch.from_numpy(blurred).unsqueeze(0).float().div_(255)
    mask_pil = TF.to_pil_image(mask.squeeze(0))
    return TF.to_tensor(_gaussian_blur(mask_pil, kernel_size))

HEAVY_AUGMENTATION = {
    "p_augmentation": 0.95,
    "missing_data_prob": 0.50,
    "p_flip": 0.0,
    "p_rotate": 0.0,
    "p_brightness": 0.70,
    "p_gradient": 0.40,
    "p_glare": 0.20,
    "p_extreme_glare": 0.05,
    "p_water_droplet": 0.10,
    # Glare-negative augmentations teach that reflection artifacts are background.
    "p_specular_streak": 0.15,  # wakeless diagonal glint streak (swash/pier glare)
    "p_stationary_band": 0.12,  # near-horizontal band (shoreline foam line / pier)
    "p_glint_speckle": 0.10,    # incoherent offshore sun-glint sparkle
    "p_blur": 0.50,
    "p_motion_blur": 0.20,
    "p_elastic": 0.30,
    "p_local_gamma": 0.40,
    "p_horizontal_scale": 0.40,
    "p_patch_blur": 0.40,
    "p_random_lines": 0.40,
}

def apply_brightness_contrast(image, random_state, p=0.25):
    """Apply brightness/contrast adjustment with controlled randomness using ColorJitter"""
    if random_state.random() > p:
        return image

    brightness = random_state.uniform(0.5, 1.5)
    contrast = random_state.uniform(0.5, 1.5)

    image = TF.adjust_brightness(image, brightness)
    image = TF.adjust_contrast(image, contrast)

    gamma_factor = 1 + random_state.choice((-1, 1)) * random_state.uniform(0.08, 0.15)
    image = _adjust_gamma(image, gamma_factor)

    return image

def apply_glare_transform(image, random_state, p=0.25, extreme=False):
    """Blend broad, localized fields of specular reflection into valid pixels."""
    if random_state.random() > p:
        return image

    # Restrict glare to the valid image area; padding stays untouched.
    non_zero_mask = (image.sum(dim=0) > 0).float()
    non_zero_rows = torch.nonzero(non_zero_mask.sum(dim=1) > 0)
    non_zero_cols = torch.nonzero(non_zero_mask.sum(dim=0) > 0)

    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return image

    height, width = image.shape[1:3]
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    mask = torch.zeros((height, width), dtype=image.dtype)
    # These are deliberately broad enough to read as a reflection in a
    # 224-pixel training crop, while remaining local rather than recreating a
    # full-width exposure band.  Smooth texture prevents salt-and-pepper noise
    # from being mistaken for a useful glare cue.
    for _ in range(random_state.randint(2, 5)):
        cx = random_state.uniform(non_zero_cols[0].item(), non_zero_cols[-1].item())
        cy = random_state.uniform(non_zero_rows[0].item(), non_zero_rows[-1].item())
        rx = random_state.uniform(0.15, 0.35) * width
        ry = random_state.uniform(0.10, 0.24) * height
        lobe = torch.exp(-0.75 * (((x - cx) / max(rx, 1.0)) ** 2 + ((y - cy) / max(ry, 1.0)) ** 2))
        texture = torch.from_numpy(random_state.normal(1.0, 0.18, size=(height, width))).to(image.dtype)
        texture = 0.65 + 0.35 * _gaussian_blur(texture.unsqueeze(0), 21).squeeze(0)
        mask = mask + lobe * texture
    mask = torch.clamp(mask, 0, 0.95).unsqueeze(0) * non_zero_mask.unsqueeze(0)

    # Explicit reflection-field blending is visible even on already bright foam,
    # where multiplicative brightness changes would simply clip with little effect.
    strength = random_state.uniform(0.72, 0.90) if extreme else random_state.uniform(0.48, 0.68)
    reflection_level = random_state.uniform(0.92, 1.00)
    shimmer = torch.from_numpy(random_state.normal(0.0, 0.09, size=(height, width))).to(image.dtype)
    reflection = torch.clamp(reflection_level + shimmer, 0, 1).unsqueeze(0).repeat(3, 1, 1)
    alpha = torch.clamp(mask * strength, 0, 0.85)
    return image * (1 - alpha) + reflection * alpha

def apply_specular_streak_transform(image, random_state, p=0.25, extreme=False):
    """Inject thin, wakeless specular-glint streak(s) that stay labelled background.

    The model segments per-transect x-t timestacks, where sun glint or swash
    glare drifting shoreward becomes a bright diagonal streak like a propagating
    crest. The distinguishing physics is that a real crest
    carries a trailing whitewater wake and comes in a periodic train; specular
    glint is an isolated, wakeless bright line. This transform paints exactly
    that — a thin, isolated, wakeless diagonal streak — onto the image ONLY,
    leaving the mask as background, so the model learns 'bright streak lacking a
    foam wake != crest'. Deliberately omits any broadband trailing lobe.
    """
    if random_state.random() > p:
        return image

    # Confine the streak to the valid (non-padding) image area.
    non_zero_mask = (image.sum(dim=0) > 0).float()
    non_zero_rows = torch.nonzero(non_zero_mask.sum(dim=1) > 0)
    non_zero_cols = torch.nonzero(non_zero_mask.sum(dim=0) > 0)
    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
        return image

    height, width = image.shape[1:3]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    yy = yy.to(image.dtype)
    xx = xx.to(image.dtype)

    streak = torch.zeros((height, width), dtype=image.dtype)
    # 1-2 isolated streaks. Never a periodic comb (that would read as a real
    # wave train), which is why the count stays tiny.
    for _ in range(random_state.randint(1, 3)):
        cy = random_state.uniform(non_zero_rows[0].item(), non_zero_rows[-1].item())
        cx = random_state.uniform(non_zero_cols[0].item(), non_zero_cols[-1].item())
        # Diagonal like a propagating feature; avoid near-axis-aligned lines.
        angle = random_state.uniform(20.0, 70.0) * (np.pi / 180.0)
        angle *= random_state.choice((-1.0, 1.0))
        dx, dy = float(np.cos(angle)), float(np.sin(angle))
        perp = (xx - cx) * (-dy) + (yy - cy) * dx      # distance across the line
        along = (xx - cx) * dx + (yy - cy) * dy         # distance along the line
        line_width = random_state.uniform(0.8, 2.5)     # thin
        half_len = random_state.uniform(0.35, 0.60) * max(height, width)
        line = torch.exp(-0.5 * (perp / line_width) ** 2)
        # Smooth taper at the ends (no hard segment caps).
        edge = 0.15 * max(height, width) + 1e-6
        taper = torch.clamp(1.0 - torch.clamp(along.abs() - half_len, min=0.0) / edge, 0.0, 1.0)
        line = line * taper
        # Specular sparkle ALONG the streak — glint texture, not a foam wake.
        shimmer = torch.from_numpy(random_state.normal(1.0, 0.25, size=(height, width))).to(image.dtype)
        shimmer = 0.6 + 0.4 * _gaussian_blur(shimmer.unsqueeze(0), 5).squeeze(0)
        streak = torch.maximum(streak, line * shimmer)

    streak = torch.clamp(streak, 0, 1) * non_zero_mask
    # Bright near-specular reflection blended over the thin line only.
    strength = random_state.uniform(0.75, 0.95) if extreme else random_state.uniform(0.50, 0.80)
    reflection_level = random_state.uniform(0.90, 1.00)
    micro = torch.from_numpy(random_state.normal(0.0, 0.06, size=(height, width))).to(image.dtype)
    reflection = torch.clamp(reflection_level + micro, 0, 1).unsqueeze(0).repeat(3, 1, 1)
    alpha = torch.clamp(streak * strength, 0, 0.90).unsqueeze(0)
    return image * (1 - alpha) + reflection * alpha

def apply_stationary_band_transform(image, random_state, p=0.25):
    """Inject a near-horizontal stationary band (zero cross-shore celerity) that
    stays labelled background.

    Companion to apply_specular_streak_transform. In an x-t timestack a fixed
    structure (the persistent shoreline foam line, or the FRF pier deck) sits at
    constant cross-shore across time -> a near-horizontal band, whereas a real
    breaking crest always has a finite shoreward slope. Painting such a band
    (bright foam line, or occasionally a dark pier shadow) onto the image only,
    with the mask left background, teaches 'zero-slope band != crest'. Slope is
    kept small so it never mimics a steep real crest.
    """
    if random_state.random() > p:
        return image
    non_zero_mask = (image.sum(dim=0) > 0).float()
    rows = torch.nonzero(non_zero_mask.sum(dim=1) > 0)
    cols = torch.nonzero(non_zero_mask.sum(dim=0) > 0)
    if len(rows) == 0 or len(cols) == 0:
        return image
    height, width = image.shape[1:3]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    yy = yy.to(image.dtype); xx = xx.to(image.dtype)
    band = torch.zeros((height, width), dtype=image.dtype)
    for _ in range(random_state.randint(1, 3)):
        cy = random_state.uniform(rows[0].item(), rows[-1].item())
        slope = random_state.uniform(-0.06, 0.06)          # <= ~3.4 deg: near-horizontal
        thick = random_state.uniform(1.5, 4.0)
        dist = yy - (cy + slope * (xx - width / 2))
        line = torch.exp(-0.5 * (dist / thick) ** 2)
        cx = random_state.uniform(cols[0].item(), cols[-1].item())
        half = random_state.uniform(0.4, 0.6) * width
        taper = torch.clamp(1.0 - torch.clamp((xx - cx).abs() - half, min=0.0) / (0.15 * width + 1e-6), 0.0, 1.0)
        line = line * taper
        shimmer = torch.from_numpy(random_state.normal(1.0, 0.20, size=(height, width))).to(image.dtype)
        shimmer = 0.6 + 0.4 * _gaussian_blur(shimmer.unsqueeze(0), 5).squeeze(0)
        band = torch.maximum(band, line * shimmer)
    band = torch.clamp(band, 0, 1) * non_zero_mask
    dark = random_state.random() < 0.30                     # 30% dark pier shadow, else bright foam
    level = random_state.uniform(0.0, 0.12) if dark else random_state.uniform(0.90, 1.00)
    strength = random_state.uniform(0.55, 0.85)
    micro = torch.from_numpy(random_state.normal(0.0, 0.06, size=(height, width))).to(image.dtype)
    fill = torch.clamp(level + micro, 0, 1).unsqueeze(0).repeat(3, 1, 1)
    alpha = torch.clamp(band * strength, 0, 0.90).unsqueeze(0)
    return image * (1 - alpha) + fill * alpha

def apply_glint_speckle_transform(image, random_state, p=0.25):
    """Inject an incoherent specular-glint speckle field (background label).

    Scattered point-like sparkle inside a soft envelope, mimicking sun glint on
    open water offshore of the surf zone. Unlike a crest it forms no coherent
    propagating streak, so labelling it background discourages the offshore
    glint false positives without touching real crests.
    """
    if random_state.random() > p:
        return image
    non_zero_mask = (image.sum(dim=0) > 0).float()
    rows = torch.nonzero(non_zero_mask.sum(dim=1) > 0)
    cols = torch.nonzero(non_zero_mask.sum(dim=0) > 0)
    if len(rows) == 0 or len(cols) == 0:
        return image
    height, width = image.shape[1:3]
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    yy = yy.to(image.dtype); xx = xx.to(image.dtype)
    ecx = random_state.uniform(cols[0].item(), cols[-1].item())
    ecy = random_state.uniform(rows[0].item(), rows[-1].item())
    erx = random_state.uniform(0.25, 0.5) * width
    ery = random_state.uniform(0.25, 0.5) * height
    env = torch.exp(-0.5 * (((xx - ecx) / erx) ** 2 + ((yy - ecy) / ery) ** 2))
    seed = torch.zeros((height, width), dtype=image.dtype)
    n = random_state.randint(60, 220)
    ys = random_state.randint(0, height, size=n)
    xs = random_state.randint(0, width, size=n)
    seed[ys, xs] = torch.from_numpy(random_state.uniform(0.6, 1.0, size=n)).to(image.dtype)
    field = _gaussian_blur(seed.unsqueeze(0), 3).squeeze(0)
    field = field / (field.max() + 1e-6)
    field = field * env * non_zero_mask
    level = random_state.uniform(0.90, 1.00)
    strength = random_state.uniform(0.60, 0.90)
    micro = torch.from_numpy(random_state.normal(0.0, 0.05, size=(height, width))).to(image.dtype)
    reflection = torch.clamp(level + micro, 0, 1).unsqueeze(0).repeat(3, 1, 1)
    alpha = torch.clamp(field * strength, 0, 0.90).unsqueeze(0)
    return image * (1 - alpha) + reflection * alpha

def add_water_droplets(image, random_state, p=0.3, num_droplets=None, thickness_range=(3, 15)):
    """Add water droplets as full-width horizontal dark lines that blend at the edges."""
    if random_state.random() > p:
        return image

    result = image.clone()
    height, width = image.shape[1:3]

    non_zero_rows = torch.where(image.sum(dim=[0, 2]) > 0)[0]
    if len(non_zero_rows) == 0:
        return image

    if num_droplets is None:
        num_droplets = random_state.randint(1, 6)

    for _ in range(num_droplets):
        if len(non_zero_rows) == 0:
            continue

        idx = random_state.randint(0, len(non_zero_rows))
        y_center = non_zero_rows[idx].item()

        thickness = random_state.randint(thickness_range[0], thickness_range[1] + 1)

        y_min, y_max = max(0, y_center - thickness//2), min(height, y_center + thickness//2 + 1)

        mask = torch.ones((y_max - y_min, width))
        darkness = random_state.uniform(0.1, 0.3)

        if thickness > 1:
            edge_fade = torch.linspace(0.5, 1.0, thickness//2 + 1)**4
            if len(edge_fade) > 1:
                if y_min > 0:
                    mask[0:len(edge_fade), :] *= edge_fade.unsqueeze(1)
                if y_max < height:
                    mask[-len(edge_fade):, :] *= torch.flip(edge_fade, [0]).unsqueeze(1)

        for c in range(3):
            result[c, y_min:y_max, :] = result[c, y_min:y_max, :] * (1 - (1 - darkness) * mask)

    return result

def apply_vertical_gradient_transform(image, random_state, p=0.25, blur_radius=5):
    """Apply vertical gradient with brightness/contrast/gamma adjustment"""
    if random_state.random() > p:
        return image

    blur_radius = 2 * random_state.randint(2, 5) + 1

    height, width = image.shape[1:3]
    y_split = random_state.randint(height // 8, 7 * height // 8)

    mask = torch.zeros((1, height, width))
    mask[0, :y_split, :] = 1

    if random_state.random() < 0.5:
        mask = 1 - mask

    # Adapt blur radius to image size to avoid padding errors
    max_blur = min(height, width)
    if max_blur < 3:  # Too small for any blur
        mask_blurred = mask
    else:
        adaptive_blur = min(blur_radius, max_blur)
        if adaptive_blur % 2 == 0:
            adaptive_blur -= 1
        adaptive_blur = max(3, adaptive_blur)
        mask_blurred = _blur_unit_mask(mask, adaptive_blur)

    brightness_factor = 1 + random_state.choice((-1, 1)) * random_state.uniform(0.15, 0.4)
    contrast_factor = 1 + random_state.choice((-1, 1)) * random_state.uniform(0, 0.5)
    gamma_factor = 1 + random_state.choice((-1, 1)) * random_state.uniform(0, 0.3)

    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(transformed_image, brightness_factor)
    transformed_image = TF.adjust_contrast(transformed_image, contrast_factor)
    transformed_image = _adjust_gamma(transformed_image, gamma_factor)

    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred

    return blended_image

def apply_angled_gradient_transform(image, random_state, p=0.25, blur_radius=51):
    """Apply angled exposure gradient transform"""
    if random_state.random() > p:
        return image

    height, width = image.shape[1:3]

    point_x = random_state.uniform(width // 4, 3 * width // 4)
    point_y = random_state.uniform(height // 4, 3 * height // 4)
    angle = random_state.choice((-1, 1)) * random_state.uniform(35, 55)

    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")

    x = x - point_x
    y = y - point_y

    angle_rad = math.radians(angle)
    x_rot = x * math.cos(angle_rad) + y * math.sin(angle_rad)

    mask = x_rot.unsqueeze(0)
    mask = torch.where(
        ((1 if random_state.random() < 0.5 else -1) * mask) < 0,
        torch.zeros_like(mask),
        torch.ones_like(mask),
    )

    # Adapt blur radius to image size to avoid padding errors
    max_blur = min(height, width)
    if max_blur < 3:  # Too small for any blur
        mask_blurred = mask
    else:
        adaptive_blur = min(blur_radius, max_blur)
        if adaptive_blur % 2 == 0:
            adaptive_blur -= 1
        adaptive_blur = max(3, adaptive_blur)
        mask_blurred = _blur_unit_mask(mask, adaptive_blur)

    transformed_image = image.clone()
    transformed_image = TF.adjust_brightness(
        transformed_image, random_state.uniform(0.8, 1.2)
    )
    transformed_image = TF.adjust_contrast(
        transformed_image, random_state.uniform(0.8, 1.2)
    )
    transformed_image = _adjust_gamma(
        transformed_image, random_state.uniform(0.85, 1.15)
    )

    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred

    return blended_image

def apply_local_gamma_transform(image, random_state, p=0.25, blur_radius=31):
    """
    Apply gamma correction to a random region of the image.

    Simulates local lighting/exposure variations that occur in real imagery.
    The region can be rectangular, elliptical, or corner-based.

    Args:
        image: Input image tensor (C, H, W)
        random_state: Random state for reproducibility
        p: Probability of applying the transform
        blur_radius: Blur radius for smooth blending at edges
    """
    if random_state.random() > p:
        return image

    height, width = image.shape[1:3]

    mask = torch.zeros((1, height, width))

    # Choose region type: 0=rectangle, 1=ellipse, 2=corner, 3=horizontal band
    region_type = random_state.choice([0, 1, 2, 3], p=[0.3, 0.3, 0.2, 0.2])

    if region_type == 0:  # Rectangle
        # Random rectangle covering 20-60% of image
        rect_w = int(random_state.uniform(0.3, 0.7) * width)
        rect_h = int(random_state.uniform(0.3, 0.7) * height)
        x_start = random_state.randint(0, max(1, width - rect_w))
        y_start = random_state.randint(0, max(1, height - rect_h))
        mask[0, y_start:y_start+rect_h, x_start:x_start+rect_w] = 1

    elif region_type == 1:  # Ellipse
        # Random ellipse
        center_x = random_state.randint(width // 4, 3 * width // 4)
        center_y = random_state.randint(height // 4, 3 * height // 4)
        radius_x = int(random_state.uniform(0.2, 0.5) * width)
        radius_y = int(random_state.uniform(0.2, 0.5) * height)

        y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
        ellipse = ((x - center_x) / max(radius_x, 1)) ** 2 + ((y - center_y) / max(radius_y, 1)) ** 2
        mask[0] = (ellipse <= 1).float()

    elif region_type == 2:  # Corner
        # Random corner region
        corner = random_state.choice(['tl', 'tr', 'bl', 'br'])
        corner_w = int(random_state.uniform(0.3, 0.6) * width)
        corner_h = int(random_state.uniform(0.3, 0.6) * height)

        if corner == 'tl':
            mask[0, :corner_h, :corner_w] = 1
        elif corner == 'tr':
            mask[0, :corner_h, width-corner_w:] = 1
        elif corner == 'bl':
            mask[0, height-corner_h:, :corner_w] = 1
        else:  # br
            mask[0, height-corner_h:, width-corner_w:] = 1

    else:  # Horizontal band
        # Random horizontal band (like sun reflection)
        band_h = int(random_state.uniform(0.15, 0.4) * height)
        y_start = random_state.randint(0, max(1, height - band_h))
        mask[0, y_start:y_start+band_h, :] = 1

    max_blur = min(height, width)
    if max_blur < 3:
        mask_blurred = mask
    else:
        adaptive_blur = min(blur_radius, max_blur)
        if adaptive_blur % 2 == 0:
            adaptive_blur -= 1
        adaptive_blur = max(3, adaptive_blur)
        mask_blurred = _blur_unit_mask(mask, adaptive_blur)

    # A local tone curve simulates both gamma changes and luminance saturation
    # (white/black-point clipping). The source imagery is grayscale replicated
    # into RGB, so colour-saturation jitter would be a no-op here.
    gamma = random_state.uniform(0.65, 1.45)
    transformed_image = _adjust_gamma(image.clone(), gamma)
    if random_state.random() < 0.65:
        if random_state.random() < 0.5:
            black_point = random_state.uniform(0.02, 0.16)
            transformed_image = torch.clamp(
                (transformed_image - black_point) / (1 - black_point), 0, 1
            )
        else:
            white_point = random_state.uniform(0.72, 0.94)
            transformed_image = torch.clamp(transformed_image / white_point, 0, 1)

    # Blend original and transformed based on mask
    mask_blurred = mask_blurred.repeat(3, 1, 1)
    blended_image = image * (1 - mask_blurred) + transformed_image * mask_blurred

    return blended_image

def apply_missing_data_transform(image, random_state, p=0.25):
    """
    Apply vertical bar dropout to simulate missing timestack frames.

    The authoritative policy chooses equally between patterned bars and random
    complete-frame dropout.

    Args:
        image: Input image tensor (C, H, W)
        random_state: Random state for reproducibility
        p: Probability of applying the transform
    """
    if random_state.random() > p:
        return image

    height, width = image.shape[1:3]

    mask = torch.ones_like(image)

    mode = random_state.choice(['pattern', 'random_frames'])

    if mode == 'random_frames':
        # Random frame dropout: mask 2-10 complete vertical slices
        num_frames = random_state.randint(2, 11)  # 2 to 10 frames

        # Randomly select frame positions (avoid duplicates)
        available_positions = list(range(width))
        frame_positions = random_state.choice(available_positions, size=num_frames, replace=False)

        # Mask the selected frames (full height)
        for x in frame_positions:
            mask[:, :, x] = 0

    else:  # mode == 'pattern'
        # Original pattern-based approach
        # Number of different bar patterns to apply (1-3), biased towards fewer patterns
        # 60% chance of 1 pattern, 30% chance of 2 patterns, 10% chance of 3 patterns
        num_patterns = random_state.choice([1, 2, 3], p=[0.6, 0.3, 0.1])

        for pattern_idx in range(num_patterns):
            # Bar width: 90% chance of 1 column, 10% chance of 2-3 columns
            if random_state.random() < 0.8:
                bar_width = 1
            else:
                bar_width = random_state.randint(2, 4)  # 2 or 3

            # Frequency: spacing between bars (2-8 columns between bar starts)
            frequency = random_state.randint(2*bar_width, 9)

            # First pattern: ensure it covers at least 50% of the image
            # Other patterns: random extent
            if pattern_idx == 0:
                # Guarantee at least 50% coverage in at least one dimension
                if random_state.random() < 0.5:
                    # Keep the horizontal extent at least 50%.
                    x_extent = random_state.uniform(0.5, 1.0)
                    x_start = random_state.randint(0, max(1, int(width * (1 - x_extent))))
                    x_end = int(x_start + width * x_extent)

                    # Vertical can be random
                    if random_state.random() < 0.5:
                        y_extent = random_state.uniform(0.3, 0.8)
                        y_start = random_state.randint(0, int(height * (1 - y_extent)))
                        y_end = int(y_start + height * y_extent)
                    else:
                        y_start = 0
                        y_end = height
                else:
                    # Keep the vertical extent at least 50%.
                    y_extent = random_state.uniform(0.5, 1.0)
                    y_start = random_state.randint(0, max(1, int(height * (1 - y_extent))))
                    y_end = int(y_start + height * y_extent)

                    # Horizontal can be random
                    if random_state.random() < 0.3:
                        x_extent = random_state.uniform(0.3, 0.8)
                        x_start = random_state.randint(0, int(width * (1 - x_extent)))
                        x_end = int(x_start + width * x_extent)
                    else:
                        x_start = 0
                        x_end = width
            else:
                # Subsequent patterns: random extent
                # Random horizontal extent (which x region to apply bars to)
                if random_state.random() < 0.3:
                    # Partial horizontal extent (30% chance)
                    x_extent = random_state.uniform(0.3, 0.8)  # 30-80% of width
                    x_start = random_state.randint(0, int(width * (1 - x_extent)))
                    x_end = int(x_start + width * x_extent)
                else:
                    # Full width (70% chance)
                    x_start = 0
                    x_end = width

                # Random vertical extent (which y region to apply bars to)
                if random_state.random() < 0.5:
                    # Partial vertical extent (50% chance)
                    y_extent = random_state.uniform(0.3, 0.8)  # 30-80% of height
                    y_start = random_state.randint(0, int(height * (1 - y_extent)))
                    y_end = int(y_start + height * y_extent)
                else:
                    # Full height (50% chance)
                    y_start = 0
                    y_end = height

            # Random offset for bar pattern start position
            offset = random_state.randint(0, frequency)

            x = x_start + offset
            while x < x_end:
                # Keep the bar within the image width.
                actual_bar_width = min(bar_width, x_end - x)
                if actual_bar_width > 0:
                    mask[:, y_start:y_end, x:x + actual_bar_width] = 0
                x += frequency

    # Feather only the vertical (cross-shore) edges - where bars terminate inside
    # the frame - to a varying strength (sometimes none). torchvision kernel_size
    # is (kx, ky), so kx=1 leaves the along-time edges absolutely sharp.
    ky = int(random_state.choice([1, 5, 11, 21, 31]))
    if ky > 1:
        ky = min(ky, height - 1 if height % 2 == 0 else height)
        if ky % 2 == 0:
            ky += 1
        if ky > 1:
            mask = _gaussian_blur(mask, kernel_size=(1, ky))

    return image * mask

def add_gaussian_noise(
    image, noise_std=0.01, torch_generator=None,
    numpy_generator=None,
):
    """Add gaussian noise to image tensor"""
    if (
        numpy_generator is not None
        and image.device.type == 'cpu'
        and image.dtype in (torch.float32, torch.float64)
    ):
        numpy_dtype = np.float32 if image.dtype == torch.float32 else np.float64
        noise = torch.from_numpy(
            numpy_generator.standard_normal(image.shape, dtype=numpy_dtype)
        )
    else:
        noise = torch.randn(
            image.shape,
            dtype=image.dtype,
            device=image.device,
            generator=torch_generator,
        )
    noise = noise * noise_std
    return torch.clamp(image + noise, 0, 1)

def apply_gaussian_blur(image, random_state, p=0.5):
    """Apply slight gaussian blur"""
    if random_state.random() > p:
        return image

    kernel_size = int(random_state.choice([3, 5, 7]))
    sigma = float(random_state.uniform(0.5, 1.5))

    return _gaussian_blur(image, kernel_size=kernel_size, sigma=sigma)

def apply_directional_motion_blur(image, random_state, p=0.2):
    """Apply a short horizontal, vertical, or diagonal motion blur."""
    if random_state.random() > p:
        return image

    _, height, width = image.shape
    half_length = int(random_state.randint(1, 5))  # 3-9 pixel kernel
    kernel_size = 2 * half_length + 1
    angle = math.radians(float(random_state.choice([0, 90, 35, -35, 55, -55])))
    yy, xx = torch.meshgrid(
        torch.arange(-half_length, half_length + 1, dtype=image.dtype),
        torch.arange(-half_length, half_length + 1, dtype=image.dtype),
        indexing='ij',
    )
    along = xx * math.cos(angle) + yy * math.sin(angle)
    across = -xx * math.sin(angle) + yy * math.cos(angle)
    kernel = torch.exp(-0.5 * (across / 0.55) ** 2) * (along.abs() <= half_length)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(image.shape[0], 1, 1, 1)

    # Replication avoids dark seams at the crop boundary. Padding pixels remain
    # zero after the caller reapplies its valid-pixel mask.
    padded = F.pad(image.unsqueeze(0), (half_length,) * 4, mode='replicate')
    return F.conv2d(padded, kernel, groups=image.shape[0]).squeeze(0)

def apply_patch_gaussian_blur(image, random_state, p=0.3, num_patches=None):
    """Apply gaussian blur to random rectangular patches of the image.

    Simulates localized focus issues or lens artifacts where only part
    of the image is blurred while the rest stays sharp.

    Args:
        image: Input image tensor (C, H, W)
        random_state: Random state for reproducibility
        p: Probability of applying the transform
        num_patches: Number of blur patches (default: random 1-3)
    """
    if random_state.random() > p:
        return image

    result = image.clone()
    _, height, width = image.shape

    if num_patches is None:
        num_patches = random_state.randint(1, 2)  # 1-3 patches

    for _ in range(num_patches):
        # Random patch size: 15-50% of each dimension
        patch_h = int(random_state.uniform(0.15, 0.3) * height)
        patch_w = int(random_state.uniform(0.15, 0.3) * width)
        y_start = random_state.randint(0, max(1, height - patch_h))
        x_start = random_state.randint(0, max(1, width - patch_w))

        patch = result[:, y_start:y_start+patch_h, x_start:x_start+patch_w]

        # Random kernel size and sigma
        kernel_size = int(random_state.choice([5, 7, 9, 11]))
        sigma = float(random_state.uniform(0.5, 2.0))

        if patch_h >= kernel_size and patch_w >= kernel_size:
            blurred_patch = _gaussian_blur(patch, kernel_size=kernel_size, sigma=sigma)

            # Blend with soft edges (fade over ~10% of patch dims)
            fade_h = max(2, patch_h // 10)
            fade_w = max(2, patch_w // 10)
            blend = torch.ones(1, patch_h, patch_w)
            # Vertical fade
            blend[0, :fade_h, :] *= torch.linspace(0, 1, fade_h).unsqueeze(1)
            blend[0, -fade_h:, :] *= torch.linspace(1, 0, fade_h).unsqueeze(1)
            # Horizontal fade
            blend[0, :, :fade_w] *= torch.linspace(0, 1, fade_w).unsqueeze(0)
            blend[0, :, -fade_w:] *= torch.linspace(1, 0, fade_w).unsqueeze(0)

            result[:, y_start:y_start+patch_h, x_start:x_start+patch_w] = (
                patch * (1 - blend) + blurred_patch * blend
            )

    return result

def apply_random_lines(image, random_state, p=0.3, num_lines=None):
    """Blend irregular nuisance streak texture without changing the target.

    These are low-to-moderate contrast horizontal and vertical streaks such as
    rain/sensor texture. They intentionally differ from
    ``apply_missing_data_transform``: that transform remains responsible for
    the known production frame-dropout artifact.
    """
    if random_state.random() > p:
        return image

    result = image.clone()
    _, height, width = image.shape

    if num_lines is None:
        num_lines = random_state.randint(4, 11)  # 4-10 correlated streaks

    for _ in range(num_lines):
        thickness = random_state.randint(1, 4)  # 1-3 pixels
        value = random_state.uniform(0.0, 0.22) if random_state.random() < 0.7 else random_state.uniform(0.78, 1.0)
        opacity = random_state.uniform(0.12, 0.42)
        is_vertical = random_state.random() < 0.55
        if is_vertical:
            x0 = random_state.randint(0, width)
            y0 = random_state.randint(0, max(1, height // 3))
            length = random_state.randint(max(2, height // 3), height + 1)
            drift = random_state.randint(-max(1, width // 20), max(2, width // 20 + 1))
            # Faint, slightly drifting vertical streak with broken coverage.
            ys = []
            xs = []
            local_opacities = []
            for offset in range(length):
                y = y0 + offset
                if y >= height or random_state.random() < 0.12:
                    continue
                x = int(round(x0 + drift * (offset / max(length - 1, 1))))
                x_lo, x_hi = max(0, x - thickness // 2), min(width, x + (thickness + 1) // 2)
                if x_lo < x_hi:
                    local_opacity = opacity * random_state.uniform(0.65, 1.15)
                    ys.append(y)
                    xs.append(x)
                    local_opacities.append(local_opacity)
            if ys:
                y_indices = torch.tensor(ys, dtype=torch.long, device=result.device)
                x_centers = torch.tensor(xs, dtype=torch.long, device=result.device)
                alpha = torch.tensor(
                    local_opacities, dtype=result.dtype, device=result.device,
                ).unsqueeze(0)
                for delta in range(-(thickness // 2), (thickness + 1) // 2):
                    x_indices = x_centers + delta
                    valid = (x_indices >= 0) & (x_indices < width)
                    region = result[:, y_indices[valid], x_indices[valid]]
                    local_alpha = alpha[:, valid]
                    result[:, y_indices[valid], x_indices[valid]] = (
                        region * (1 - local_alpha) + value * local_alpha
                    )
        else:
            y = random_state.randint(0, height)
            x0 = random_state.randint(0, max(1, width // 3))
            length = random_state.randint(max(2, width // 3), width + 1)
            # Long horizontal interference bands, also with intermittent gaps.
            xs = []
            local_opacities = []
            for offset in range(length):
                x = x0 + offset
                if x >= width or random_state.random() < 0.12:
                    continue
                y_lo, y_hi = max(0, y - thickness // 2), min(height, y + (thickness + 1) // 2)
                if y_lo < y_hi:
                    local_opacity = opacity * random_state.uniform(0.65, 1.15)
                    xs.append(x)
                    local_opacities.append(local_opacity)
            if xs:
                x_indices = torch.tensor(xs, dtype=torch.long, device=result.device)
                alpha = torch.tensor(
                    local_opacities, dtype=result.dtype, device=result.device,
                ).unsqueeze(0)
                for delta in range(-(thickness // 2), (thickness + 1) // 2):
                    y_index = y + delta
                    if 0 <= y_index < height:
                        region = result[:, y_index, x_indices]
                        result[:, y_index, x_indices] = (
                            region * (1 - alpha) + value * alpha
                        )

    return result

def apply_sharpen(image, random_state, p=0.5):
    """Apply sharpening to image"""
    if random_state.random() > p:
        return image

    image_pil = TF.to_pil_image(image)
    enhancer = ImageEnhance.Sharpness(image_pil)
    factor = random_state.uniform(1.3, 2.0)
    sharpened = enhancer.enhance(factor)

    return TF.to_tensor(sharpened)

def apply_elastic_transform(image, mask, random_state, p=0.5, torch_generator=None):
    """Apply minimal elastic transform to both image and mask (< 10 pixels displacement)"""
    if random_state.random() > p:
        return image, mask

    # Very small alpha (displacement magnitude) - max 8 pixels
    alpha = random_state.uniform(3, 8)
    # Larger sigma (smoothness) for more natural deformation
    sigma = random_state.uniform(3, 5)

    height, width = image.shape[-2:]
    kernel_size = int(8 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Match Torchvision ElasticTransform's displacement construction, but use
    # the optimized CPU blur and an explicit per-sample Torch generator.
    dx = torch.rand(
        (1, 1, height, width), dtype=image.dtype, device=image.device,
        generator=torch_generator,
    ) * 2 - 1
    dx = _gaussian_blur(dx, [kernel_size, kernel_size], [sigma, sigma])
    dx = dx * alpha / width
    dy = torch.rand(
        (1, 1, height, width), dtype=image.dtype, device=image.device,
        generator=torch_generator,
    ) * 2 - 1
    dy = _gaussian_blur(dy, [kernel_size, kernel_size], [sigma, sigma])
    dy = dy * alpha / height
    displacement = torch.cat([dx, dy], dim=1).permute(0, 2, 3, 1)

    stacked = torch.cat([image, mask], dim=0)  # (4, H, W)
    transformed = transforms_v2_functional.elastic(
        stacked,
        displacement=displacement,
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )

    image = transformed[:3]
    mask = transformed[3:4]

    return image, mask

def apply_horizontal_scale_transform(image, mask, random_state, p=0.3, scale_range=(0.5, 1.5)):
    """Apply horizontal scaling to simulate different wave periods.

    Horizontal compression (scale < 1.0) simulates shorter wave periods (closer spacing).
    Horizontal stretch (scale > 1.0) simulates longer wave periods (wider spacing).

    Args:
        image: torch.Tensor (C, H, W)
        mask: torch.Tensor (1, H, W)
        random_state: numpy RandomState for reproducibility
        p: probability of applying transform
        scale_range: (min_scale, max_scale) for horizontal scaling

    Returns:
        Transformed image and mask tensors
    """
    if random_state.random() > p:
        return image, mask

    _, H, W = image.shape

    h_scale = random_state.uniform(scale_range[0], scale_range[1])

    new_W = int(W * h_scale)

    # Resize with horizontal scaling only
    # Use interpolate for image (bilinear) and mask (nearest to preserve binary values)
    image_scaled = torch.nn.functional.interpolate(
        image.unsqueeze(0), size=(H, new_W), mode='bilinear', align_corners=False
    ).squeeze(0)
    mask_scaled = torch.nn.functional.interpolate(
        mask.unsqueeze(0), size=(H, new_W), mode='nearest'
    ).squeeze(0)

    # Crop or pad to original width
    if new_W > W:
        # Crop from center
        start = (new_W - W) // 2
        image_scaled = image_scaled[:, :, start:start + W]
        mask_scaled = mask_scaled[:, :, start:start + W]
    elif new_W < W:
        # Pad with zeros on both sides
        pad_left = (W - new_W) // 2
        pad_right = W - new_W - pad_left
        image_scaled = torch.nn.functional.pad(image_scaled, (pad_left, pad_right), mode='constant', value=0)
        mask_scaled = torch.nn.functional.pad(mask_scaled, (pad_left, pad_right), mode='constant', value=0)

    return image_scaled, mask_scaled
