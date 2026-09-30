"""SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

Standalone PyTorch implementation based on:
https://github.com/Visual-Attention-Network/SegNeXt

This module retains only SegNeXt-T, the upstream LightHamHead, and
the parameterized learned-up decoder used by the nested skip ablation.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DWConv(nn.Module):
    """Depthwise Convolution"""
    def __init__(self, dim=768, padding_mode='zeros'):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim, padding_mode=padding_mode)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    """MLP with depthwise convolution"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., padding_mode='zeros'):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features, padding_mode=padding_mode)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    """Stem convolution: 4x downsampling with two 3x3 convs"""
    def __init__(
        self, in_channels, out_channels, padding_mode='zeros'
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=3, stride=2,
                padding=1, padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(
                out_channels // 2, out_channels, kernel_size=3, stride=2,
                padding=1, padding_mode=padding_mode,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W



class SafePadConv2d(nn.Module):
    """Conv2d with manual padding that falls back to 'zeros' when the input
    spatial dimension is too small for the requested padding mode (e.g.
    'reflect' requires pad < input size)."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=(0, 0),
                 groups=1, padding_mode='zeros'):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=0, groups=groups)
        self.padding = padding  # (pad_h, pad_w)
        self.padding_mode = padding_mode

    def forward(self, x):
        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            return self.conv(x)
        _, _, H, W = x.shape
        # F.pad expects (left, right, top, bottom)
        # reflect/replicate require pad < corresponding dim
        mode = self.padding_mode
        if mode != 'zeros' and (pad_h >= H or pad_w >= W):
            mode = 'zeros'
        if mode == 'zeros':
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        else:
            x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode=mode)
        return self.conv(x)


class AttentionModule(nn.Module):
    """Multi-scale convolutional attention with multiple kernel sizes"""
    def __init__(self, dim, padding_mode='zeros'):
        super().__init__()
        self.conv0 = SafePadConv2d(dim, dim, 5, padding=(2, 2), groups=dim, padding_mode=padding_mode)
        self.conv0_1 = SafePadConv2d(dim, dim, (1, 7), padding=(0, 3), groups=dim, padding_mode=padding_mode)
        self.conv0_2 = SafePadConv2d(dim, dim, (7, 1), padding=(3, 0), groups=dim, padding_mode=padding_mode)
        self.conv1_1 = SafePadConv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim, padding_mode=padding_mode)
        self.conv1_2 = SafePadConv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim, padding_mode=padding_mode)
        self.conv2_1 = SafePadConv2d(dim, dim, (1, 21), padding=(0, 10), groups=dim, padding_mode=padding_mode)
        self.conv2_2 = SafePadConv2d(dim, dim, (21, 1), padding=(10, 0), groups=dim, padding_mode=padding_mode)
        self.conv3 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)

        attn_0 = self.conv0_1(attn)
        attn_0 = self.conv0_2(attn_0)

        attn_1 = self.conv1_1(attn)
        attn_1 = self.conv1_2(attn_1)

        attn_2 = self.conv2_1(attn)
        attn_2 = self.conv2_2(attn_2)

        attn = attn + attn_0 + attn_1 + attn_2
        attn = self.conv3(attn)

        return attn * u


class SpatialAttention(nn.Module):
    """Spatial attention with projection and gating"""
    def __init__(self, d_model, padding_mode='zeros'):
        super().__init__()
        self.d_model = d_model
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = AttentionModule(d_model, padding_mode=padding_mode)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    """SegNeXt block with spatial attention and MLP"""
    def __init__(
        self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU,
        padding_mode='zeros',
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, padding_mode=padding_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, padding_mode=padding_mode)

        # Layer scale
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding for downsampling"""
    def __init__(
        self, patch_size=7, stride=4, in_chans=3, embed_dim=768,
        padding_mode='zeros',
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=patch_size // 2, padding_mode=padding_mode,
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class MSCAN(nn.Module):
    """Multi-Scale Convolutional Attention Network (MSCAN) backbone

    Args:
        in_chans (int): Number of input channels. Default: 3
        embed_dims (list): Embedding dimensions for each stage
        mlp_ratios (list): MLP expansion ratios
        drop_path_rate (float): Stochastic depth rate
        depths (list): Number of blocks in each stage
        num_stages (int): Number of stages
    """
    def __init__(self,
                 in_chans=3,
                 embed_dims=[32, 64, 160, 256],
                 mlp_ratios=[8, 8, 4, 4],
                 drop_path_rate=0.1,
                 depths=[3, 3, 5, 2],
                 num_stages=4,
                 padding_mode='zeros'):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.embed_dims = embed_dims

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(
                    in_chans, embed_dims[0], padding_mode=padding_mode,
                )
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3,
                    stride=2,
                    in_chans=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                    padding_mode=padding_mode,
                )

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=0.0,
                    drop_path=dpr[cur + j],
                    padding_mode=padding_mode,
                )
                for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, mean=0, std=math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


class NMF2D(nn.Module):
    """Faithful, dependency-free port of SegNeXt's upstream NMF2D.

    There is no whole-tensor normalization, coefficient/basis clamping, dtype
    promotion, or NaN fallback. The equations and random-basis behavior match
    upstream SegNeXt. Device-aware basis creation replaces upstream's hard-coded
    CUDA allocation.
    """

    def __init__(self, ham_kwargs=None):
        super().__init__()
        args = dict(ham_kwargs or {})
        self.S = int(args.get('MD_S', 1))
        self.R = int(args.get('MD_R', 64))
        self.train_steps = int(args.get('TRAIN_STEPS', 6))
        self.eval_steps = int(args.get('EVAL_STEPS', 7))
        # NMF2D overrides the generic matrix-decomposition temperature to one.
        self.inv_t = 1.0

    @staticmethod
    def _build_bases(B, S, D, R, device, dtype):
        bases = torch.rand((B * S, D, R), device=device, dtype=dtype)
        return F.normalize(bases, dim=1)

    @staticmethod
    def local_step(x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        coef = coef * numerator / (denominator + 1e-6)

        numerator = torch.bmm(x, coef)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        bases = bases * numerator / (denominator + 1e-6)
        return bases, coef

    @staticmethod
    def compute_coef(x, bases, coef):
        numerator = torch.bmm(x.transpose(1, 2), bases)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        return coef * numerator / (denominator + 1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        D = C // self.S
        N = H * W
        x_flat = x.view(B * self.S, D, N)
        bases = self._build_bases(B, self.S, D, self.R, x.device, x.dtype)

        coef = torch.bmm(x_flat.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)
        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):
            bases, coef = self.local_step(x_flat, bases, coef)
        coef = self.compute_coef(x_flat, bases, coef)
        out = torch.bmm(bases, coef.transpose(1, 2))

        return out.view(B, C, H, W)


class _ConvModule1x1(nn.Module):
    """The subset of MMCV ConvModule used by the upstream LightHamHead.

    Attribute names mirror MMCV (``conv``, ``gn``, ``activate``), which keeps
    the decoder state dictionary compatible with upstream checkpoints.
    """

    def __init__(self, in_channels, out_channels, *, norm, activate):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1, bias=not norm
        )
        self.gn = nn.GroupNorm(32, out_channels) if norm else None
        self.activate = nn.ReLU(inplace=True) if activate else None
        self.reset_parameters()

    def reset_parameters(self):
        # MMCV ConvModule's default ``kaiming_init`` uses fan_out and a normal
        # distribution for ReLU modules.
        nn.init.kaiming_normal_(
            self.conv.weight, mode='fan_out', nonlinearity='relu'
        )
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if self.gn is not None:
            nn.init.ones_(self.gn.weight)
            nn.init.zeros_(self.gn.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.gn is not None:
            x = self.gn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class Hamburger(nn.Module):
    """Exact LightHamHead Hamburger block with upstream GroupNorm."""

    def __init__(self, ham_channels=512, ham_kwargs=None):
        super().__init__()
        self.ham_in = _ConvModule1x1(
            ham_channels, ham_channels, norm=False, activate=False
        )
        self.ham = NMF2D(ham_kwargs)
        self.ham_out = _ConvModule1x1(
            ham_channels, ham_channels, norm=True, activate=False
        )

    def forward(self, x):
        enjoy = self.ham_in(x)
        enjoy = F.relu(enjoy, inplace=True)
        enjoy = self.ham(enjoy)
        enjoy = self.ham_out(enjoy)
        return F.relu(x + enjoy, inplace=True)


class LightHamHead(nn.Module):
    """Standalone port of the original SegNeXt LightHamHead.

    The upstream SegNeXt configuration selects backbone stages 1--3, resizes
    them to the stage-1 (1/8-resolution) grid, uses 32-group GroupNorm in the
    squeeze/Hamburger/align blocks, and applies the unmodified upstream NMF2D.
    """

    def __init__(self,
                 in_channels,
                 in_index=None,
                 ham_channels=512,
                 channels=512,
                 num_classes=2,
                 ham_kwargs=None,
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.in_index = list(in_index or [1, 2, 3])
        self.align_corners = align_corners
        selected_channels = [in_channels[i] for i in self.in_index]
        for value, label in ((ham_channels, 'ham_channels'), (channels, 'channels')):
            if value % 32:
                raise ValueError(
                    f"Upstream LightHamHead requires {label} divisible by 32 "
                    f"for GroupNorm, got {value}"
                )

        self.squeeze = _ConvModule1x1(
            sum(selected_channels), ham_channels, norm=True, activate=True
        )
        self.hamburger = Hamburger(ham_channels, ham_kwargs)
        self.align = _ConvModule1x1(
            ham_channels, channels, norm=True, activate=True
        )
        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
        )
        self.conv_seg = nn.Conv2d(channels, num_classes, 1)
        # BaseDecodeHead's upstream init_cfg initializes only ``conv_seg`` this way.
        nn.init.normal_(self.conv_seg.weight, mean=0.0, std=0.01)
        if self.conv_seg.bias is not None:
            nn.init.zeros_(self.conv_seg.bias)

    def forward_features(self, inputs):
        """Return the channel-rich H/8 tensor before class projection."""
        selected = [inputs[i] for i in self.in_index]
        target_size = selected[0].shape[2:]
        selected = [
            F.interpolate(
                feature,
                size=target_size,
                mode='bilinear',
                align_corners=self.align_corners,
            )
            for feature in selected
        ]
        x = self.squeeze(torch.cat(selected, dim=1))
        x = self.hamburger(x)
        return self.align(x)

    def forward(self, inputs):
        x = self.forward_features(inputs)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.conv_seg(x)


def _icnr_init_(weight, scale_factor=2):
    """Initialize subpixel kernels so every phase starts identically."""
    phase_count = scale_factor ** 2
    if weight.shape[0] % phase_count:
        raise ValueError(
            f"PixelShuffle output channels {weight.shape[0]} must be divisible "
            f"by scale_factor**2={phase_count}"
        )
    subkernel = weight.new_empty(
        weight.shape[0] // phase_count, *weight.shape[1:]
    )
    nn.init.kaiming_normal_(subkernel, mode='fan_out', nonlinearity='relu')
    with torch.no_grad():
        weight.copy_(subkernel.repeat_interleave(phase_count, dim=0))


def _group_count(channels, maximum=8):
    """Return the largest useful GroupNorm group count that divides channels."""
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ICNRSubpixelBlock(nn.Module):
    """Learned 2x feature upsampling with checkerboard-safe initialization."""

    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super().__init__()
        self.out_channels = out_channels
        self.expand = nn.Conv2d(
            in_channels, 4 * out_channels, 3, padding=1,
            padding_mode=padding_mode, bias=False,
        )
        _icnr_init_(self.expand.weight, scale_factor=2)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.activate = nn.GELU()

    def forward(self, x):
        return self.activate(self.norm(self.shuffle(self.expand(x))))



class _DetailConv(nn.Sequential):
    """Normalized spatial projection for a shallow detail feature."""

    def __init__(
        self, in_channels, out_channels, *, groups=1, activate=True,
        padding_mode='zeros',
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                groups=groups,
                padding_mode=padding_mode,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)



class _ResidualConcatFusion(nn.Module):
    """Cheap concat fusion with a projected residual and group-4 context."""

    def __init__(self, in_channels, out_channels, padding_mode='zeros'):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.update = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                groups=4,
                padding_mode=padding_mode,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.activate = nn.GELU()

    def forward(self, x):
        return self.activate(self.shortcut(x) + self.update(x))



class LearnedUpsampleHead(nn.Module):
    """Learned upsampling with optional H/4 and H/2 detail skips.

    The three resolution changes use the same ICNR PixelShuffle blocks as the
    learned-up decoder. H/4 encoder detail and H/2 image detail are fused with
    narrow group-4 residual blocks. The final learned upsample is classified
    directly, avoiding a costly full-resolution refinement block.
    """

    def __init__(
        self,
        in_channels,
        *args,
        raw_in_channels=3,
        tap_half_channels=16,
        upsample_channels,
        skip_resolutions,
        padding_mode='zeros',
        **kwargs,
    ):
        if args:
            raise TypeError("LearnedUpsampleHead accepts keyword arguments only")
        channels = int(kwargs.pop("channels"))
        ham_channels = int(kwargs.pop("ham_channels"))
        num_classes = int(kwargs.pop("num_classes"))
        ham_kwargs = kwargs.pop("ham_kwargs", None)
        dropout_ratio = float(kwargs.pop("dropout_ratio", 0.1))
        align_corners = bool(kwargs.pop("align_corners", False))
        if kwargs:
            raise TypeError(f"Unexpected decoder arguments: {sorted(kwargs)}")
        super().__init__()
        self.align_corners = align_corners
        skip_resolutions = tuple(int(value) for value in skip_resolutions)
        if skip_resolutions not in ((), (4,), (4, 2)):
            raise ValueError(
                "skip_resolutions must be (), (4,), or (4, 2)"
            )
        self.skip_resolutions = frozenset(skip_resolutions)
        coarse_channels = channels // 2
        if len(upsample_channels) != 3 or any(
                int(channels) <= 0 for channels in upsample_channels):
            raise ValueError(
                "upsample_channels must contain three positive channel counts"
            )
        quarter_channels, half_channels, full_channels = (
            int(channels) for channels in upsample_channels
        )
        fine_channels = max(quarter_channels // 2, 16)

        self.coarse_squeeze = _ConvModule1x1(
            sum(in_channels[1:]), ham_channels, norm=True, activate=True
        )
        self.hamburger = Hamburger(ham_channels, ham_kwargs)
        # The H/8 bottleneck keeps the first spatial upsampler inexpensive.
        self.coarse_project = _ConvModule1x1(
            ham_channels, coarse_channels, norm=True, activate=True
        )
        self.coarse_to_quarter = ICNRSubpixelBlock(
            coarse_channels, quarter_channels,
            padding_mode=padding_mode,
        )
        if 4 in self.skip_resolutions:
            self.fine_projection = nn.Sequential(
                nn.Conv2d(in_channels[0], fine_channels, 1, bias=False),
                nn.GroupNorm(_group_count(fine_channels), fine_channels),
                nn.GELU(),
            )
            self.quarter_fusion = _ResidualConcatFusion(
                quarter_channels + fine_channels, quarter_channels,
                padding_mode=padding_mode,
            )

        self.quarter_to_half = ICNRSubpixelBlock(
            quarter_channels, half_channels,
            padding_mode=padding_mode,
        )
        if 2 in self.skip_resolutions:
            self.tap_half = _DetailConv(
                raw_in_channels, int(tap_half_channels),
                padding_mode=padding_mode,
            )
            self.half_fusion = _ResidualConcatFusion(
                half_channels + int(tap_half_channels), half_channels,
                padding_mode=padding_mode,
            )

        self.half_to_full = ICNRSubpixelBlock(
            half_channels, full_channels,
            padding_mode=padding_mode,
        )
        self.dropout = (
            nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else nn.Identity()
        )
        self.conv_seg = nn.Conv2d(full_channels, num_classes, 1)
        nn.init.normal_(self.conv_seg.weight, mean=0.0, std=0.01)
        if self.conv_seg.bias is not None:
            nn.init.zeros_(self.conv_seg.bias)

    @staticmethod
    def _match_size(feature, reference):
        if feature.shape[2:] == reference.shape[2:]:
            return feature
        return F.interpolate(
            feature,
            size=reference.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, inputs, raw_image):
        target_size = inputs[1].shape[2:]
        coarse = torch.cat(
            [
                inputs[1],
                *[
                    F.interpolate(
                        feature,
                        size=target_size,
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for feature in inputs[2:]
                ],
            ],
            dim=1,
        )
        coarse = self.coarse_squeeze(coarse)
        coarse = self.hamburger(coarse)
        coarse = self.coarse_project(coarse)
        coarse = self.coarse_to_quarter(coarse)
        if 4 in self.skip_resolutions:
            fine = self._match_size(self.fine_projection(inputs[0]), coarse)
            quarter = self.quarter_fusion(torch.cat((coarse, fine), dim=1))
        else:
            quarter = coarse

        x = self.quarter_to_half(quarter)
        if 2 in self.skip_resolutions:
            raw_half = F.interpolate(
                raw_image,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            detail = self.tap_half(raw_half)
            half = self.half_fusion(torch.cat((x, detail), dim=1))
        else:
            half = x

        full = self.half_to_full(half)
        full = self.dropout(full)
        return self.conv_seg(full)



class SegNeXt(nn.Module):
    """MSCAN encoder with the decoder variants retained in the study."""

    CONFIG = {
        "embed_dims": [32, 64, 160, 256],
        "depths": [3, 3, 5, 2],
        "mlp_ratios": [8, 8, 4, 4],
        "decoder_channels": 256,
    }
    DECODERS = {"ham", "learned_up"}

    def __init__(
        self,
        in_chans=3,
        num_classes=2,
        drop_path_rate=0.1,
        decoder_type="ham",
        ham_kwargs=None,
        dropout_ratio=0.1,
        padding_mode="zeros",
        decoder_upsample_channels=None,
        decoder_tap_half_channels=None,
        decoder_skips=None,
    ):
        super().__init__()
        if decoder_type not in self.DECODERS:
            raise ValueError(f"Unsupported study decoder: {decoder_type!r}")
        config = self.CONFIG
        embed_dims = config["embed_dims"]
        decoder_channels = config["decoder_channels"]
        self.embed_dims = embed_dims
        self.decoder_type = decoder_type
        self.backbone = MSCAN(
            in_chans=in_chans,
            embed_dims=embed_dims,
            mlp_ratios=config["mlp_ratios"],
            drop_path_rate=drop_path_rate,
            depths=config["depths"],
            padding_mode=padding_mode,
        )
        common = dict(
            in_channels=embed_dims,
            ham_channels=decoder_channels,
            channels=decoder_channels,
            num_classes=num_classes,
            ham_kwargs=ham_kwargs or {"MD_R": 64},
            dropout_ratio=dropout_ratio,
        )
        if decoder_type == "ham":
            self.decode_head = LightHamHead(**common)
        else:
            if decoder_upsample_channels is None:
                raise ValueError(
                    "learned_up requires decoder_upsample_channels"
                )
            if decoder_skips is None:
                raise ValueError("learned_up requires decoder_skips")
            if 2 in decoder_skips and decoder_tap_half_channels is None:
                raise ValueError(
                    "an H/2 skip requires decoder_tap_half_channels"
                )
            self.decode_head = LearnedUpsampleHead(
                **common,
                raw_in_channels=in_chans,
                upsample_channels=decoder_upsample_channels,
                tap_half_channels=decoder_tap_half_channels,
                skip_resolutions=decoder_skips,
                padding_mode=padding_mode,
            )

    def forward(self, x):
        output_size = x.shape[-2:]
        features = self.backbone(x)
        if self.decoder_type == "learned_up":
            logits = self.decode_head(features, x)
        else:
            logits = self.decode_head(features)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )
        return logits

    def load_pretrained_backbone(self, checkpoint_path):
        """Load compatible MSCAN encoder weights from an upstream checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        candidate = {}
        for key, value in state.items():
            if key.startswith("backbone."):
                candidate[key[len("backbone."):]] = value
            elif not key.startswith("decode_head"):
                candidate[key] = value

        remapped = {}
        for key, value in candidate.items():
            parts = key.rsplit(".", 1)
            if len(parts) == 2:
                prefix, suffix = parts
                module = self.backbone
                try:
                    for attribute in prefix.split("."):
                        module = getattr(module, attribute)
                except (AttributeError, TypeError):
                    module = None
                if isinstance(module, SafePadConv2d):
                    remapped[f"{prefix}.conv.{suffix}"] = value
                    continue
            remapped[key] = value

        target = self.backbone.state_dict()
        compatible = {
            key: value for key, value in remapped.items()
            if key in target and target[key].shape == value.shape
        }
        self.backbone.load_state_dict(compatible, strict=True)
        print(
            f"Strictly loaded {len(compatible)} pretrained MSCAN tensors from "
            f"{checkpoint_path}"
        )
