"""02_timex_comparison: timex image, Qb map, and cross-shore profiles at 3 alongshore locations."""

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use('Agg')
import dunex_paths
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from common.figure_style import (
    CMAP,
    PAGE_W,
    PROFILE_COLORS,
    XLABEL_ALONG,
    XLABEL_CROSS,
    apply_style,
    panel_label,
    savefig,
    styled_legend,
)

DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"
VIDEO_DIR = dunex_paths.ARGUS_VIDEO_DIR
OUTPUT_PATH = Path(__file__).parent / "02_timex_comparison.png"

DATESTR = '20210920T190100Z'

# FRF display limits (metres)
XLIM          = [1500, -100]         # alongshore (image x-axis)
YLIM          = [0,    500]          # cross-shore / xFRF for image panels
YTICKS        = [100, 200, 300, 400, 500]
PROFILE_YLIM  = [75,  275]           # xFRF range shown in profile panels
PROFILE_YTICKS = [75, 175, 275]

# Three profile locations (yFRF, m) — south, centre, north
PROFILE_YFRF  = [1200, 750, 300]
PROFILE_BAND  = 25   # ± metres to average over

QB_CMAP = CMAP['Qb']
QB_VMIN = 0


def load_frames(path):
    """Load all frames from a video as (N, H, W) float32 grayscale."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        frames.append(gray)
    cap.release()
    return np.stack(frames).astype(np.float32)


def main():
    apply_style()

    # -------------------------------------------------------------------------
    # Data
    # -------------------------------------------------------------------------
    ds   = xr.open_dataset(DATASET)
    slot = ds.sel(time=pd.to_datetime(DATESTR, format='%Y%m%dT%H%M%SZ'),
                  method='nearest')

    Tm             = float(slot.waveTm.values)
    video_duration = float(slot.video_duration.values)
    print(f"Tm = {Tm:.1f} s,  duration = {video_duration:.0f} s")

    # Qb = Nb / Ntotal = crest_count / (video_duration / Tm)  — shape (yFRF, xFRF)
    # Same crest-count formula as figures 05/06 (video_duration = trimmed_frames / fps).
    Qb = np.clip((slot.crest_count.values / video_duration) * Tm, None, 1.0)

    # Timex from video frames
    print("Loading video frames…")
    video_file = VIDEO_DIR / f'ArgusFF_{DATESTR}_RectifiedVideo.avi'
    frames     = load_frames(video_file)
    timex      = frames.mean(axis=0)   # (H=1600, W=500) — rows=yFRF, cols=xFRF

    qb_vmax = float(np.percentile(Qb[Qb > 0], 99))
    print(f"Qb 99th pct: {qb_vmax:.3f}")

    # Rotate 90° CCW for map display: rows → yFRF (x), cols → xFRF (y)
    timex_rot = np.rot90(timex, k=1)   # (500, 1600)
    Qb_rot    = np.rot90(Qb,    k=1)

    extent = [float(ds.yFRF[0]),  float(ds.yFRF[-1]),   # left, right
              float(ds.xFRF[0]),  float(ds.xFRF[-1])]   # bottom, top

    # Qb RGBA overlay
    norm    = mcolors.Normalize(vmin=QB_VMIN, vmax=1)
    qb_norm = np.clip(norm(Qb_rot), 0, 1)
    qb_rgba = plt.get_cmap(QB_CMAP)(qb_norm)
    qb_rgba[..., 3] = np.where(Qb_rot > 0.02, np.clip(qb_norm * 2, 0, 0.85), 0)

    # Cross-shore profiles: average over a PROFILE_BAND-wide yFRF strip
    xFRF_vals = ds.xFRF.values                          # (500,) 0…500 m
    yFRF_vals = ds.yFRF.values                          # (1600,) 1500…-100 m
    xfrf_crop = (xFRF_vals >= PROFILE_YLIM[0]) & (xFRF_vals <= PROFILE_YLIM[1])
    xfrf_axis = xFRF_vals[xfrf_crop]                   # profile y-axis values

    profiles = []
    for y0 in PROFILE_YFRF:
        band = (yFRF_vals >= y0 - PROFILE_BAND) & (yFRF_vals <= y0 + PROFILE_BAND)
        profiles.append({
            'intensity': timex[band, :][:, xfrf_crop].mean(axis=0),
            'Qb':        Qb[band,   :][:, xfrf_crop].mean(axis=0),
        })

    # -------------------------------------------------------------------------
    # Figure — subplot_mosaic
    # -------------------------------------------------------------------------
    mosaic = [
        ['timex', 'timex', 'timex', '.'   ],
        ['qb',    'qb',    'qb',    'cbar'],
        ['c0',    'c1',    'c2',    '.'   ],
    ]
    fig, axd = plt.subplot_mosaic(
        mosaic,
        figsize=(PAGE_W, 6),
        gridspec_kw={
            'height_ratios': [1, 1, 1],
            'width_ratios':  [1, 1, 1, 0.04],
        },
        layout='constrained',
    )

    ax_timex = axd['timex']
    ax_qb    = axd['qb']
    ax_cbar  = axd['cbar']

    # Share x between image panels
    ax_qb.sharex(ax_timex)

    colors_profile = PROFILE_COLORS

    # --- Timex panel ---
    # aspect='auto' so the axes fills its gridspec cell — avoids colorbar height mismatch
    ax_timex.imshow(timex_rot, cmap='gray', vmin=0, vmax=255,
                    extent=extent, aspect='auto')
    ax_timex.set_xlim(XLIM)
    ax_timex.set_ylim(YLIM)
    ax_timex.set_yticks(YTICKS)
    ax_timex.tick_params(labelbottom=False)
    ax_timex.set_ylabel('Cross-shore distance [m]')
    panel_label(ax_timex, '(a)')
    for y0, c in zip(PROFILE_YFRF, colors_profile):
        ax_timex.axvline(y0, color=c, linewidth=0.8, linestyle='--')

    # --- Qb image panel ---
    ax_qb.imshow(timex_rot, cmap='gray', vmin=0, vmax=255,
                 extent=extent, aspect='auto')
    ax_qb.imshow(qb_rgba, extent=extent, aspect='auto')
    ax_qb.set_xlim(XLIM)
    ax_qb.set_ylim(YLIM)
    ax_qb.set_yticks(YTICKS)
    ax_qb.tick_params(labelbottom=True)
    ax_qb.set_xlabel(XLABEL_ALONG)
    ax_qb.set_ylabel('Cross-shore distance [m]')
    panel_label(ax_qb, '(b)')
    for y0, c in zip(PROFILE_YFRF, colors_profile):
        ax_qb.axvline(y0, color=c, linewidth=0.8, linestyle='--')

    sm = plt.cm.ScalarMappable(cmap=QB_CMAP, norm=norm)
    cb = fig.colorbar(sm, cax=ax_cbar, label=r'$Q_b$')
    cb.set_ticks([0, 0.5, 1])

    # --- Profile panels (xFRF on y-axis, twiny for Qb) ---
    # Reversed so left→right matches the image's yFRF direction (1500→-100)
    panel_keys = ['c0', 'c1', 'c2']
    from matplotlib.lines import Line2D
    for i, (key, y0, prof, c) in enumerate(
            zip(panel_keys, PROFILE_YFRF, profiles, colors_profile)):
        ax = axd[key]

        ax.plot(prof['intensity'], xfrf_axis, color=c, linewidth=0.8, linestyle='-')
        ax.set_ylim(PROFILE_YLIM)
        ax.set_yticks(PROFILE_YTICKS)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        ax.set_xlabel('Intensity')

        if i == 0:
            ax.set_ylabel(XLABEL_CROSS)
        else:
            ax.tick_params(labelleft=False)

        # Second x-axis (top) for Qb — uncolored
        ax2 = ax.twiny()
        ax2.plot(prof['Qb'], xfrf_axis, color=c, linewidth=0.8, linestyle='--')
        ax2.tick_params(axis='x', labelsize=7, labeltop=True)
        ax2.set_xlim(0, 1)
        ax2.set_xticks([0, 0.5, 1])
        ax2.set_xlabel(r'$Q_b$', fontsize=7)

        panel_label(ax, f'({"cde"[i]})')

    # Solid/dash legend on the leftmost profile panel
    legend_handles = [
        Line2D([0], [0], color='k', linestyle='-',  linewidth=0.8, label='Intensity'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=0.8, label=r'$Q_b$'),
    ]
    styled_legend(axd['c0'], handles=legend_handles, loc='upper right', fontsize=6)

    savefig(fig, OUTPUT_PATH)


if __name__ == '__main__':
    main()
