"""Sample Argus frame + alongshore-row timestack."""

from pathlib import Path

import matplotlib.pyplot as plt
import dunex_paths
from common.figure_style import PAGE_W, XLABEL_ALONG, XLABEL_CROSS, apply_style, panel_label, savefig

from utils.video_io import load_frames

# Raw Argus frame — not in the vendored subset, so it defaults to the external
# dunex checkout (override with DUNEX_ARGUS_RAW_DIR).
VIDEO_PATH  = str(dunex_paths.ARGUS_RAW_DIR / 'ArgusFF_20211014T123100Z_RectifiedVideo.avi')
OUTPUT_PATH = str(Path(__file__).parent / '00a_datasample.png')

FPS = 2.0
M_PER_PX_X = 1.0  # cross-shore meters per pixel
M_PER_PX_Y = 1.0  # alongshore meters per pixel
N_FRAMES   = 1600


def plot_sample_and_timestack(frames, frame_idx=0, row=None,
                              m_per_px_x=M_PER_PX_X, m_per_px_y=M_PER_PX_Y,
                              fps=FPS):
    N, H, W = frames.shape
    if row is None:
        row = H // 2

    x_extent = W * m_per_px_x
    y_extent = H * m_per_px_y
    t_extent = N / fps
    row_m = row * m_per_px_y

    fig_w = PAGE_W
    fig_h = fig_w / 1.4
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, fig_h),
                                   sharey=True,
                                   gridspec_kw={'hspace': 0.3})

    ax1.imshow(frames[frame_idx].T, cmap='gray', aspect=1,
               extent=[0, y_extent, 0, x_extent], origin='lower')
    ax1.set_box_aspect(x_extent / y_extent)
    ax1.axvline(row_m, color='red', lw=1.2, alpha=0.6, zorder=2)
    ax1.set_xlabel(XLABEL_ALONG)
    ax1.set_ylabel(XLABEL_CROSS)

    ax2.imshow(frames[:, row, :].T, cmap='gray', aspect='auto',
               extent=[0, t_extent, 0, x_extent], origin='lower')
    ax2.set_box_aspect(x_extent / y_extent)
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel(XLABEL_CROSS)

    for ax, label in zip((ax1, ax2), ('(a)', '(b)')):
        panel_label(ax, label)

    return fig


def main():
    apply_style()
    frames = load_frames(VIDEO_PATH, stop=N_FRAMES)
    fig = plot_sample_and_timestack(frames)
    savefig(fig, OUTPUT_PATH)


if __name__ == '__main__':
    main()
