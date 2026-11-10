"""Shared style for paper figures (Coastal Engineering, Elsevier elsarticle).

Single source of truth for every figure script in this directory. Change a value
here (font, palette, colormap, output format) and it propagates to all figures.
"""

from pathlib import Path

import cmocean  # noqa: F401  — registers 'cmo.*' colormaps with matplotlib
import matplotlib as mpl
from cycler import cycler

# ── Config — change these to retune the whole paper ─────────────────────────────
FONT_FAMILY  = "sans-serif"       # set to "serif" to match Elsevier body text
SAVE_FORMATS = ("pdf", "png")     # vector for submission + raster for preview/PowerPoint

# Figure widths (inches)
COL_W  = 3.5    # single column (~3p or 3p,twocolumn)
PAGE_W = 7.0    # full text width (~3p single or preprint)

DPI = 300

# Font sizes (pt) — legible at single-column print size
FONTSIZE_LABEL  = 9
FONTSIZE_TICK   = 8
FONTSIZE_LEGEND = 7

# ── Categorical palette — IBM colorblind-safe ───────────────────────────────────
# blue, purple, magenta, orange, amber
IBM = ["#648FFF", "#785EF0", "#DC267F", "#FE6100", "#FFB000"]

# Semantic categorical roles reused across figures
COLOR_SURVEY   = IBM[0]   # #648FFF
COLOR_TRANSECT = IBM[2]   # #DC267F
COLOR_ARGUS    = IBM[4]   # #FFB000

# Profile / alongshore-location colors (location 1/2/3) — used by 02 & 04
PROFILE_COLORS = [IBM[0], IBM[4], IBM[2]]

# ── Sequential colormap registry — one per physical quantity ────────────────────
CMAP = {
    "Qb":      "magma",     # breaking fraction Q_b
    "depth":   "cmo.deep_r",  # bed elevation (reversed so deep=dark, beach=light)
    "diss":    "inferno",   # dissipation rate D
    "density": "RdPu",      # 2D observation-density hist2d
    "spectro": "Greys",     # spectrograms
    "diverge": "RdBu_r",    # survey - inverted depth error
}

# ── Axis-label constants — kill wording drift ───────────────────────────────────
XLABEL_CROSS = "Cross-shore distance [m]"
XLABEL_ALONG = "Alongshore distance [m]"
LABEL_DEPTH  = "Depth [m]"
LABEL_ELEV   = "Elevation [m]"
LABEL_QB     = r"$Q_b$"

# Shared colour limits for total water depth (m) — unify 04 & 05.
# Land (depth < 0) is masked saddlebrown.
DEPTH_VMIN, DEPTH_VMAX = 0.0, 5.0
DEPTH_TICKS = [0, 2, 4]


def apply_style():
    mpl.rcParams.update({
        "font.family":        FONT_FAMILY,
        "font.size":          FONTSIZE_TICK,
        "axes.labelsize":     FONTSIZE_LABEL,
        "axes.titlesize":     FONTSIZE_LABEL,
        "xtick.labelsize":    FONTSIZE_TICK,
        "ytick.labelsize":    FONTSIZE_TICK,
        "legend.fontsize":    FONTSIZE_LEGEND,
        "lines.linewidth":    1.0,
        "axes.linewidth":     0.8,
        "axes.prop_cycle":    cycler(color=IBM),
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.minor.width":  0.6,
        "ytick.minor.width":  0.6,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        # Consistent legend frame across all figures (no outline)
        "legend.framealpha":  0.9,
        "legend.edgecolor":   "none",
        "legend.fancybox":    True,
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })


def savefig(fig, path, **kwargs):
    """Save `fig` once per extension in SAVE_FORMATS.

    `path` may include an extension (it is stripped) or be a bare stem. The
    raster preview `<stem>.png` is written beside the script; the vector
    `<stem>.pdf` is collected in a `pdf/` subfolder next to it.
    """
    kwargs.setdefault("dpi", DPI)
    kwargs.setdefault("bbox_inches", "tight")
    stem = Path(path).with_suffix("")
    for ext in SAVE_FORMATS:
        out_dir = stem.parent / "pdf" if ext == "pdf" else stem.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{stem.name}.{ext}"
        fig.savefig(out, **kwargs)
        print(f"Saved: {out}")


# ── Shared helpers ──────────────────────────────────────────────────────────────

_PANEL_BBOX = {"boxstyle": "round,pad=0.15", "facecolor": "white",
                   "edgecolor": "none", "alpha": 0.8}

_PANEL_PAD = 3.0   # points from the corner — fixed visual gap, aspect-independent

# corner anchor (axes fraction), offset direction (points), alignment
_PANEL_LOC = {
    "upper left":  ((0.0, 1.0), ( 1, -1), "top",    "left"),
    "upper right": ((1.0, 1.0), (-1, -1), "top",    "right"),
    "lower left":  ((0.0, 0.0), ( 1,  1), "bottom", "left"),
    "lower right": ((1.0, 0.0), (-1,  1), "bottom", "right"),
}


def panel_label(ax, text, loc="upper left", box=True):
    """Canonical panel tag, e.g. 'a'. Bold, white round box, tucked into corner.

    Offset is in points from the corner so the visual gap is identical across
    panels of any aspect ratio (tall-skinny vs short-squat). Surrounding
    parentheses are stripped so callers may pass '(a)' or 'a'.
    """
    text = text.strip("()")
    (ax_x, ax_y), (dx, dy), va, ha = _PANEL_LOC[loc]
    ax.annotate(
        text, xy=(ax_x, ax_y), xycoords="axes fraction",
        xytext=(dx * _PANEL_PAD, dy * _PANEL_PAD), textcoords="offset points",
        fontsize=FONTSIZE_LABEL, fontweight="bold", va=va, ha=ha,
        bbox=_PANEL_BBOX if box else None)


def styled_legend(ax, **kwargs):
    """Legend with a consistent white frame and no outline."""
    kwargs.setdefault("fontsize", FONTSIZE_LEGEND)
    leg = ax.legend(**kwargs)
    if leg is not None:
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_alpha(0.9)
    return leg


def add_colorbar(fig, ax, im, label, **kwargs):
    """Colorbar height-matched to its parent axes."""
    kwargs.setdefault("fraction", 0.046)
    kwargs.setdefault("pad", 0.04)
    cb = fig.colorbar(im, ax=ax, **kwargs)
    cb.set_label(label)
    return cb


def draw_land(ax, depth_da, x="xFRF", y="yFRF"):
    """Saddlebrown land fill + black shoreline contour from a depth DataArray.

    `depth_da` should already be reduced to a single time slice.
    """
    depth_da.plot.contourf(
        ax=ax, x=x, y=y, levels=[-100, 0],
        colors=["saddlebrown"], alpha=1, add_colorbar=False,
        extend="neither", zorder=0)
    depth_da.plot.contour(
        ax=ax, x=x, y=y, levels=[0], colors=["black"],
        linewidths=1.0, zorder=2)


# FRF pier coords from SBFRF/cmtb: github.com/SBFRF/cmtb/blob/1e6451fa/plotting/operationalPlots.py#L239
PIER_Y          = 516    # FRF pier centerline (yFRF, m)
PIER_X_END      = 580    # pier seaward extent (xFRF, m)


def draw_pier(ax, orientation="horizontal", label="FRF Pier"):
    """Dimgray FRF pier line. orientation: 'horizontal' (yFRF const, Fig 04)
    or 'vertical' (xFRF const, Fig 05)."""
    if orientation == "horizontal":
        ax.plot([0, PIER_X_END], [PIER_Y, PIER_Y], color="dimgray",
                linewidth=3.5, linestyle="-", zorder=6, label=label)
    else:
        ax.plot([PIER_Y, PIER_Y], [0, PIER_X_END], color="dimgray",
                linewidth=3.5, linestyle="-", zorder=6, label=label)
