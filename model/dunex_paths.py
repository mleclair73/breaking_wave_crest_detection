"""Repository paths and optional environment overrides.

Install ``model/`` or run scripts from that directory to import this module.
``resolve()`` expands the placeholders below in YAML paths, then expands shell
variables and anchors relative paths at the repository root.

    DUNEX_DATA_ROOT / DATA_ROOT   dataset root (default: <repo>/data)
    DUNEX_OUTPUTS_DIR             trained-model outputs
    DUNEX_SAVGOL_DIR              external Stage-2 speed fields
    DUNEX_ARGUS_RAW_DIR           full raw Argus archive, used only by figure 00a
"""

import os
from pathlib import Path

# This file is <repo>/model/dunex_paths.py, so the repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(default, *names):
    """First set env var among ``names`` wins; else ``default``."""
    for name in names:
        val = os.environ.get(name)
        if val:
            return Path(val).expanduser()
    return Path(default)


# ── Roots (env-overridable) ────────────────────────────────────────────────
DATA_ROOT = _env_path(REPO_ROOT / "data", "DUNEX_DATA_ROOT", "DATA_ROOT")
OUTPUTS_DIR = _env_path(REPO_ROOT / "model" / "outputs", "DUNEX_OUTPUTS_DIR")
PREDICTIONS_DIR = REPO_ROOT / "model" / "predictions"

# Optional external data roots.
SAVGOL_DIR = _env_path(
    REPO_ROOT.parent / "mae_dunex_revamp" / "output_segnext_savgol",
    "DUNEX_SAVGOL_DIR",
)
ARGUS_RAW_DIR = _env_path(
    REPO_ROOT.parent / "dunex" / "data" / "argus",
    "DUNEX_ARGUS_RAW_DIR",
)

# ── Derived dataset locations ──────────────────────────────────────────────
ARGUS_VIDEO_DIR = DATA_ROOT / "video_dataset" / "argus"   # vendored rectified videos
BATHY_DIR = DATA_ROOT / "bathy"
WATER_LEVEL_DIR = DATA_ROOT / "water_level"
WAVES_8M_DIR = DATA_ROOT / "8m_array_waves"
WATER_LEVEL_NC = WATER_LEVEL_DIR / "frf_eopNoaaTide_water_levels_2021_aug_nov.nc"

# Placeholders honoured inside YAML/config path strings by resolve().
_SUBS = {
    "REPO_ROOT": REPO_ROOT,
    "DATA_ROOT": DATA_ROOT,
    "OUTPUTS_DIR": OUTPUTS_DIR,
    "PREDICTIONS_DIR": PREDICTIONS_DIR,
    "SAVGOL_DIR": SAVGOL_DIR,
    "ARGUS_VIDEO_DIR": ARGUS_VIDEO_DIR,
    "ARGUS_RAW_DIR": ARGUS_RAW_DIR,
}


def resolve(path_str, base=REPO_ROOT):
    """Resolve a config path string to an absolute ``Path``.

    ``${DATA_ROOT}`` and the other placeholders in ``_SUBS`` are substituted
    first (so defaults work with no env vars set), then ``~`` and ``$ENV``
    vars are expanded, and finally a still-relative path is anchored at
    ``base`` (the repo root). Absolute paths pass through unchanged.
    """
    s = str(path_str)
    for name, val in _SUBS.items():
        s = s.replace(f"${{{name}}}", str(val))
    s = os.path.expanduser(os.path.expandvars(s))
    p = Path(s)
    return p if p.is_absolute() else (base / p)
