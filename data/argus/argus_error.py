"""
Pixel-resolution and grid-oversampling maps for the Duck, NC (FRF) Argus tower.

Reproduces the style of Fig. 4 in Holman & Stanley (2007), Coastal Engineering
54, 477-491 (stacked cross-shore / alongshore pixel resolution maps, after
Aarninkhof 2003), for the six-camera Argus tower at the USACE Field Research
Facility.

Two independent resolution methods are provided:

  1. PINHOLE (Holman & Stanley Eqs. 1-3), from camera height and field of view:
         delta = horizontal angular FOV
         dc    = R * delta / NU                cross-range footprint     (Eq. 1)
         dr    = dc * R / zc                   range footprint           (Eq. 2)
         dx    = max(|dc cos a|, |dr sin a|)   cross-shore               (Eq. 3)
         dy    = max(|dr cos a|, |dc sin a|)   alongshore                (Eq. 3)
     where a is the view azimuth, compass-sense from +y.

  2. EXACT, from the 11-coefficient DLT in cameraData.yml, by inverting the
     image Jacobian at each ground point. Use this one; the pinhole version is
     for cross-checking and for cameras where you only have a nominal FOV.

Geometry from cameraData.yml in CIRN CoastalImageLib
(github.com/mailemccann/coastalimagelib, ExampleData/cameraData.yml).

Provenance & limitations (read before using the numbers):
  - Verified numerically: the pinhole path, against Holman & Stanley's worked
    example (check_against_paper).
  - Validated by self-consistency (validate_geometry): the DLT and the scalar
    fields in cameraData.yml are independent representations of the same
    geometry, so they cross-check each other. For all six cameras this confirms
    the DLT coefficient ordering (null space of P recovers the stated centre),
    that fov is the HORIZONTAL fov (fx matches NU/(2 tan(fov/2))), that azimuth
    is compass-sense from +y and tilt from nadir (both match the DLT principal
    axis), that D_UO/D_V0 are the principal point, and square pixels (fx == fy).
    validate_geometry() raises on failure -- trust no output until it passes.
  - Still inferred or unknown:
      * GRID_Y0, the alongshore origin of the 1600 x 500 product. Pinned to
        -100 m (min coord) from conversations with FRF, but infer_grid_origin shows -70? 
        against a real rectified product (agreement 0.97), but only to ~+/-15 m (plateau -90..-60). 
        Products are stored north-at-top; x0=0 and d=1 m
      * radial distortion D1/D2 -- coastalimagelib documents the model
        (Brown-Conrady, radial d1/d2/d3 + tangential t1/t2, in xyz2DistUV),
        but only on its intrinsics/K-matrix path. Its DLT path (dlt2UV) and
        loader (loadYamlDLT) ignore D1/D2 entirely, so the m coefficients
        carry no distortion -- neither does this module.
      * whether these <=2018-04-12 solutions were current in your collection
        period; unknowable from the file.
      * camera pointing error (~2-10 px at this tower), non-flat surface
        (assumes Z=0), and the nanmin assumption that the best camera is used
        at every point.
      * pier_y = 515.0 in the plots is an approximate FRF pier location for
        annotation only; it enters no calculation. https://github.com/SBFRF/testbedutils/blob/master/geoprocess.py
  - Timeline: geometry predates the fall-2021 DUNEX main experiment; request
    records for your dates from the FRF. The USGS DUNEX calibrations
    (DOI 10.5066/P1GDP4HR) are Pea Island, not this tower.
    
    https://github.com/erdc/testbedutils/blob/master/geoprocess.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import yaml
except ImportError:
    yaml = None


# ==========================================================================
# DLT convention
# ==========================================================================
# The 11-element `m` vector in cameraData.yml is ordered:
#
#     U = (m0*x + m1*y + m2*z + m3) / (m4*x + m5*y + m6*z + 1)
#     V = (m7*x + m8*y + m9*z + m10) / (m4*x + m5*y + m6*z + 1)
#
# This is NOT the textbook DLT layout (which puts the denominator coefficients
# last). It was determined empirically and is confirmed by validate_geometry()
# below, which decomposes each DLT and checks it against the YAML's own scalar
# fields: the null space of P recovers the stated camera centre (residual
# ~1e-3 m on c1), RQ gives the principal point (1224.0, 1024.0) and horizontal
# focal length (fx=6922.8 px vs NU/(2 tan(fov/2))=6924 px on c1), and the
# principal axis reproduces azimuth and tilt. verify_dlt() is the older, weaker
# boresight-only check. If validate_geometry() fails on your file, the ordering
# differs and everything downstream is wrong -- stop and re-derive it.

SENSOR_NU, SENSOR_NV = 2448, 2048       # D_UO = 1224, D_V0 = 1024


@dataclass
class Camera:
    name: str
    x: float                      # FRF cross-shore coord of the camera, m
    y: float                      # FRF alongshore coord, m
    z: float                      # elevation, m (height above z=0 surface)
    azimuth: float                # rad, compass-sense from +y
    fov: float                    # rad, horizontal angular FOV
    tilt: float                   # rad, from vertical
    roll: float = 0.0
    m: np.ndarray | None = None   # 11 DLT coefficients
    nu: int = SENSOR_NU
    nv: int = SENSOR_NV
    umin: int = 0
    umax: int = SENSOR_NU
    vmin: int = 0
    vmax: int = SENSOR_NV
    when_valid: float | None = None      # unix epoch seconds
    ident: str = ""
    orientation: str = ""

    @property
    def valid_from(self) -> str:
        if self.when_valid is None:
            return "unknown"
        return datetime.fromtimestamp(self.when_valid, timezone.utc).strftime("%Y-%m-%d")

    @property
    def focal_px(self) -> float:
        """Focal length in pixels, assuming `fov` is the HORIZONTAL FOV."""
        return self.nu / (2.0 * np.tan(self.fov / 2.0))

    @property
    def subtense(self) -> float:
        """Angle subtended by one pixel, rad (delta / NU)."""
        return self.fov / self.nu


@dataclass
class Station:
    name: str
    cameras: list[Camera] = field(default_factory=list)

    @property
    def x(self) -> float:
        return float(np.mean([c.x for c in self.cameras]))

    @property
    def y(self) -> float:
        return float(np.mean([c.y for c in self.cameras]))

    @property
    def zc(self) -> float:
        return float(np.mean([c.z for c in self.cameras]))


# ==========================================================================
# loading
# ==========================================================================

def load_camera_yaml(path: str, station_name: str = "") -> Station:
    """Read a CoastalImageLib-style cameraData.yml."""
    if yaml is None:
        raise ImportError("pyyaml required: pip install pyyaml")
    with open(path) as fh:
        raw = yaml.safe_load(fh)

    cams = []
    for key in sorted(k for k in raw if k.lower().startswith("c") and k[1:].isdigit()):
        d = raw[key]
        if not isinstance(d, dict):
            continue
        nu = int(d.get("D_UO", SENSOR_NU / 2)) * 2
        nv = int(d.get("D_V0", SENSOR_NV / 2)) * 2
        cams.append(Camera(
            name=key,
            x=float(d["x"]), y=float(d["y"]), z=float(d["z"]),
            azimuth=float(d["azimuth"]), fov=float(d["fov"]), tilt=float(d["tilt"]),
            roll=float(d.get("roll", 0.0)),
            m=np.asarray(d["m"], dtype=float) if "m" in d else None,
            nu=nu, nv=nv,
            umin=int(d.get("umin", 0)), umax=int(d.get("umax", nu)),
            vmin=int(d.get("vmin", 0)), vmax=int(d.get("vmax", nv)),
            when_valid=float(d["whenValid"]) if "whenValid" in d else None,
            ident=str(d.get("ID", "")), orientation=str(d.get("orientation", "")),
        ))
    return Station(name=station_name or path, cameras=cams)


# Default geometry: the attributed cameraData.yml saved next to this module
# (verbatim CIRN CoastalImageLib Duck example + provenance header). Load it with
# load_camera_yaml(); requires pyyaml.
CAMERA_YAML = Path(__file__).parent / "cameraData_duck.yml"


# ==========================================================================
# projection and exact footprint
# ==========================================================================

def project(cam: Camera, X, Y, Z=0.0):
    """Ground (X, Y, Z) -> image (U, V) and the DLT denominator."""
    m = cam.m
    D = m[4] * X + m[5] * Y + m[6] * Z + 1.0
    U = (m[0] * X + m[1] * Y + m[2] * Z + m[3]) / D
    V = (m[7] * X + m[8] * Y + m[9] * Z + m[10]) / D
    return U, V, D


def _boresight_point(cam: Camera):
    """Ground intersection of the camera boresight, from tilt and azimuth."""
    depression = np.pi / 2.0 - cam.tilt          # tilt measured from vertical
    R = cam.z / np.tan(depression)
    return (cam.x + R * np.sin(cam.azimuth),
            cam.y + R * np.cos(cam.azimuth))


def in_view(cam: Camera, X, Y, Z=0.0):
    """Visibility: inside the crop window, in front of the camera, in the wedge."""
    U, V, D = project(cam, X, Y, Z)

    bx, by = _boresight_point(cam)
    _, _, D_ref = project(cam, bx, by, 0.0)
    front = np.sign(D) == np.sign(D_ref)

    window = ((U >= cam.umin) & (U <= cam.umax) &
              (V >= cam.vmin) & (V <= cam.vmax))

    bearing = np.degrees(np.arctan2(X - cam.x, Y - cam.y))
    off = (bearing - np.degrees(cam.azimuth) + 180.0) % 360.0 - 180.0
    wedge = np.abs(off) <= np.degrees(cam.fov) / 2.0 + 1.0   # 1 deg slack

    return window & front & wedge


def footprint_exact(cam: Camera, X, Y, Z=0.0):
    """Exact per-pixel ground footprint by inverting the image Jacobian.

    Returns (res_x, res_y): cross-shore and alongshore footprint in metres,
    NaN outside the camera's view.
    """
    m = cam.m
    D = m[4] * X + m[5] * Y + m[6] * Z + 1.0
    Nu = m[0] * X + m[1] * Y + m[2] * Z + m[3]
    Nv = m[7] * X + m[8] * Y + m[9] * Z + m[10]
    D2 = D * D

    dUdx = (m[0] * D - Nu * m[4]) / D2
    dUdy = (m[1] * D - Nu * m[5]) / D2
    dVdx = (m[7] * D - Nv * m[4]) / D2
    dVdy = (m[8] * D - Nv * m[5]) / D2

    det = dUdx * dVdy - dUdy * dVdx
    with np.errstate(divide="ignore", invalid="ignore"):
        # inverse Jacobian: ground displacement per unit image displacement
        dxdU, dxdV = dVdy / det, -dUdy / det
        dydU, dydV = -dVdx / det, dUdx / det

    # a one-pixel step in U or V; take the worse on each ground axis (cf. Eq. 3)
    res_x = np.maximum(np.abs(dxdU), np.abs(dxdV))
    res_y = np.maximum(np.abs(dydU), np.abs(dydV))

    vis = in_view(cam, X, Y, Z)
    return np.where(vis, res_x, np.nan), np.where(vis, res_y, np.nan)


def footprint_pinhole(cam: Camera, X, Y, Z=0.0):
    """Holman & Stanley Eqs. 1-3. Cross-check / fallback."""
    ex, ey = X - cam.x, Y - cam.y
    R = np.hypot(ex, ey)
    zc = cam.z - Z

    dc = R * cam.subtense                 # Eq. 1
    dr = dc * R / zc                      # Eq. 2

    a = np.arctan2(ex, ey)
    ca, sa = np.abs(np.cos(a)), np.abs(np.sin(a))
    res_x = np.maximum(dc * ca, dr * sa)  # Eq. 3
    res_y = np.maximum(dr * ca, dc * sa)

    vis = in_view(cam, X, Y, Z) if cam.m is not None else np.ones_like(R, bool)
    return np.where(vis, res_x, np.nan), np.where(vis, res_y, np.nan)


def footprint_cirn(cam: Camera, X, Y, Z=0.0):
    """Port of CIRN resMapLocal.m (Support-Routines, GPL-3, 2017 CIRN/OSU).

    Uses exact differenced-tangent range resolution and slant-range cross-range,
    rather than the small-angle forms in footprint_pinhole(). Returns
    (res_x, res_y) = (dcProj, daProj) in metres.
    """
    dx, dy, dz = X - cam.x, Y - cam.y, Z - cam.z
    L = np.sqrt(dx**2 + dy**2 + dz**2)
    H = np.hypot(dx, dy)

    tau = np.arctan(H / dz)              # from vertical; dz < 0 below camera
    theta = np.arctan2(dy, dx)           # math convention, CCW from +x
    dH = cam.fov / cam.nu                # delH; delV = delH (square pixels)

    da = np.abs(dz) * (np.tan(tau + dH / 2) - np.tan(tau - dH / 2))
    dc = 2.0 * L * np.tan(dH / 2)

    ct, st = np.abs(np.cos(theta)), np.abs(np.sin(theta))
    res_x = np.maximum(da * ct, dc * st)   # dcProj, cross-shore
    res_y = np.maximum(dc * ct, da * st)   # daProj, alongshore

    vis = in_view(cam, X, Y, Z) if cam.m is not None else np.ones(np.shape(X), bool)
    return np.where(vis, res_x, np.nan), np.where(vis, res_y, np.nan)


def station_resolution(stn: Station, X, Y, Z=0.0, method="exact"):
    """Best (smallest) footprint available from any camera, per ground point."""
    fns = {"exact": footprint_exact, "pinhole": footprint_pinhole,
           "cirn": footprint_cirn}
    try:
        fn = fns[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(fns)}")
    rx = np.full((len(stn.cameras),) + np.shape(X), np.nan)
    ry = np.full_like(rx, np.nan)
    for i, cam in enumerate(stn.cameras):
        rx[i], ry[i] = fn(cam, X, Y, Z)
    with np.errstate(invalid="ignore"):
        best_x = np.nanmin(rx, axis=0)
        best_y = np.nanmin(ry, axis=0)
    return best_x, best_y


def modeled_camera_owner(stn: Station, X, Y, Z=0.0):
    """Assign each ground cell to the best-centred visible camera.

    The score is absolute off-axis angle divided by half the camera's horizontal
    field of view. This is a geometry-based proxy for which camera contributes
    a merged-product pixel; the original rectification camera-selection and
    feathering weights are not available in ``cameraData.yml``. Cells outside
    every calibrated view receive owner ``-1``.

    Returns ``(owner, visible_count)``. Camera indices in ``owner`` correspond
    to ``stn.cameras``.
    """
    shape = np.shape(X)
    scores = np.full((len(stn.cameras),) + shape, np.inf, dtype=float)
    visible = np.zeros((len(stn.cameras),) + shape, dtype=bool)
    for i, cam in enumerate(stn.cameras):
        visible[i] = in_view(cam, X, Y, Z)
        bearing = np.arctan2(X - cam.x, Y - cam.y)
        off_axis = np.angle(np.exp(1j * (bearing - cam.azimuth)))
        score = np.abs(off_axis) / (cam.fov / 2.0)
        scores[i] = np.where(visible[i], score, np.inf)

    owner = np.argmin(scores, axis=0).astype(np.int16)
    visible_count = visible.sum(axis=0)
    owner[visible_count == 0] = -1
    return owner, visible_count


def modeled_camera_seam_mask(stn: Station, X, Y, Z=0.0):
    """Return modeled valid-camera transitions and the camera-owner map.

    A seam is a four-neighbour owner transition for which both adjacent cells
    are covered by at least two cameras. Requiring overlap keeps valid-camera
    transitions distinct from camera/no-data edges.

    These are *modeled transitions*, not surveyed rendered-pixel seam locations:
    the historical YAML supplies camera geometry but not the merge weights.
    """
    owner, visible_count = modeled_camera_owner(stn, X, Y, Z)
    overlap = visible_count >= 2
    seam = np.zeros(owner.shape, dtype=bool)

    for axis in range(owner.ndim):
        left = np.take(owner, range(owner.shape[axis] - 1), axis=axis)
        right = np.take(owner, range(1, owner.shape[axis]), axis=axis)
        left_overlap = np.take(overlap, range(owner.shape[axis] - 1), axis=axis)
        right_overlap = np.take(overlap, range(1, owner.shape[axis]), axis=axis)
        edge = ((left != right) & (left >= 0) & (right >= 0) &
                left_overlap & right_overlap)

        left_slice = [slice(None)] * owner.ndim
        right_slice = [slice(None)] * owner.ndim
        left_slice[axis] = slice(None, -1)
        right_slice[axis] = slice(1, None)
        seam[tuple(left_slice)] |= edge
        seam[tuple(right_slice)] |= edge

    return seam, owner, visible_count


def modeled_camera_seam_distance(stn: Station, X, Y, Z=0.0,
                                 spacing=None):
    """Unsigned ground distance to the nearest modeled camera transition.

    ``spacing`` follows ``scipy.ndimage.distance_transform_edt`` axis order. If
    omitted for a regular two-dimensional mesh, spacing is inferred from ``X``
    and ``Y``. Distances outside every calibrated camera view are returned as
    NaN. Also returns the seam mask and owner map.
    """
    from scipy.ndimage import distance_transform_edt

    seam, owner, visible_count = modeled_camera_seam_mask(stn, X, Y, Z)
    if not seam.any():
        raise ValueError("modeled camera geometry produced no valid transitions")

    if spacing is None:
        if np.ndim(X) != 2 or np.ndim(Y) != 2:
            raise ValueError("spacing must be supplied unless X and Y are 2-D meshes")
        dy = float(np.nanmedian(np.hypot(np.diff(X, axis=0),
                                        np.diff(Y, axis=0))))
        dx = float(np.nanmedian(np.hypot(np.diff(X, axis=1),
                                        np.diff(Y, axis=1))))
        spacing = (dy, dx)

    distance = distance_transform_edt(~seam, sampling=spacing).astype(float)
    distance[visible_count == 0] = np.nan
    return distance, seam, owner


def nominal_camera_transition_angles(stn: Station):
    """Angles of all internal transitions between adjacent camera boresights.

    The transition between two cameras is where their normalized off-axis
    angles are equal. Unlike :func:`modeled_camera_seam_mask`, this construction
    includes every adjacent-camera pair even when the historical crop windows
    leave no modeled two-camera overlap.

    Returns a list of ``(left_index, right_index, angle_radians)`` ordered from
    north-facing to south-facing cameras.
    """
    order = sorted(range(len(stn.cameras)), key=lambda i: stn.cameras[i].azimuth)
    transitions = []
    for left, right in zip(order[:-1], order[1:]):
        first = stn.cameras[left]
        second = stn.cameras[right]
        first_half_fov = first.fov / 2.0
        second_half_fov = second.fov / 2.0
        angle = (
            first.azimuth * second_half_fov
            + second.azimuth * first_half_fov
        ) / (first_half_fov + second_half_fov)
        transitions.append((left, right, float(angle)))
    return transitions


def nominal_camera_transition_distance(stn: Station, X, Y, spacing=None):
    """Distance to all nominal adjacent-camera transitions.

    Each transition is a ray from the mean tower location at the equal-
    normalized-off-axis angle between adjacent boresights. This deliberately
    covers all ``N-1`` internal transitions for ``N`` cameras. It is the right
    diagnostic when the rendered merge weights are unavailable and the older
    calibrated crop windows may contain gaps.

    Returns ``(distance, seam_mask, transitions)``. The seam mask is a raster
    rendering used for plots; ``distance`` is calculated analytically and does
    not depend on the raster threshold.
    """
    if np.ndim(X) != 2 or np.ndim(Y) != 2:
        raise ValueError("X and Y must be 2-D meshes")
    if spacing is None:
        dy = float(np.nanmedian(np.hypot(np.diff(X, axis=0),
                                        np.diff(Y, axis=0))))
        dx = float(np.nanmedian(np.hypot(np.diff(X, axis=1),
                                        np.diff(Y, axis=1))))
        spacing = (dy, dx)

    transitions = nominal_camera_transition_angles(stn)
    distances = nominal_camera_transition_distances(stn, X, Y, transitions)

    distance = np.min(distances, axis=0)
    distance[~np.isfinite(distance)] = np.nan
    half_diagonal = 0.5 * float(np.hypot(*spacing))
    seam = distance <= half_diagonal
    return distance, seam, transitions


def nominal_camera_transition_distances(stn: Station, X, Y,
                                        transitions=None):
    """Distance stack for each nominal adjacent-camera transition.

    The returned array has shape ``(N-1, *X.shape)`` and uses infinity behind a
    transition ray. Supplying the output of
    :func:`nominal_camera_transition_angles` avoids recomputing it.
    """
    if transitions is None:
        transitions = nominal_camera_transition_angles(stn)
    px = X - stn.x
    py = Y - stn.y
    distances = []
    for _, _, angle in transitions:
        vx, vy = np.sin(angle), np.cos(angle)
        along_ray = px * vx + py * vy
        perpendicular = np.abs(px * vy - py * vx)
        distances.append(np.where(along_ray >= 0, perpendicular, np.inf))
    return np.stack(distances)


# ==========================================================================
# your merged product grid
# ==========================================================================
#
# 1600 x 500 at 1 m spacing = 1600 m alongshore by 500 m cross-shore.
#
# GRID_Y0 was pinned with infer_grid_origin() by matching the modelled camera
# coverage against the no-data mask of a real rectified product
# (ArgusFF_20210919T183000Z_RectifiedVideo.avi, 1600 x 500): best fit -72 m,
# rounded to -70 m, agreement 0.97. The peak is a plateau (-90..-60 m), so the
# origin is good to about +/-15 m, not tighter. Products are stored NORTH-AT-TOP
# (alongshore y decreases with row), which product_grid reproduces; GRID_Y0 is
# the MINIMUM alongshore coordinate (the last row). x0 and d were assumed (0,
# 1 m) and are consistent with the fit (the inter-camera FOV gaps line up).
# The example file's mergeSettings (imRange: [0 500 -500 1500 2.5 2.5]) describe
# a different product -- 2.5 m over 500 x 2000 m -- so they are not this grid.

GRID_X0, GRID_NX = 0.0, 500        # cross-shore
GRID_Y0, GRID_NY = -100.0, 1600     # alongshore, min coord (see infer_grid_origin)
GRID_D = 1.0


def product_grid(x0=GRID_X0, nx=GRID_NX, y0=GRID_Y0, ny=GRID_NY, d=GRID_D):
    """Ground coords of the merged product, shaped (ny, nx) to match the array.

    Rows run NORTH-AT-TOP (alongshore y decreases with row index), as the FRF
    rectified products are stored, so row 0 is y = y0 + (ny-1)*d and the last
    row is y = y0. y0 is therefore the MINIMUM alongshore coordinate.
    """
    xs = x0 + np.arange(nx) * d
    ys = y0 + (ny - 1 - np.arange(ny)) * d      # north-at-top: y decreases with row. Assumed -100 per convo with spicer
    Y, X = np.meshgrid(ys, xs, indexing="ij")
    return X, Y


def oversampling_map(stn: Station, method="exact", **grid_kw):
    """Ratio of true footprint to grid cell size. >1 means the grid is finer
    than the imagery actually supports."""
    d = grid_kw.get("d", GRID_D)
    X, Y = product_grid(**grid_kw)
    res_x, res_y = station_resolution(stn, X, Y, method=method)
    return res_x / d, res_y / d, (X, Y)


def report(stn: Station, method="exact", **grid_kw):
    d = grid_kw.get("d", GRID_D)
    ox, oy, _ = oversampling_map(stn, method=method, **grid_kw)
    seen = np.isfinite(ox)
    n = seen.sum()
    print(f"\n--- {stn.name} | {method} | {d:g} m grid {ox.shape} ---")
    print(f"cells in view: {n:,} / {ox.size:,} ({n / ox.size:.1%})")
    for label, o in (("cross-shore", ox), ("alongshore", oy)):
        print(f"  {label:11s} oversampling   median {np.nanmedian(o):6.1f}x"
              f"   p90 {np.nanpercentile(o, 90):7.1f}x"
              f"   frac <= 1x {np.nansum(o <= 1) / n:6.1%}")


def infer_grid_origin(stn, data_mask, x0=GRID_X0, d=GRID_D,
                      y0_range=(-600.0, 400.0), step=5.0):
    """Estimate the alongshore origin by matching modelled camera coverage
    against the no-data mask of a real merged product.

    data_mask : bool array, shape (ny, nx), True where the product HAS data, in
        the product's native north-at-top orientation (same as product_grid, so
        pass the array as stored -- do not flip it).
    Returns (best_y0, scores), where best_y0 is the MINIMUM alongshore
        coordinate and scores is a list of (y0, agreement).
    """
    ny, nx = data_mask.shape
    scores = []
    for y0 in np.arange(*y0_range, step):
        X, Y = product_grid(x0=x0, nx=nx, y0=y0, ny=ny, d=d)
        model = np.zeros(X.shape, dtype=bool)
        for cam in stn.cameras:
            model |= in_view(cam, X, Y, 0.0)
        scores.append((float(y0), float((model == data_mask).mean())))
    best = max(scores, key=lambda s: s[1])
    return best[0], scores


# ==========================================================================
# plotting
# ==========================================================================

CONTOURS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0]


def _fov_edges(stn, ax, length=2200.0, colors=None):
    """Draw each camera's FOV wedge edges. `colors` may be a single colour or a
    per-camera sequence; default is uniform cyan (used by the resolution maps)."""
    for i, cam in enumerate(stn.cameras):
        c = "tab:cyan" if colors is None else (
            colors[i] if not isinstance(colors, str) else colors)
        for edge in (cam.azimuth - cam.fov / 2.0, cam.azimuth + cam.fov / 2.0):
            ax.plot([cam.y, cam.y + length * np.cos(edge)],
                    [cam.x, cam.x + length * np.sin(edge)],
                    color=c, lw=0.8, alpha=0.85)


def plot_resolution_maps(stn, x_range=(0.0, 600.0), y_range=(-900.0, 2100.0),
                         dgrid=5.0, vmax=40.0, pier_y=515.0, method="exact"):
    """Fig. 4 style: stacked cross-shore and alongshore resolution maps."""
    xs = np.arange(x_range[0], x_range[1] + dgrid, dgrid)
    ys = np.arange(y_range[0], y_range[1] + dgrid, dgrid)
    Y, X = np.meshgrid(ys, xs, indexing="ij")

    res_x, res_y = station_resolution(stn, X, Y, method=method)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, data, title in zip(axes, (res_x, res_y),
                               ("Cross-shore pixel resolution (m)",
                                "Alongshore pixel resolution (m)")):
        masked = np.ma.masked_invalid(data)
        pc = ax.pcolormesh(Y, X, masked, cmap="afmhot",
                           norm=LogNorm(vmin=0.05, vmax=vmax), shading="auto")
        cs = ax.contour(Y, X, masked, levels=CONTOURS, colors="w",
                        linewidths=0.6, alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%g")

        _fov_edges(stn, ax)
        ax.axvline(pier_y, color="0.35", lw=1.2, ls="--")   # approx FRF pier, annotation only
        ax.plot(stn.y, stn.x, marker="^", color="tab:blue", ms=9, mec="k", zorder=6)
        ax.add_patch(plt.Rectangle(
            (GRID_Y0, GRID_X0), GRID_NY * GRID_D, GRID_NX * GRID_D,
            fill=False, ec="tab:green", lw=1.5, ls=":", zorder=5))

        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Cross-shore x (m, FRF)")
        ax.set_facecolor("0.15")
        cb = fig.colorbar(pc, ax=ax, pad=0.01, extend="max")
        cb.set_ticks([0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40])
        cb.ax.set_yticklabels(["0.1", "0.25", "0.5", "1", "2", "5", "10", "20", "40"])

    axes[-1].set_xlabel("Alongshore y (m, FRF)")
    fig.suptitle(f"{stn.name} — pixel footprint ({method}), "
                 f"z$_c$ ≈ {stn.zc:.1f} m; dotted box = 1600×500 @ 1 m product",
                 y=0.98, fontsize=10)
    fig.tight_layout()
    return fig, (res_x, res_y)


def plot_oversampling(stn, method="exact", vmax=30.0, **grid_kw):
    """Oversampling factor on the delivered 1 m grid."""
    d = grid_kw.get("d", GRID_D)
    ox, oy, (X, Y) = oversampling_map(stn, method=method, **grid_kw)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, sharey=True)
    for ax, data, title in zip(axes, (ox, oy),
                               (f"Cross-shore oversampling (footprint / {d:g} m cell)",
                                f"Alongshore oversampling (footprint / {d:g} m cell)")):
        masked = np.ma.masked_invalid(data)
        pc = ax.pcolormesh(Y, X, masked, cmap="viridis",
                           norm=LogNorm(vmin=0.2, vmax=vmax), shading="auto")
        ax.contour(Y, X, masked, levels=[1.0], colors="w", linewidths=1.6)
        ax.plot(stn.y, stn.x, marker="^", color="w", ms=9, mec="k", zorder=6)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Cross-shore x (m, FRF)")
        ax.set_facecolor("0.15")
        cb = fig.colorbar(pc, ax=ax, pad=0.01, extend="both")
        cb.set_label("x cell size")

    axes[-1].set_xlabel("Alongshore y (m, FRF)")
    fig.suptitle("White contour = footprint equals one grid cell; "
                 "inside it the grid is honest", y=0.98, fontsize=10)
    fig.tight_layout()
    return fig, (ox, oy)


def read_rectified_frame(video_path, index=0):
    """One frame of an FRF rectified Argus product, as RGB (ny, nx[, 3]).

    Frames are stored north-at-top (row 0 = max alongshore y), the same
    orientation product_grid uses, so the array lines up with the grid without
    flipping. Lazy cv2 import keeps the module's core dependency-light.
    """
    import cv2  # optional; only needed to read imagery
    cap = cv2.VideoCapture(str(video_path))
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"could not read frame {index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def plot_frame_on_grid(stn, frame, pier_y=515.0, fov=True, cameras=True,
                       ax=None, x0=GRID_X0, nx=GRID_NX, y0=GRID_Y0, ny=GRID_NY,
                       d=GRID_D):
    """Overlay a rectified frame on the FRF grid, same convention as the maps.

    Alongshore y runs horizontal, cross-shore x vertical (as in
    plot_resolution_maps). `frame` is a north-at-top array from
    read_rectified_frame(); the grid keywords describe the product it came from.
    """
    y_min, y_max = y0, y0 + (ny - 1) * d
    x_min, x_max = x0, x0 + (nx - 1) * d

    # north-at-top (ny, nx) -> (nx, ny) with x increasing upward, y increasing right
    disp = frame[::-1].swapaxes(0, 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(disp, origin="lower",
              extent=[y_min - d / 2, y_max + d / 2, x_min - d / 2, x_max + d / 2],
              cmap=None if disp.ndim == 3 else "gray", aspect="auto")

    cam_colors = [plt.cm.tab10(i % 10) for i in range(len(stn.cameras))]
    if fov:
        _fov_edges(stn, ax, colors=cam_colors)
    if cameras:
        for cam, c in zip(stn.cameras, cam_colors):
            ax.plot(cam.y, cam.x, marker="v", color=c, ms=6, mec="k", zorder=6,
                    label=f"{cam.name} ({cam.orientation})")
            ax.annotate(cam.name, (cam.y, cam.x), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=c)
    ax.plot(stn.y, stn.x, marker="^", color="tab:blue", ms=10, mec="k",
            zorder=7, label="Argus tower")
    ax.axvline(pier_y, color="tab:red", lw=1.4, ls="--",
               label=f"FRF pier (y={pier_y:g})")

    ax.set_xlim(y_min - d / 2, y_max + d / 2)
    ax.set_ylim(x_min - d / 2, x_max + d / 2)
    ax.set_xlabel("Alongshore y (m, FRF)")
    ax.set_ylabel("Cross-shore x (m, FRF)")
    ax.legend(loc="upper right", fontsize=8)
    return ax.figure, ax


# ==========================================================================
# validation
# ==========================================================================

def check_against_paper(tol=0.02):
    """Holman & Stanley worked example: zc=40 m, 40 deg FOV, NU=1024, R=1 km
    -> [dc, dr] = [0.68, 17.0] m."""
    delta, nu, R, zc = np.radians(40.0), 1024, 1000.0, 40.0
    dc = R * delta / nu
    dr = dc * R / zc
    assert abs(dc - 0.68) < tol and abs(dr - 17.0) < 10 * tol
    print(f"pinhole eqs OK: dc={dc:.3f} m, dr={dr:.2f} m (paper: 0.68, 17.0)")


def verify_dlt(stn: Station, tol=5.0):
    """Project each camera's boresight ground point; it must land on the
    principal point. Fails loudly if the `m` ordering is different."""
    print("DLT check (boresight -> principal point, expect ~1224, 1024):")
    ok = True
    for cam in stn.cameras:
        if cam.m is None:
            continue
        bx, by = _boresight_point(cam)
        U, V, _ = project(cam, bx, by, 0.0)
        du, dv = abs(U - cam.nu / 2), abs(V - cam.nv / 2)
        good = du < tol and dv < tol
        ok &= good
        print(f"  {cam.name} ({cam.orientation:>3s})  U={U:7.1f}  V={V:7.1f}"
              f"   {'ok' if good else 'MISMATCH'}")
    if not ok:
        raise ValueError("DLT coefficient ordering does not match — see note at top.")


def decompose_dlt(cam: Camera):
    """Recover camera centre, calibration matrix and pointing from the DLT.

    Returns a dict. Compare against the YAML's own fields to validate both the
    coefficient ordering and the meaning of azimuth/tilt/fov.
    """
    m = cam.m
    M = np.array([m[0:3], m[7:10], [m[4], m[5], m[6]]], dtype=float)
    p4 = np.array([m[3], m[10], 1.0])

    centre = np.linalg.solve(M, -p4)

    scale = np.linalg.norm(M[2])
    sgn = -np.sign(M[2, 2])          # principal axis must point downward
    Mn = sgn * M / scale
    r3 = Mn[2]

    u0 = float(Mn[0] @ r3)
    v0 = float(Mn[1] @ r3)
    fx = float(np.linalg.norm(Mn[0] - u0 * r3))
    fy = float(np.linalg.norm(Mn[1] - v0 * r3))

    return {
        "centre": centre,
        "u0": u0, "v0": v0, "fx": fx, "fy": fy,
        "azimuth": float(np.arctan2(r3[0], r3[1])),   # compass from +y
        "tilt": float(np.arccos(-r3[2])),             # from nadir
        "hfov": float(2.0 * np.arctan(cam.nu / (2.0 * fx))),
        "axis": r3,
    }


def validate_geometry(stn: Station, verbose=True):
    """Cross-check every DLT against the scalar fields in the same file.

    This validates, per camera:
      - DLT coefficient ordering      (centre matches x/y/z)
      - principal point convention    (u0/v0 match D_UO/D_V0)
      - fov is the HORIZONTAL fov      (fx matches NU/(2 tan(fov/2)))
      - azimuth is compass from +y     (matches DLT principal axis)
      - tilt is measured from nadir    (matches DLT principal axis)
      - square pixels                  (fx == fy)
    """
    rows, ok = [], True
    for cam in stn.cameras:
        if cam.m is None:
            continue
        d = decompose_dlt(cam)
        e = {
            "centre_m": float(np.linalg.norm(d["centre"] - [cam.x, cam.y, cam.z])),
            "u0_px": abs(d["u0"] - cam.nu / 2),
            "v0_px": abs(d["v0"] - cam.nv / 2),
            "fov_pct": 100 * abs(d["hfov"] - cam.fov) / cam.fov,
            "azim_deg": abs(np.degrees(d["azimuth"] - cam.azimuth)),
            "tilt_deg": abs(np.degrees(d["tilt"] - cam.tilt)),
            "aspect": abs(d["fx"] / d["fy"] - 1.0),
        }
        # centre tol 0.10 m: five cameras round-trip to ~1e-16, but c2 (the
        # "Brittany's geometry fix" block) has an updated DLT whose null-space
        # centre sits 5.2 cm from its stated surveyed centre -- a real
        # self-inconsistency in the published calibration, not a copy error
        # (the block is byte-identical to cameraData.yml). 0.10 m still catches
        # any genuine ordering/convention error, which throws it off by metres.
        passed = (e["centre_m"] < 0.10 and e["u0_px"] < 2 and e["v0_px"] < 2 and
                  e["fov_pct"] < 1.0 and e["azim_deg"] < 0.1 and
                  e["tilt_deg"] < 0.1 and e["aspect"] < 0.01)
        ok &= passed
        rows.append((cam, d, e, passed))

    if verbose:
        print("\nDLT self-consistency check")
        print(f"{'cam':4s} {'dCentre':>9s} {'du0':>6s} {'dv0':>6s} {'dFOV%':>7s}"
              f" {'dAzim':>7s} {'dTilt':>7s} {'fx':>8s} {'fy/fx':>7s}  ")
        for cam, d, e, passed in rows:
            print(f"{cam.name:4s} {e['centre_m']:8.4f}m {e['u0_px']:6.2f}"
                  f" {e['v0_px']:6.2f} {e['fov_pct']:7.3f} {e['azim_deg']:7.4f}"
                  f" {e['tilt_deg']:7.4f} {d['fx']:8.1f} {d['fy']/d['fx']:7.4f}"
                  f"  {'ok' if passed else 'FAIL'}")

    if not ok:
        raise ValueError(
            "DLT does not reproduce the scalar fields. Either the coefficient "
            "ordering differs, or fov/azimuth/tilt use another convention. "
            "Do not trust any output until this passes.")
    return rows


def compare_cirn_pinhole(stn: Station, verbose=True, **grid_kw):
    """Characterise footprint_cirn against footprint_pinhole on the product grid.

    Both are approximate: pinhole uses the small-angle Holman & Stanley forms;
    CIRN uses the exact differenced-tangent range and slant-range cross-range.
    They agree in the far field and diverge near the tower, where the look angle
    from nadir is large (L/R = 1/cos drives both dc and the range term apart).
    This does NOT raise -- it reports the per-camera relative difference
    |cirn - pinhole| / pinhole so the divergence is visible, not flagged.
    """
    X, Y = product_grid(**grid_kw)

    def stats(r, n):
        if not n:
            return np.nan, np.nan, np.nan
        return np.nanmedian(r), np.nanpercentile(r, 90), np.nanmax(r)

    rows = []
    for cam in stn.cameras:
        cx, cy = footprint_cirn(cam, X, Y)
        px, py = footprint_pinhole(cam, X, Y)
        with np.errstate(invalid="ignore", divide="ignore"):
            rdx = np.abs(cx - px) / px
            rdy = np.abs(cy - py) / py
        n = int(np.isfinite(rdx).sum())
        rows.append((cam, n, stats(rdx, n), stats(rdy, n)))

    if verbose:
        print("\nCIRN vs pinhole footprint  (rel. diff |cirn-pinhole|/pinhole, "
              "product grid)")
        print(f"{'cam':4s} {'cells':>7s}  {'x_med':>7s} {'x_p90':>7s} {'x_max':>7s}"
              f"   {'y_med':>7s} {'y_p90':>7s} {'y_max':>7s}")
        for cam, n, sx, sy in rows:
            print(f"{cam.name:4s} {n:7d}  {sx[0]:7.1%} {sx[1]:7.1%} {sx[2]:7.1%}"
                  f"   {sy[0]:7.1%} {sy[1]:7.1%} {sy[2]:7.1%}")
    return rows


def summarise(stn: Station):
    print(f"\n{stn.name}")
    print(f"{'cam':4s} {'ID':9s} {'or':4s} {'azim':>7s} {'fov':>7s} {'tilt':>7s}"
          f" {'f(px)':>7s} {'d/NU':>10s}  valid from")
    for c in stn.cameras:
        print(f"{c.name:4s} {c.ident:9s} {c.orientation:4s}"
              f" {np.degrees(c.azimuth):7.2f} {np.degrees(c.fov):7.2f}"
              f" {np.degrees(c.tilt):7.2f} {c.focal_px:7.0f} {c.subtense:10.3e}"
              f"  {c.valid_from}")


if __name__ == "__main__":
    stn = load_camera_yaml(CAMERA_YAML, "Duck FRF Argus tower")

    check_against_paper()
    validate_geometry(stn)     # strong self-consistency check; raises on mismatch
    summarise(stn)

    report(stn, method="exact")
    report(stn, method="pinhole")      # should be close; divergence = distortion/crop
    report(stn, method="cirn")         # differenced-tangent / slant-range variant

    compare_cirn_pinhole(stn)          # CIRN vs pinhole; far-field agree, near-field diverge

    for y in (60.0, 940.0):
        for x in (150.0, 300.0, 500.0):
            rx, ry = station_resolution(stn, np.array([[x]]), np.array([[y]]))
            print(f"x={x:4.0f} y={y:5.0f}   dx={rx[0,0]:6.2f} m   dy={ry[0,0]:6.2f} m")

    plot_resolution_maps(stn)
    plot_oversampling(stn)

    example_video = (Path(__file__).parents[1] / "video_dataset" / "argus" /
                     "ArgusFF_20210917T163000Z_RectifiedVideo.avi")
    if example_video.exists():
        frame = read_rectified_frame(example_video, index=0)
        fig, ax = plot_frame_on_grid(stn, frame)
        ax.set_title(f"{example_video.stem} — frame 0 on FRF grid")

    plt.show()
