import numpy as np

G     = 9.81
RHO   = 1025.0
GAMMA_S = 0.6   # breaker index for significant wave height

# ── Step 1: wavenumber ────────────────────────────────────────────────────────
def solve_wavenumber(omega, h):
    """Solve ω² = gk·tanh(kh) for k via Newton-Raphson. Vectorised."""
    omega, h = np.broadcast_arrays(
        np.asarray(omega, dtype=float), np.asarray(h, dtype=float)
    )
    h = np.maximum(h, 0.01)
    k = omega**2 / G
    for _ in range(25):
        kh  = k * h
        th  = np.tanh(np.minimum(kh, 500.0))
        f   = G * k * th - omega**2
        df  = G * (th + kh * (1 - th**2))
        k  -= f / df
    return np.abs(k)

# ── Step 2: group velocity ────────────────────────────────────────────────────
def group_velocity(k, h):
    r"""cg = n · c,  where n = ½(1 + 2kh/sinh(2kh)), c = \sqrt{g*tanh(kh)/k}"""
    kh = np.minimum(k * np.maximum(h, 0.01), 500.0)
    n  = 0.5 * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))
    c  = np.sqrt(G * np.tanh(kh) / np.maximum(k, 1e-8))
    return n * c

# ── Step 5: radiation stress ──────────────────────────────────────────────────
def radiation_stress(E, k, h):
    """
    Shore-normal form: Sxx = E(2n - 1/2).

    Rather than applying Snell's law to evolve theta along each 1-D
    cross-shore profile — which is only self-consistent for perfectly
    shore-parallel bathymetry — we conserve the shore-normal energy flux
    E·cg·cos(theta) with cos(theta) fixed at its offshore value. This is
    equivalent to setting theta=0 in the oblique form E[n(1+cos²theta)-1/2],
    with the offshore angle absorbed into the flux reference boundary
    condition.
    """
    kh = np.minimum(k * np.maximum(h, 0.01), 500.0)
    n  = 0.5 * (1.0 + 2.0 * kh / np.sinh(2.0 * kh))
    return E * (2.0 * n - 0.5)

# ── Main solver ───────────────────────────────────────────────────────────────
def compute_setup_1d(h, E0, T, theta_deg=0.0):
    """
    Full radiation-stress setup for a 1-D cross-shore depth profile.

    The shore-normal energy flux F = E·cg·cos(theta) is conserved shoreward.
    Rather than evolving theta via Snell's law along each transect (only valid
    for shore-parallel contours), the offshore obliquity is projected once into
    a shore-normal-equivalent energy density E0_eff = E0·cos(theta_0), which
    carries the correct shore-normal flux E0·cg_0·cos(theta_0). Propagation is
    then treated as shore-normal (the Snell limit cos(theta)->1 nearshore), so
    shoaled energy is E = E0_eff·cg_0/cg and the radiation stress uses the
    shore-normal form Sxx = E(2n - 1/2). The directional projection therefore
    reduces the energy delivered to the surf zone for oblique waves, applied
    exactly once at the offshore boundary.

    Parameters
    ----------
    h         : 1-D array   still-water depth [m], index 0 = offshore
    E0        : float       wave energy density at h[0] [J/m²]
    T         : float       wave period [s]
    theta_deg : float       incidence angle at h[0] relative to shore normal
                            [degrees]. Projects the offshore energy onto the
                            shore normal via cos(theta_0); applied once.

    Returns
    -------
    eta : 1-D array  mean water level [m]  (+ = setup, - = setdown)
    H   : 1-D array  wave height [m]
    """
    nx    = len(h)
    omega = 2 * np.pi / T
    h_    = np.maximum(h, 0.01)

    k  = solve_wavenumber(omega, h_)
    cg = group_velocity(k, h_)

    # Project the offshore energy onto the shore normal once. Propagation is
    # then treated as shore-normal (Snell limit cos(theta)->1), so cos(theta_0)
    # does NOT reappear in the shoaling denominator below.
    E0_eff = E0 * np.cos(np.deg2rad(theta_deg))

    eta = np.zeros(nx)
    E   = np.zeros(nx)

    E[0]     = E0_eff
    Sxx_prev = radiation_stress(E[0], k[0], h_[0])

    # Shore-normal energy flux carried shoreward; reduced by breaking dissipation
    flux_ref = E0_eff * cg[0]

    for i in range(1, nx):
        denom    = max(cg[i], 1e-6)
        E_shoal  = flux_ref / denom
        # E = (1/16) ρg Hs²  →  Hs = sqrt(16 E / (ρg))
        Hs_shoal = np.sqrt(16 * E_shoal / (RHO * G))

        # total_depth = max(h[i] + eta[i - 1], 0.01)
        # Linearization: neglect the mean setup in the local total depth.
        total_depth = max(h[i], 0.01)

        if Hs_shoal > GAMMA_S * total_depth:
            # Breaking dissipates energy: the flux is no longer conserved. Set E
            # by the depth limit and carry the reduced flux E·cg forward, so a
            # wave that later de-shoals into a trough re-forms from its dissipated
            # energy rather than the full offshore flux.
            E[i] = (1.0 / 16.0) * RHO * G * (GAMMA_S * total_depth) ** 2
            flux_ref = E[i] * cg[i]
        else:
            # No breaking: shore-normal flux is conserved, so flux_ref is left
            # untouched and the wave shoals as E = flux_ref / cg.
            E[i] = E_shoal

        Sxx_curr = radiation_stress(E[i], k[i], h_[i])
        # Use still-water depth in the pressure term for the same linearization.
        d_prev   = max(h[i - 1], 0.01)
        eta[i]   = eta[i - 1] - (Sxx_curr - Sxx_prev) / (RHO * G * d_prev)
        Sxx_prev = Sxx_curr

    Hs = np.sqrt(16 * E / (RHO * G))
    return eta, Hs


def compute_setup_1_5d(E0, Tp, h, peak_direction=None, shore_normal_deg=0.0):
    """
    Full radiation-stress setup for every cross-shore profile in the dataset.

    h shape: (time, yFRF, xFRF).  xFRF=0 is onshore, increases offshore.
    Parallelises over time steps.

    The peak wave direction is used only to compute cos(theta_0) at the
    offshore boundary; it is not evolved shoreward via Snell's law.

    Parameters
    ----------
    E0              : ndarray, shape (time,)   offshore wave energy density [J/m²]
    Tp              : ndarray, shape (time,)
    h               : ndarray, shape (time, n_along, n_cross)
    peak_direction  : ndarray, shape (time,) or None
                      Peak wave direction in the same coordinate system as
                      `shore_normal_deg`. Used to compute cos(theta_0) only.
                      If None, waves are treated as shore-normal.
    shore_normal_deg : float
                      Direction of the shore normal (pointing offshore) in the
                      same convention as `peak_direction`.

    Returns
    -------
    eta : ndarray, shape (time, n_along, n_cross)
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    nt = h.shape[0]
    if peak_direction is None:
        theta_arr = np.zeros(nt)
    else:
        theta_arr = np.asarray(peak_direction, dtype=float) - shore_normal_deg
        theta_arr = (theta_arr + 180.0) % 360.0 - 180.0
        theta_arr = np.clip(theta_arr, -89.0, 89.0)

    def compute(t, E0_t, T, theta, h_t):
        n_along = h_t.shape[0]
        eta_t = np.zeros_like(h_t)
        for j in range(n_along):
            profile = h_t[j, :]        # xFRF=0 (onshore) → xFRF=500 (offshore)
            valid = np.isfinite(profile)
            if valid.sum() < 2:
                continue
            i0 = np.argmax(valid)
            i1 = len(valid) - 1 - np.argmax(valid[::-1])
            sub = profile[i0:i1 + 1]
            if not np.all(np.isfinite(sub)):
                continue
            # xFRF increases offshore → flip so offshore end is first
            eta_1d, _ = compute_setup_1d(sub[::-1], E0_t, T, theta_deg=theta)
            eta_t[j, i0:i1 + 1] = eta_1d[::-1]
        return t, eta_t

    eta = np.zeros_like(h)

    results = Parallel(n_jobs=-1)(
        delayed(compute)(t, float(E0[t]), float(Tp[t]),
                         float(theta_arr[t]), h[t])
        for t in tqdm(range(nt))
    )
    for t, eta_t in results:
        eta[t] = eta_t

    return eta
