"""Estimate the optimal Booij celerity gamma from observed speeds and depths.

Inversion model (see 04_bathy_planview.py):
    c^2 = g * h * (1 + gamma/2)   ->   h_inv = c^2 / (g * (1 + gamma/2))
where c^2 = mean_squared_speed and h is the TOTAL water depth (depth + setup).

"Optimal" gamma = the single value that makes the inverted depth best match the
survey total depth, over all valid data points, for several shallow-depth cutoffs.
Prints only; no plotting.
"""

from pathlib import Path

import numpy as np
import xarray as xr
import dunex_paths

G = 9.81
DATASET = dunex_paths.OUTPUTS_DIR / "combined_dunex_dataset_lerp.nc"
CUTOFFS = [0.1, 0.5, 1.0]   # minimum survey total depth [m] to include


def rmse(pred, obs):
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def report(label, c2, h_true, base):
    """Print the gamma sweep for one depth definition (with or without setup)."""
    print(f"\n=== {label} ===")
    hdr = f"{'cutoff [m]':>10} {'N points':>10} {'gamma* (LSQ)':>13} " \
          f"{'gamma median':>13} {'gamma mean':>11} {'RMSE* [m]':>10} {'RMSE@0.6 [m]':>13}"
    print(hdr)
    print('-' * len(hdr))

    for cut in CUTOFFS:
        m = base & (h_true >= cut)
        x = c2[m] / G          # = h * (1 + gamma/2)
        h = h_true[m]

        # LSQ-optimal: minimize sum( k*x - h )^2 with k = 1/(1+gamma/2), h_inv = k*x.
        k_opt = np.sum(x * h) / np.sum(x * x)
        gamma_lsq = 2.0 * (1.0 / k_opt - 1.0)

        # Per-point gamma: c^2 = g*h*(1+gamma/2) -> gamma_i = 2*(x/h - 1).
        gamma_pt = 2.0 * (x / h - 1.0)
        gamma_med = float(np.median(gamma_pt))
        gamma_mn = float(np.mean(gamma_pt))

        h_opt = x / (1.0 + gamma_lsq / 2.0)
        h_06 = x / (1.0 + 0.6 / 2.0)

        print(f"{cut:>10.1f} {m.sum():>10,} {gamma_lsq:>13.3f} "
              f"{gamma_med:>13.3f} {gamma_mn:>11.3f} "
              f"{rmse(h_opt, h):>10.3f} {rmse(h_06, h):>13.3f}")


def main():
    ds = xr.open_dataset(DATASET)

    c2 = ds.mean_squared_speed.values.ravel()          # observed c^2 [m^2/s^2]
    h_still = ds.depth.values.ravel()                  # still-water depth [m]
    h_total = (ds.depth + ds.setup).values.ravel()     # total depth (incl. setup) [m]

    # Base validity: finite speed/depths, positive speed, positive still-water depth.
    base = (np.isfinite(c2) & np.isfinite(h_still) & np.isfinite(h_total)
            & (c2 > 0) & (h_still > 0))

    print(f"Dataset: {Path(DATASET).name}")
    print(f"Total grid*time points: {c2.size:,}   valid (finite, c2>0, h>0): {base.sum():,}")
    print("Reference gamma currently used in figure: 0.60")

    report("WITH setup   (h = depth + setup)", c2, h_total, base)
    report("WITHOUT setup (h = depth)",        c2, h_still, base)

    print("\nNotes:")
    print("  gamma* (LSQ)  : minimizes RMSE between inverted and survey depth.")
    print("  gamma median  : median of per-point gamma_i = 2*(c^2/(g*h) - 1).")
    print("  RMSE*         : depth RMSE at gamma*.  RMSE@0.6: depth RMSE at gamma=0.6.")
    print("  Same valid points and cutoffs applied to both cases for comparability;")
    print("  the cutoff is applied to the depth definition being fit.")


if __name__ == '__main__':
    main()
