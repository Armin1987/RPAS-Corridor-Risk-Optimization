"""Precomputed risk tables indexed by wind state.

speed bins s = 0..12 m/s (step 1; hours with W > 12 m/s make every lane unusable)
direction bins d = 0..11 (wind blowing TOWARD 30*d degrees)
C_table[k][c]     : (13, 12, n_lanes)  expected fatalities per crash, mean over lane points
Cmax_table[k][c]  : same, maximum over lane points (reported, not constrained)
K_table[k][c]     : (13, n_lanes, n_lanes) collision coefficients (depend on speed only)
"""
from __future__ import annotations
import numpy as np
from .geometry import lanes, lane_points
from . import kernels
from .navmodels import kappa_matrix

SPEEDS = np.arange(0, 13)
DIRS = np.arange(0, 360, 30)


def harm_kernels(rng, platform, heading_alt_pairs, cell, ps, n, harm, r_uas):
    """Kernel dictionary keyed by (heading, alt, speed bin, dir bin)."""
    out, stats = {}, []
    for (h, a) in heading_alt_pairs:
        for s in SPEEDS:
            for di, d in enumerate(DIRS):
                K, st = kernels.kernel(rng, platform, h, float(s), float(d), a, cell, ps, n=n,
                                       r_uas=r_uas, harm=harm)
                out[(h, a, int(s), di)] = K
                st.update(heading=h, alt=a, speed=int(s), dir_toward=int(d)); stats.append(st)
    return out, stats


def consequence_tables(corridors, menu, kern, rho, cell, r0, c0):
    C, Cmax = [], []
    for cor in corridors:
        Ck, Cmk = [], []
        for cfg in menu:
            ln = lanes(cor, cfg) if cfg.n_lanes else []
            tab = np.zeros((len(SPEEDS), len(DIRS), len(ln))); tmx = np.zeros_like(tab)
            for li, L in enumerate(ln):
                xs, ys = lane_points(cor, L["offset"], cell)
                pts = list(zip((np.floor(ys / cell) - r0).astype(int), (np.floor(xs / cell) - c0).astype(int)))
                for s in SPEEDS:
                    for di in range(len(DIRS)):
                        K = kern[(L["heading"], L["alt"], int(s), di)]
                        m, mx = kernels.consequence_along(pts, K, rho)
                        tab[s, di, li] = m; tmx[s, di, li] = mx
            Ck.append(tab); Cmk.append(tmx)
        C.append(Ck); Cmax.append(Cmk)
    return C, Cmax


def kappa_tables(corridors, menu, nav, kin, gamma):
    out = []
    for _ in corridors:
        row = []
        for cfg in menu:
            row.append(np.stack([kappa_matrix(cfg, nav, kin, float(s), gamma) for s in SPEEDS])
                       if cfg.n_lanes else np.zeros((len(SPEEDS), 0, 0)))
        out.append(row)
    return out


def bins(speed: np.ndarray, from_dir: np.ndarray):
    """Map hourly speed and meteorological FROM-direction to table bins.
    Returns (speed_bin, dir_bin, usable_mask)."""
    speed = np.asarray(speed, float); from_dir = np.asarray(from_dir, float)
    known = ~np.isnan(speed) & ~np.isnan(from_dir)
    usable = known & (np.nan_to_num(speed, nan=np.inf) <= 12.0)   # unknown wind -> unusable (explicit)
    sb = np.clip(np.rint(np.nan_to_num(speed, nan=12.0)), 0, 12).astype(int)
    toward = (np.nan_to_num(from_dir, nan=0.0) + 180.0) % 360.0
    db = (np.rint(toward / 30.0).astype(int)) % 12
    return sb, db, usable
