"""Harm kernels: ballistic Monte Carlo -> cell-offset kernels that already include the
lethal area and the fatality probability for each exposure class.

For heading h, wind speed W (blowing TOWARD bearing b) and altitude H:
   K_j(dr, dc) = (1/N) sum_n 1{impact n in cell offset (dr,dc)} * A_L(gamma_n) * Pf(E_n; p_s,j)
so that the expected fatalities per crash initiated at cell (r, c) are
   sum_j sum_{dr,dc} K_j(dr, dc) * rho_j(r+dr, c+dc)        [persons/m^2 * m^2 = persons].
"""
from __future__ import annotations
import numpy as np
from .risk import ballistic, ground


def kernel(rng, platform: dict, heading_deg: float, wind_speed: float, wind_toward_deg: float,
           alt: float, cell: float, ps: dict, n: int = 4000, r_uas: float = 0.45,
           harm=(1e6, 100.0), r_person=0.3, h_person=1.8):
    p = dict(platform)
    p["h0"] = {"dist": "normal", "mean": alt, "sd": 5.0, "min": 10.0}
    rel = np.deg2rad(wind_toward_deg - heading_deg)
    inp = ballistic.sample_inputs(rng, n, p, wind_speed, rel)     # x along heading, y to the right? see below
    out = ballistic.simulate(inp, dt=0.01)
    # ballistic frame: x along heading, y = +90 deg (to the right of travel when rel=+90 deg)
    hdg = np.deg2rad(heading_deg)
    east = out["x"] * np.sin(hdg) + out["y"] * np.sin(hdg + np.pi / 2)
    north = out["x"] * np.cos(hdg) + out["y"] * np.cos(hdg + np.pi / 2)
    dc = np.floor(east / cell + 0.5).astype(int); dr = np.floor(north / cell + 0.5).astype(int)
    AL = ground.lethal_area(out["gamma"], r_uas, r_person, h_person)
    H = int(max(np.abs(dr).max(), np.abs(dc).max()))
    K = {}
    for j, psj in ps.items():
        w = AL * ground.pf_dalamagkidis(out["ke"], psj, harm[0], harm[1])
        grid = np.zeros((2 * H + 1, 2 * H + 1))
        np.add.at(grid, (dr + H, dc + H), w / n)
        K[j] = grid
    stats = {"ke_mean": float(np.mean(out["ke"])), "ke_p95": float(np.percentile(out["ke"], 95)),
             "reach_mean": float(np.mean(np.hypot(east, north))), "reach_p99": float(np.percentile(np.hypot(east, north), 99)),
             "AL_mean": float(AL.mean()), "gamma_mean_deg": float(np.degrees(out["gamma"]).mean()),
             "n": n, "H_cells": H}
    return K, stats


def consequence_along(points_rc, K: dict, rho: dict):
    """Mean expected fatalities per crash over crash-initiation cells points_rc."""
    vals = []
    nr, nc = next(iter(rho.values())).shape
    for (r, c) in points_rc:
        tot = 0.0
        for j, Kj in K.items():
            H = Kj.shape[0] // 2
            r0, r1, c0, c1 = r - H, r + H + 1, c - H, c + H + 1
            kr0, kc0 = max(0, -r0), max(0, -c0)
            kr1 = Kj.shape[0] - max(0, r1 - nr); kc1 = Kj.shape[1] - max(0, c1 - nc)
            sub = rho[j][max(r0, 0):min(r1, nr), max(c0, 0):min(c1, nc)]
            tot += float((Kj[kr0:kr1, kc0:kc1] * sub).sum())
        vals.append(tot)
    return float(np.mean(vals)), float(np.max(vals))
