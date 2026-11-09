"""Map (configuration, wind state) -> (U, C, kappa) arrays for the optimisation model.

Wind enters in three places, each a separate, testable assumption:
  1. operating limit: lane unusable if wind at flight level > w_max (platform rating)
  2. navigation/flight-technical error: core Laplace scale a1(W) = a1_0 + beta_a * W
     (beta_a is an ASSUMED research parameter until a sourced FTE-wind relation is found)
  3. ballistic drift: per-crash consequence C(W) from a table computed with ballistic.py +
     ground.py and linearly interpolated in wind speed (direction handled per lane).
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from ..risk import reich
from .two_stage import ConfigData


@dataclass
class CorridorConfig:
    name: str
    n_lat: int          # lateral lanes
    n_vert: int         # altitude layers
    S_y: float          # lateral separation (m)
    S_z: float          # vertical separation (m)
    S_x: float          # minimum in-trail spacing (m)
    fixed_cost: float


def lane_neighbours(cfg: CorridorConfig):
    """Lanes indexed l = layer * n_lat + lateral position.  Returns list of
    (l, l2, kind) with kind in {'lat', 'vert'} for directly adjacent lanes."""
    out = []
    for v in range(cfg.n_vert):
        for h in range(cfg.n_lat):
            l = v * cfg.n_lat + h
            if h + 1 < cfg.n_lat:
                out += [(l, l + 1, "lat"), (l + 1, l, "lat")]
            if v + 1 < cfg.n_vert:
                out += [(l, l + cfg.n_lat, "vert"), (l + cfg.n_lat, l, "vert")]
    return out


def config_arrays(cfg: CorridorConfig, wind: np.ndarray, geo: reich.Geometry,
                  nav: dict, C_of_wind, w_max: float) -> ConfigData:
    """wind: (T, W) wind speed at flight level (m/s).  C_of_wind: callable(W)->fatalities
    per crash (vectorised).  nav: a1_0, beta_a, a2, alpha, a1_z, a1_x."""
    T, W = wind.shape
    n = cfg.n_lat * cfg.n_vert
    if n == 0:
        z0 = np.zeros((T, W, 0))
        return ConfigData(0, cfg.fixed_cost, z0, z0, np.zeros((T, W, 0, 0)))
    nb = lane_neighbours(cfg)
    U = np.zeros((T, W, n)); C = np.zeros((T, W, n)); K = np.zeros((T, W, n, n))
    cap = 3600.0 * geo.V / cfg.S_x
    for t in range(T):
        for w in range(W):
            ws = wind[t, w]
            if ws > w_max:
                continue
            a1 = nav["a1_0"] + nav["beta_a"] * ws
            ey = reich.LaplaceMixture(a1, nav["a2"], nav["alpha"])
            ex = reich.LaplaceMixture(nav["a1_x"] + nav["beta_a"] * ws, nav["a2"], nav["alpha"])
            ez = reich.LaplaceMixture(nav["a1_z"], nav["a2_z"], nav["alpha"])
            k_lat = reich.kappa_lateral(cfg.S_y, geo, ey, ez)
            k_vert = reich.kappa_vertical(cfg.S_z, geo, ey, ez)
            k_self = reich.kappa_intrail(cfg.S_x, geo, ex, ey, ez)
            U[t, w, :] = cap
            C[t, w, :] = C_of_wind(ws)
            for l in range(n):
                K[t, w, l, l] = k_self
            for l, l2, kind in nb:
                K[t, w, l, l2] = k_lat if kind == "lat" else k_vert
    return ConfigData(n, cfg.fixed_cost, U, C, K)
