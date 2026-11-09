"""Corridor geometry, configuration menu, lane layout and crash-initiation cells.

Grid convention: cell (row, col) covers y in [row*cell, (row+1)*cell), x in
[col*cell, (col+1)*cell) of the local equirectangular frame (x east, y north) used by the
pNEUMA processing (reference LAT0, LON0).  Bearings are degrees clockwise from north.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Config:
    name: str
    n_lat: int
    n_vert: int
    S_y: float      # lateral spacing between adjacent lanes (m)
    S_z: float      # vertical spacing between layers (m)
    S_x: float      # minimum in-trail spacing (m)
    fixed_cost: float

    @property
    def n_lanes(self):
        return self.n_lat * self.n_vert


@dataclass
class Corridor:
    name: str
    x0: float; y0: float; x1: float; y1: float   # centreline end points (m, local frame)
    base_alt: float                               # altitude of the lowest layer (m AGL)

    @property
    def length(self):
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    @property
    def bearing(self):
        return float((np.degrees(np.arctan2(self.x1 - self.x0, self.y1 - self.y0)) + 360) % 360)


def lanes(cor: Corridor, cfg: Config):
    """List of lanes: dict(lateral offset m (+ = right of travel), layer, altitude, heading).
    Layer 0 flies the corridor bearing; layer 1 the reverse bearing (opposite direction)."""
    out = []
    offs = (np.arange(cfg.n_lat) - (cfg.n_lat - 1) / 2) * cfg.S_y
    for v in range(cfg.n_vert):
        for h in range(cfg.n_lat):
            out.append({"offset": float(offs[h]), "layer": v, "alt": cor.base_alt + v * cfg.S_z,
                        "heading": (cor.bearing + 180.0 * v) % 360})
    return out


def lane_points(cor: Corridor, offset: float, step: float):
    """Crash-initiation points along a lane, every `step` metres (local frame)."""
    L = cor.length
    ux, uy = (cor.x1 - cor.x0) / L, (cor.y1 - cor.y0) / L
    rx, ry = uy, -ux                          # right-hand normal for a north-up frame
    s = np.arange(step / 2, L, step)
    return cor.x0 + s * ux + offset * rx, cor.y0 + s * uy + offset * ry


def pair_separations(cfg: Config):
    """Lateral and vertical separations (m) and direction relation for every lane pair."""
    n = cfg.n_lanes
    offs = (np.arange(cfg.n_lat) - (cfg.n_lat - 1) / 2) * cfg.S_y
    dy = np.zeros((n, n)); dz = np.zeros((n, n)); opp = np.zeros((n, n), bool)
    for a in range(n):
        va, ha = divmod(a, cfg.n_lat)
        for b in range(n):
            vb, hb = divmod(b, cfg.n_lat)
            dy[a, b] = abs(offs[ha] - offs[hb]); dz[a, b] = abs(va - vb) * cfg.S_z
            opp[a, b] = (va % 2) != (vb % 2)
    return dy, dz, opp


DEFAULT_MENU = [
    Config("closed", 0, 0, 0, 0, 120, 0.0),
    Config("single", 1, 1, 0, 0, 120, 1.0),
    Config("wide-2", 2, 1, 60, 0, 120, 2.0),
    Config("std-3", 3, 1, 40, 0, 120, 3.0),
    Config("dense-2x2", 2, 2, 30, 20, 120, 4.0),
    Config("dense-3x2", 3, 2, 25, 20, 120, 5.0),
]
