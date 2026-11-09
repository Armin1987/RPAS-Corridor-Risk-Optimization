"""Alternative navigation-error models (the Stage 3 uncertainty framework).

Evidence (see Supplementary S3):
  * Geister et al. (2018, ICNS, DLR): octocopter, stand-alone GPS, ground wind ~4 and ~7 m/s:
    measured lateral total-system-error RMS 1.06-6.05 m (trial 1) and 3.78-9.40 m (trial 2);
    horizontal NSE maxima up to 20.4 m; tail shape not analysed.
  * DJI M300 specification: hover accuracy +/-1.5 m horizontal, +/-0.5 m vertical (GPS),
    +/-0.1 m (RTK) - static hover, not along-track error in wind.
  * Kim & Bae (2022): Laplace core from KASS 16 m (95 %), tail scale = lateral separation,
    tail weight 0.05 (described by them as conservative).
  * Pang et al. (2022, AIAA): vertical deviations small relative to lateral.
No source identifies the tail weight or tail scale for drones.  The core scale is anchored
to measured RMS values (Laplace RMS = sqrt(2) a); the tails are bracketed, not estimated.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .risk import reich

SQ2 = np.sqrt(2.0)


@dataclass(frozen=True)
class NavModel:
    name: str
    a1_ref: float        # lateral/longitudinal core scale at W_ref (m)
    a2: float | None     # tail scale (m); None -> equal to the lateral separation (Kim & Bae)
    alpha: float         # tail weight
    a1z: float
    a2z: float
    alpha_z: float
    basis: str

    def lateral(self, wind: float, gamma: float, S_y: float, W_ref: float = 5.0) -> reich.LaplaceMixture:
        a1 = max(self.a1_ref * (1.0 + gamma * (wind - W_ref)), 0.5 * self.a1_ref)
        a2 = S_y if self.a2 is None else self.a2
        return reich.LaplaceMixture(a1, max(a2, a1 * 1.01), self.alpha)

    def vertical(self) -> reich.LaplaceMixture:
        return reich.LaplaceMixture(self.a1z, self.a2z, self.alpha_z)


MODELS = {
    "E1-precise": NavModel("E1-precise", 1.06 / SQ2, 5.0, 0.001, 0.5 / np.log(20), 3.0, 0.001,
                           "core = lowest measured lateral TSE RMS (1.06 m, Geister et al. 2018); light tail"),
    "E2-measured": NavModel("E2-measured", 3.5 / SQ2, 20.0, 0.01, 0.5 / np.log(20), 5.0, 0.01,
                            "core = mid-range measured RMS (~3.5 m); tail scale ~ largest observed NSE (20 m); 1 % tail"),
    "E3-degraded": NavModel("E3-degraded", 6.0 / SQ2, 30.0, 0.02, 1.0 / np.log(20), 10.0, 0.02,
                            "core = upper trial-1 RMS (6 m); heavier, wider tail (urban multipath, gusts)"),
    "E4-KimBae": NavModel("E4-KimBae", 16.0 / np.log(20), None, 0.05, 0.5 / np.log(20), 5.0, 0.05,
                          "Kim & Bae (2022): KASS 16 m 95 % core, tail scale = S_y, tail weight 0.05"),
}
REFERENCE = "E2-measured"


@dataclass(frozen=True)
class Kinematics:
    lam_x: float = 1.43   # tip-to-tip span: 0.895 m wheelbase (DJI) + ~0.53 m propeller diameter (assumed)
    lam_y: float = 1.43
    lam_z: float = 0.43   # DJI height
    V: float = 15.0       # cruise speed (m/s), below the 17 m/s P-mode maximum
    dV: float = 1.0       # same-direction relative along-track speed (m/s), assumed
    ydot: float = 0.5     # relative cross-track speed during overlap (m/s), assumed
    zdot: float = 0.2     # relative vertical speed (m/s), assumed


def kappa_matrix(cfg, nav: NavModel, kin: Kinematics, wind: float, gamma: float):
    """Collision-involvement coefficients (FH^-1 per fl/h) for every lane pair of a
    configuration.  Diagonal = in-trail; off-diagonal = lateral/vertical/diagonal pairs."""
    from .geometry import pair_separations
    n = cfg.n_lanes
    if n == 0:
        return np.zeros((0, 0))
    dy, dz, opp = pair_separations(cfg)
    geo = reich.Geometry(kin.lam_x, kin.lam_y, kin.lam_z, kin.V, kin.dV, kin.ydot, kin.zdot)
    ey = nav.lateral(wind, gamma, cfg.S_y if cfg.S_y > 0 else 60.0)
    ez = nav.vertical()
    K = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            if a == b:
                K[a, b] = reich.kappa_intrail(cfg.S_x, geo, ey, ey, ez)
            else:
                Py = reich.overlap_probability_exact(dy[a, b], geo.lam_y, ey)
                Pz = reich.overlap_probability_exact(dz[a, b], geo.lam_z, ez)
                K[a, b] = Py * Pz * 2 * geo.lam_x * geo.G(bool(opp[a, b])) / geo.V
    return K
