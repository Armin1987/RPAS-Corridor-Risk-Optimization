"""Reich-type collision-risk coefficients for parallel RPAS lanes.

Conventions (SI units throughout):
    distances in m, speeds in m/s, flows x in flights per hour (fl/h),
    rates returned in collision involvements per aircraft flight hour (1/FH).

For a subject aircraft in lane l, the expected number of collision involvements
per flight hour caused by traffic in a neighbouring lane l' is LINEAR in the
neighbour flow x_l' (Reich 1966; ICAO lateral CRM form):

    r_{l<-l'} = P_y(S_y) * P_z(S_z) * 2*lambda_x * n_l' * G * 3600
    n_l'      = x_l' / (3600 * V)               [aircraft per metre of lane]
    G         = dV/(2 lambda_x) + ydot/(2 lambda_y) + zdot/(2 lambda_z)   [1/s]

because the ICAO term (lambda_x / S_x) * E_y with occupancy E_y = 2 S_x n_l'
reduces to 2 lambda_x n_l'.  Hence
    r_{l<-l'} = kappa_{ll'} * x_l',  kappa_{ll'} = P_y P_z * 2*lambda_x * G / V.
(The factor 3600 from s->h cancels the 1/3600 in n.)  kappa has units FH^-1 per (fl/h).

Same-lane (in-trail) risk with an enforced minimum spacing S_x: a subject aircraft
has on average 2*phi in-trail neighbours at spacing S_x, phi = x / x_max with
x_max = 3600 V / S_x (slot occupancy).  Then
    r_self = 2*phi * P_x(S_x) * P_y(0) * P_z(0) * G_long * 3600,
which is again linear in x.  All per-aircraft constraints are therefore linear.
Only CORRIDOR-HOUR totals (x_l * r_l(x)) are bilinear.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy import integrate


@dataclass(frozen=True)
class LaplaceMixture:
    """Double-double-exponential error density (core a1, tail a2, tail weight alpha)."""
    a1: float
    a2: float
    alpha: float

    def pdf(self, y):
        y = np.abs(np.asarray(y, dtype=float))
        return ((1 - self.alpha) / (2 * self.a1) * np.exp(-y / self.a1)
                + self.alpha / (2 * self.a2) * np.exp(-y / self.a2))


def _laplace_selfconv(a: float, b: float, s: float) -> float:
    """Closed form of int f_a(y) f_b(y+s) dy for zero-mean Laplace densities, s>=0."""
    s = abs(s)
    if abs(a - b) < 1e-12:
        return np.exp(-s / a) * (1 + s / a) / (4 * a)
    # general case: convolution of two Laplace densities evaluated at s
    return (a * np.exp(-s / a) - b * np.exp(-s / b)) / (2 * (a * a - b * b))


def overlap_probability(S: float, lam: float, err: LaplaceMixture, numeric: bool = False) -> float:
    """P(S) ~= 2*lam * int f(y) f(y+S) dy  (Reich small-lambda approximation).

    Closed form for the Laplace mixture is used by default; numeric=True uses quadrature
    (kept as an independent check in the tests)."""
    if numeric:
        R = 60.0 * max(err.a1, err.a2) + abs(S)
        v, _ = integrate.quad(lambda y: err.pdf(y) * err.pdf(y + S), -R, R,
                              points=sorted({0.0, -float(S)}), limit=1000)
        return 2 * lam * v
    w = [(1 - err.alpha, err.a1), (err.alpha, err.a2)]
    tot = 0.0
    for wi, ai in w:
        for wj, aj in w:
            tot += wi * wj * _laplace_selfconv(ai, aj, S)
    return 2 * lam * tot


def difference_density(u, err: LaplaceMixture):
    """Density h(u) of D = E1 - E2 for independent errors E1, E2 ~ err (symmetric),
    so that int f(y) f(y+S) dy = h(S)."""
    w = [(1 - err.alpha, err.a1), (err.alpha, err.a2)]
    u = np.abs(np.asarray(u, float))
    out = np.zeros_like(u)
    for wi, ai in w:
        for wj, aj in w:
            if abs(ai - aj) < 1e-12:
                out += wi * wj * np.exp(-u / ai) * (1 + u / ai) / (4 * ai)
            else:
                out += wi * wj * (ai * np.exp(-u / ai) - aj * np.exp(-u / aj)) / (2 * (ai * ai - aj * aj))
    return out


def overlap_probability_exact(S: float, lam: float, err: LaplaceMixture) -> float:
    """Exact P(|D + S| < lam) = int_{S-lam}^{S+lam} h(u) du.  The Reich form 2*lam*h(S) is
    its small-lam limit; for RPAS the airframe size is often comparable to (or larger than)
    the vertical navigation error, where 2*lam*h(S) can exceed 1."""
    pts = [p for p in (0.0,) if S - lam < p < S + lam]
    v, _ = integrate.quad(lambda u: difference_density(u, err), S - lam, S + lam,
                          points=pts or None, limit=400)
    return float(min(1.0, v))


@dataclass(frozen=True)
class Geometry:
    lam_x: float   # along-track size (m)
    lam_y: float   # cross-track size (m)
    lam_z: float   # height (m)
    V: float       # nominal ground speed (m/s)
    dV: float      # mean |relative along-track speed| between same-direction aircraft (m/s)
    ydot: float    # mean |relative cross-track speed| during overlap (m/s)
    zdot: float    # mean |relative vertical speed| during overlap (m/s)

    def G(self, opposite: bool = False) -> float:
        dv = 2 * self.V if opposite else self.dV
        return dv / (2 * self.lam_x) + self.ydot / (2 * self.lam_y) + self.zdot / (2 * self.lam_z)


def _P(S, lam, err, exact):
    return overlap_probability_exact(S, lam, err) if exact else overlap_probability(S, lam, err)


def kappa_lateral(S_y: float, geo: Geometry, ey: LaplaceMixture, ez: LaplaceMixture,
                  opposite: bool = False, exact: bool = True) -> float:
    """Collision involvements per FH of a subject aircraft per (fl/h) in an adjacent
    lane at lateral offset S_y and the same altitude."""
    Py = _P(S_y, geo.lam_y, ey, exact)
    Pz = _P(0.0, geo.lam_z, ez, exact)
    return Py * Pz * 2 * geo.lam_x * geo.G(opposite) / geo.V


def kappa_vertical(S_z: float, geo: Geometry, ey: LaplaceMixture, ez: LaplaceMixture,
                   opposite: bool = False, exact: bool = True) -> float:
    """As kappa_lateral, for a lane directly above/below at vertical offset S_z."""
    Py = _P(0.0, geo.lam_y, ey, exact)
    Pz = _P(S_z, geo.lam_z, ez, exact)
    return Py * Pz * 2 * geo.lam_x * geo.G(opposite) / geo.V


def kappa_intrail(S_x: float, geo: Geometry, ex: LaplaceMixture, ey: LaplaceMixture,
                  ez: LaplaceMixture, exact: bool = True) -> float:
    """Same-lane coefficient: r_self = kappa_intrail * x  (x in fl/h), given a minimum
    in-trail spacing S_x enforced by slot allocation (x <= 3600 V / S_x)."""
    Px = _P(S_x, geo.lam_x, ex, exact)
    Py = _P(0.0, geo.lam_y, ey, exact)
    Pz = _P(0.0, geo.lam_z, ez, exact)
    x_max = 3600.0 * geo.V / S_x
    G_long = geo.dV / (2 * geo.lam_x) + geo.ydot / (2 * geo.lam_y) + geo.zdot / (2 * geo.lam_z)
    return 2.0 / x_max * Px * Py * Pz * G_long * 3600.0


def laplace_scale_from_95(acc95: float) -> float:
    """Laplace scale a such that P(|Y| <= acc95) = 0.95  ->  a = acc95 / ln(20)."""
    return acc95 / np.log(20.0)
