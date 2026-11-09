"""Event -> impact -> exposure -> harm chain, expressed per crash.

C_l = E[third-party fatalities | one crash while flying lane l]
    = (1/|P_l|) sum_{p in P_l} sum_i K_p(i) * rho_i * A_L * (1 - S_i) * Pf(E)

K_p(i)  probability that a crash initiated at lane point p impacts cell i (sums to 1)
rho_i   people per m^2 in cell i (unsheltered-equivalent before S_i)
A_L     lethal area of one impact (m^2) -- NOT the cell area (Stage 1 defect ME-05)
S_i     sheltering fraction in [0,1]
Pf(E)   P(fatality | person within A_L, impact energy E)

Ground fatalities per aircraft flight hour: g_l = (lambda_f + r_l) * C_l, where
lambda_f = loss-of-control rate (1/FH) and r_l = collision-involvement rate (1/FH,
from reich.py).  Each crash is attributed to the aircraft that falls, so a mid-air
collision contributes once per falling aircraft and never as a separate fatality
term (Stage 1 defect ME-02/03).
"""
from __future__ import annotations
import numpy as np


def lethal_area(gamma: np.ndarray, r_uas: float, r_person: float = 0.3,
                h_person: float = 1.8) -> np.ndarray:
    """Glide-type lethal area: A_L = 2 (r_p + r_u) h_p / tan(gamma) + pi (r_p + r_u)^2.
    gamma = impact angle from horizontal (rad).  Form used in UAS ground-risk
    literature (source to be verified in Stage 3)."""
    R = r_person + r_uas
    g = np.clip(gamma, np.deg2rad(5.0), np.pi / 2)
    return 2 * R * h_person / np.tan(g) + np.pi * R ** 2


def pf_dalamagkidis(E: np.ndarray, sheltering: float, alpha: float = 1e6,
                    beta: float = 100.0) -> np.ndarray:
    """Fatality probability with sheltering parameter p_s in (0,1]:
        Pf = 1 / (1 + sqrt(alpha/beta) * (beta/E)^(1/(4 p_s)))
    (parameters alpha [J], beta [J] to be verified in Stage 3)."""
    ps = max(float(sheltering), 1e-6)
    E = np.maximum(np.asarray(E, float), 1e-9)
    return 1.0 / (1.0 + np.sqrt(alpha / beta) * (beta / E) ** (1.0 / (4.0 * ps)))


def pf_logistic(E, E50: float, k: float):
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(E, float) - E50)))


def impact_kernel(dx: np.ndarray, dy: np.ndarray, cell: float, half_width_cells: int):
    """2-D histogram of impact offsets (m) on a (2H+1)^2 cell grid centred on the
    crash-initiation point.  Returns probabilities summing to the captured mass and the
    captured mass itself (should be ~1; reported, never silently renormalised)."""
    H = half_width_cells
    edges = (np.arange(-H, H + 2) - 0.5) * cell
    hist, _, _ = np.histogram2d(dx, dy, bins=[edges, edges])
    p = hist / dx.size
    return p, p.sum()


def rotate(x: np.ndarray, y: np.ndarray, heading: float):
    c, s = np.cos(heading), np.sin(heading)
    return c * x - s * y, s * x + c * y


def lane_consequence(lane_cells: list[tuple[int, int]], kernel: np.ndarray,
                     rho: np.ndarray, shelter: np.ndarray, AL_pf: float) -> float:
    """Average expected fatalities per crash along a lane.
    lane_cells: grid indices of crash-initiation points (row, col)
    kernel: impact probabilities (already rotated to the lane heading/wind)
    rho, shelter: grids (people/m^2, fraction)
    AL_pf: E[A_L * Pf] for this platform and wind state (m^2), from the descent sample."""
    H = kernel.shape[0] // 2
    nr, nc = rho.shape
    expo = rho * (1.0 - shelter)
    vals = []
    for (r, c) in lane_cells:
        r0, r1 = r - H, r + H + 1
        c0, c1 = c - H, c + H + 1
        kr0, kc0 = max(0, -r0), max(0, -c0)
        kr1 = kernel.shape[0] - max(0, r1 - nr)
        kc1 = kernel.shape[1] - max(0, c1 - nc)
        sub = expo[max(r0, 0):min(r1, nr), max(c0, 0):min(c1, nc)]
        vals.append(float((kernel[kr0:kr1, kc0:kc1] * sub).sum()))
    return AL_pf * float(np.mean(vals))
