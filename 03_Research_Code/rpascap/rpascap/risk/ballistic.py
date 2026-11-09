"""Point-mass ballistic descent with quadratic drag and horizontal wind.

State per sample: horizontal position (x along heading, y cross), altitude z,
ground-frame velocity (u, v, w).  Drag acts on air-relative velocity:
    a = -g e_z - k |v_rel| v_rel,   k = rho Cd A / (2 m),   v_rel = v - w_wind.
Vectorised semi-implicit Euler with per-sample stopping at z<=0 and linear
interpolation of the ground crossing.  Validated in tests against the drag-free
closed form and the terminal-velocity limit.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

G0 = 9.80665
RHO_AIR = 1.225


@dataclass
class DescentInputs:
    mass: np.ndarray        # kg
    cd: np.ndarray          # drag coefficient (-)
    area: np.ndarray        # reference area (m^2)
    vx0: np.ndarray         # initial along-track ground speed (m/s)
    vz0: np.ndarray         # initial vertical speed, positive DOWN (m/s)
    h0: np.ndarray          # initial height above ground (m)
    wind_x: np.ndarray      # wind component along heading (m/s, tail wind > 0)
    wind_y: np.ndarray      # cross wind (m/s)


def simulate(inp: DescentInputs, dt: float = 0.005, t_max: float = 60.0):
    n = inp.mass.size
    k = RHO_AIR * inp.cd * inp.area / (2.0 * inp.mass)
    x = np.zeros(n); y = np.zeros(n); z = inp.h0.astype(float).copy()
    u = inp.vx0.astype(float).copy(); v = np.zeros(n); w = -inp.vz0.astype(float)
    alive = np.ones(n, bool)
    out_x = np.full(n, np.nan); out_y = np.full(n, np.nan)
    out_v = np.full(n, np.nan); out_t = np.full(n, np.nan); out_gamma = np.full(n, np.nan)
    t = 0.0
    while alive.any() and t < t_max:
        a = alive.copy()          # snapshot: 'alive' is modified inside this step
        ur = u[a] - inp.wind_x[a]; vr = v[a] - inp.wind_y[a]; wr = w[a]
        sp = np.sqrt(ur * ur + vr * vr + wr * wr)
        u[a] += -k[a] * sp * ur * dt
        v[a] += -k[a] * sp * vr * dt
        w[a] += (-G0 - k[a] * sp * wr) * dt
        zn = z[a] + w[a] * dt
        xn = x[a] + u[a] * dt; yn = y[a] + v[a] * dt
        hit = zn <= 0
        if hit.any():
            idx = np.flatnonzero(a)[hit]
            frac = z[idx] / (z[idx] - zn[hit])
            out_x[idx] = x[idx] + frac * (xn[hit] - x[idx])
            out_y[idx] = y[idx] + frac * (yn[hit] - y[idx])
            out_v[idx] = np.sqrt(u[idx] ** 2 + v[idx] ** 2 + w[idx] ** 2)
            out_t[idx] = t + frac * dt
            out_gamma[idx] = np.arctan2(-w[idx], np.hypot(u[idx], v[idx]))  # impact angle from horizontal
            alive[idx] = False
        keep = np.flatnonzero(a)[~hit]
        z[keep] = zn[~hit]; x[keep] = xn[~hit]; y[keep] = yn[~hit]
        t += dt
    ke = 0.5 * inp.mass * out_v ** 2
    return {"x": out_x, "y": out_y, "speed": out_v, "ke": ke, "t": out_t, "gamma": out_gamma}


def vacuum_solution(vx0, vz0, h0):
    """Closed form without drag or wind (vz0 positive down)."""
    t = (-vz0 + np.sqrt(vz0 ** 2 + 2 * G0 * h0)) / G0
    vz = vz0 + G0 * t
    return {"x": vx0 * t, "t": t, "speed": np.sqrt(vx0 ** 2 + vz ** 2)}


def sample_inputs(rng: np.random.Generator, n: int, p: dict, wind_speed: float = 0.0,
                  wind_dir_rel: float = 0.0) -> DescentInputs:
    """Sample descent inputs from a parameter dict (see configs/base.yaml: platform).
    wind_dir_rel: direction the wind blows TOWARD, relative to aircraft heading (rad)."""
    def draw(spec):
        kind = spec["dist"]
        if kind == "uniform":
            return rng.uniform(spec["low"], spec["high"], n)
        if kind == "normal":
            vals = rng.normal(spec["mean"], spec["sd"], n)
            lo = spec.get("min", -np.inf)
            return np.maximum(vals, lo)
        if kind == "fixed":
            return np.full(n, float(spec["value"]))
        raise ValueError(kind)
    return DescentInputs(mass=draw(p["mass"]), cd=draw(p["cd"]), area=draw(p["area"]),
                         vx0=draw(p["vx0"]), vz0=draw(p["vz0"]), h0=draw(p["h0"]),
                         wind_x=np.full(n, wind_speed * np.cos(wind_dir_rel)),
                         wind_y=np.full(n, wind_speed * np.sin(wind_dir_rel)))
