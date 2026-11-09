"""Preliminary solver-feasibility check (Stage 2, item 13).

SYNTHETIC coefficients only: wind scenarios are drawn from a Weibull distribution and
the per-crash consequence is a placeholder function of wind.  The output is a runtime
and model-size log, NOT a study result.
"""
import json, platform, sys, time
import numpy as np
import scipy
from rpascap.risk import reich
from rpascap.model.coefficients import CorridorConfig, config_arrays
from rpascap.model.two_stage import Instance, solve

geo = reich.Geometry(0.895, 0.895, 0.43, 15.0, 1.0, 1.0, 0.1)
nav = dict(a1_0=0.5, beta_a=0.1, a1_x=0.5, a1_z=0.17, a2=20.0, a2_z=10.0, alpha=0.01)
menu = [CorridorConfig("closed", 0, 0, 1, 1, 1, 0.0),
        CorridorConfig("1x2 wide", 2, 1, 60, 30, 150, 1.0),
        CorridorConfig("1x3 std", 3, 1, 40, 30, 120, 1.5),
        CorridorConfig("2x2 std", 2, 2, 40, 20, 120, 2.0),
        CorridorConfig("2x3 dense", 3, 2, 25, 15, 90, 3.0)]
C_of_wind = lambda W: 0.02 * (1 + 0.05 * W)          # placeholder fatalities/crash


def make(K, T, B, Wn, seed):
    rng = np.random.default_rng(seed)
    wind = 6.0 * rng.weibull(2.0, (T, Wn)) + 1.0
    configs = [[config_arrays(c, wind * rng.uniform(0.9, 1.1), geo, nav, C_of_wind, 12.0)
                for c in menu] for _ in range(K)]
    D = rng.poisson(45, (K, T, Wn)).astype(float)
    block = np.repeat(np.arange(B), T // B)
    return Instance(configs, D, np.full(Wn, 1 / Wn), block, np.full(K, 0.1),
                    np.full((K, T), 60.0), max_switches=2, theta_A=1e-6, lambda_f=1e-5)


log = {"python": sys.version.split()[0], "scipy": scipy.__version__,
       "machine": platform.machine(), "cpu": platform.processor() or "n/a", "runs": []}
for (K, T, B, Wn) in [(4, 16, 4, 20), (6, 16, 4, 50), (6, 16, 4, 100), (8, 16, 4, 100)]:
    t0 = time.perf_counter(); ins = make(K, T, B, Wn, 0); tb = time.perf_counter() - t0
    r = solve(ins, time_limit=900, mip_rel_gap=1e-4)
    rec = dict(K=K, T=T, B=B, scenarios=Wn, vars=r.n_vars, cons=r.n_cons, ints=r.n_int,
               build_s=round(tb, 1), solve_s=round(r.runtime_s, 1), status=r.status,
               message=r.message, gap=r.mip_gap, residual=r.residual_max)
    log["runs"].append(rec); print(rec, flush=True)
json.dump(log, open("solver_feasibility_log.json", "w"), indent=1)
