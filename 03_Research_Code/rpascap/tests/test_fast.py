import numpy as np
import pytest
from rpascap.model import two_stage as ts
from rpascap.model.fast import solve_fast
from tests.helpers import random_instance


@pytest.mark.parametrize("seed", range(12))
def test_structured_algorithm_equals_milp(seed):
    ins = random_instance(seed=seed, K=2, B=3, T=6, W=4, theta_A=[1e-5, 2e-6, 5e-7][seed % 3])
    ins.max_switches = seed % 3
    ins.mitigation = [0.0, 0.5, 0.9][seed % 3]
    a = ts.solve(ins); b = solve_fast(ins)
    assert abs(a.obj - b.obj) <= 1e-6 * max(1.0, abs(a.obj))


@pytest.mark.parametrize("seed", range(4))
def test_structured_algorithm_equals_enumeration(seed):
    ins = random_instance(seed=seed + 100)
    bf, _ = ts.brute_force(ins)
    assert abs(solve_fast(ins).obj - bf) <= 1e-6 * max(1.0, abs(bf))


@pytest.mark.parametrize("seed", range(4))
def test_structured_solution_is_feasible_and_safe(seed):
    ins = random_instance(seed=seed + 200, theta_A=2e-6)
    r = solve_fast(ins)
    rk = ts.recompute_risk(ins, r)
    assert rk["max_ground_ratio"] <= 1 + 1e-7 and rk["max_air_ratio"] <= 1 + 1e-7
    # admissions respect pre-approvals, demand and lane caps
    for (k, t, w), yy in r.y.items():
        assert yy <= r.q[k, t] + 1e-9 and r.o[k, t, w] <= ins.O[k, t, w] + 1e-9
    # the MILP re-evaluating this first stage gives the same value
    ev = ts.solve(ins, fixed_z={k: float(v) for k, v in r.z.items()}, fixed_q=r.q)
    assert abs(ev.obj - r.obj) <= 1e-6 * max(1.0, abs(r.obj))


def test_fixed_first_stage_evaluation_matches_milp():
    ins = random_instance(seed=300)
    m = ts.solve(ins)
    a = solve_fast(ins, fixed_z=m.z, fixed_q=m.q)
    assert abs(a.obj - m.obj) <= 1e-6 * max(1.0, abs(m.obj))


def _phantom_instance(seed, mitigation=0.0):
    """Synthetic instance (NOT study data) in which per-lane risk rows bind strongly and
    some lanes are excluded by the ground test (lambda_f C > theta_G), so that leaving a
    lane empty can raise capacity.  Exercises the open-lane semantics."""
    rng = np.random.default_rng(seed)
    ins = random_instance(seed=seed, K=1, B=2, T=4, W=3, lanes=(0, 4, 3), theta_A=1e-6)
    for cd in ins.configs[0]:
        n = cd.n_lanes
        if n == 0:
            continue
        cd.kappa *= rng.uniform(5, 30)
        cd.C = rng.uniform(0.005, 0.12, cd.C.shape)       # 0.12*1e-5 > 1e-6 -> some lanes excluded
        # make outer lanes strongly coupled to their neighbour, as in stacked layers
        cd.kappa[:, :, 0, 1] *= 20; cd.kappa[:, :, 1, 0] *= 20
    ins.mitigation = mitigation
    return ins


@pytest.mark.parametrize("seed", range(10))
def test_open_lane_semantics_milp_equals_fast(seed):
    ins = _phantom_instance(400 + seed, mitigation=[0.0, 0.5][seed % 2])
    ins.max_switches = seed % 2
    a = ts.solve(ins); b = solve_fast(ins)
    assert a.status == 0
    assert abs(a.obj - b.obj) <= 1e-6 * max(1.0, abs(a.obj))
    assert a.residual_max <= 1e-6 and b.residual_max <= 1e-6


def test_capacity_equals_support_enumeration():
    """F* from lane_capacity_lp equals brute-force enumeration over lane supports."""
    import itertools
    from scipy.optimize import linprog
    from rpascap.model.fast import lane_capacity_lp
    n_checked = n_strict = 0
    for seed in range(6):
        ins = _phantom_instance(500 + seed)
        for cd in ins.configs[0]:
            for t in range(ins.T):
                for w in range(ins.W):
                    if cd.n_lanes == 0:
                        continue
                    F, x = lane_capacity_lp(cd, t, w, ins)
                    ub = np.array([cd.U[t, w, l] if ts.usable(ins, cd, t, w, l) else 0 for l in range(cd.n_lanes)])
                    K = cd.kappa[t, w] * (1 - ins.mitigation); C = cd.C[t, w]
                    best = 0.0
                    for S in itertools.product([0, 1], repeat=cd.n_lanes):
                        S = np.array(S, bool) & (ub > 0)
                        if not S.any():
                            continue
                        A = [r for l in np.flatnonzero(S) for r in (C[l] * K[l] / ins.theta_G, K[l] / ins.theta_A)]
                        b = [r for l in np.flatnonzero(S) for r in (1 - ins.lambda_f * C[l] / ins.theta_G, 1.0)]
                        r = linprog(-np.ones(cd.n_lanes), A_ub=np.array(A), b_ub=b,
                                    bounds=[(0, ub[l] if S[l] else 0) for l in range(cd.n_lanes)])
                        best = max(best, -r.fun)
                    # all-lanes rows (the stricter, superseded semantics)
                    on = ub > 0
                    if on.any():
                        A = [r for l in np.flatnonzero(on) for r in (C[l] * K[l] / ins.theta_G, K[l] / ins.theta_A)]
                        b = [r for l in np.flatnonzero(on) for r in (1 - ins.lambda_f * C[l] / ins.theta_G, 1.0)]
                        r = linprog(-np.ones(cd.n_lanes), A_ub=np.array(A), b_ub=b, bounds=[(0, u) for u in ub])
                        n_strict += (-r.fun) < best - 1e-6
                    assert abs(F - best) <= 1e-6 * max(1, best)
                    n_checked += 1
    assert n_checked > 50 and n_strict > 0      # the test does exercise the open-lane case


def test_ground_exclusion_is_exercised():
    ins = _phantom_instance(600)
    excluded = sum(not ts.usable(ins, cd, t, w, l) and cd.U[t, w, l] > 0
                   for cd in ins.configs[0] for t in range(ins.T) for w in range(ins.W) for l in range(cd.n_lanes))
    assert excluded > 0
