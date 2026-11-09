import numpy as np
import pytest
from rpascap.model import two_stage as ts
from tests.helpers import random_instance


@pytest.mark.parametrize("seed,eps", [(0, False), (1, False), (2, True), (3, True), (4, False), (5, True)])
def test_milp_equals_exhaustive_enumeration(seed, eps):
    ins = random_instance(seed=seed, eps=eps)
    r = ts.solve(ins)
    assert r.status == 0
    bf, _ = ts.brute_force(ins)
    assert abs(r.obj - bf) <= 1e-6 * max(1, abs(bf))
    assert r.residual_max < 1e-7


@pytest.mark.parametrize("seed", range(6))
def test_solution_safe_when_recomputed_independently(seed):
    ins = random_instance(seed=seed, theta_A=2e-6)
    r = ts.solve(ins)
    rk = ts.recompute_risk(ins, r)
    assert rk["max_ground_ratio"] <= 1 + 1e-7 and rk["max_air_ratio"] <= 1 + 1e-7


def test_societal_linear_bound_is_conservative():
    ins = random_instance(seed=2, eps=True)
    r = ts.solve(ins)
    F = ts.recompute_risk(ins, r)["F_per_hour"]
    assert (F <= ins.eps[:, None] * (1 + 1e-7)).all()     # exact F never exceeds eps


def test_zero_demand_gives_zero_admissions_and_no_preapproval():
    ins = random_instance(seed=7)
    ins.O[:] = 0; ins.P[:] = 0
    r = ts.solve(ins)
    assert max(r.x.values()) < 1e-9 and max(r.q.values()) < 1e-9


def test_preapproval_trades_off_cancellations():
    ins = random_instance(seed=12)
    base = ts.solve(ins)
    ins.cancel_penalty = 1e3                 # prohibitive penalty -> only safe pre-approvals
    hi = ts.solve(ins)
    assert sum(hi.q.values()) <= sum(base.q.values()) + 1e-6
    # with prohibitive penalty no scenario may cancel
    for (k, t), qv in hi.q.items():
        for w in range(ins.W):
            assert hi.y[k, t, w] >= qv - 1e-6


def test_switching_limit_respected():
    ins = random_instance(seed=8, B=4, T=8)
    ins.max_switches = 0
    r = ts.solve(ins)
    for k, cfgs in enumerate(ins.configs):
        plan = [max(range(len(cfgs)), key=lambda c: r.z[k, c, b]) for b in range(ins.B)]
        assert len(set(plan)) == 1


def test_tighter_target_never_increases_objective():
    ins = random_instance(seed=9)
    base = ts.solve(ins).obj
    ins.theta_A = 1e-7
    assert ts.solve(ins).obj <= base + 1e-9


def test_mitigation_never_decreases_objective():
    ins = random_instance(seed=9, theta_A=2e-6)
    base = ts.solve(ins).obj
    ins.mitigation = 0.9
    assert ts.solve(ins).obj >= base - 1e-9


def test_first_stage_scenario_independent():
    I = ts.Index(random_instance(seed=10))
    assert all(len(k) == 3 for k in I.z) and all(len(k) == 2 for k in I.q)


def test_fixed_first_stage_reevaluation_matches():
    ins = random_instance(seed=11)
    r = ts.solve(ins)
    ev = ts.solve(ins, fixed_z={k: float(v) for k, v in r.z.items()}, fixed_q=r.q)
    assert ev.status == 0 and abs(ev.obj - r.obj) < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_corridor_decomposition_is_exact(seed):
    ins = random_instance(seed=seed, K=3)
    a = ts.solve(ins); b = ts.solve_decomposed(ins)
    assert abs(a.obj - b.obj) <= 1e-6 * max(1, abs(a.obj))
    # fixed first stage through the decomposed path reproduces the objective
    c = ts.solve_decomposed(ins, fixed_z={k: float(v) for k, v in b.z.items()}, fixed_q=b.q)
    assert abs(c.obj - b.obj) < 1e-6
