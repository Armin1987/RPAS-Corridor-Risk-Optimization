"""The curve formulation (model/societal.py) solves the same model as the big-M extensive
form (two_stage.solve) when the societal constraint is active.  Synthetic instances only."""
import numpy as np
import pytest
from tests.helpers import random_instance
from rpascap.model import two_stage as ts
from rpascap.model.societal import solve_societal
from tests.test_fast import _phantom_instance


def _with_eps(ins, frac, seed):
    """eps_t = frac x the societal (linear-bound) load of the unconstrained optimum."""
    r = ts.solve(ins.__class__(**{**ins.__dict__, "eps": None}))
    load = np.zeros((ins.T, ins.W))
    for (k, c, l, t, w), v in r.x.items():
        cd = ins.configs[k][c]
        load[t, w] += ins.tau_h[k] * (ins.lambda_f + ins.theta_A) * cd.C[t, w, l] * v
    ins.eps = np.maximum(frac * load.max(axis=1), 1e-12)
    return ins


@pytest.mark.parametrize("seed", range(8))
def test_curve_formulation_equals_bigM_milp(seed):
    if seed % 2:
        ins = _phantom_instance(700 + seed, mitigation=0.0)
    else:
        ins = random_instance(seed=700 + seed, eps=False)
    ins = _with_eps(ins, [0.3, 0.6][seed % 4 // 2], seed)
    a = ts.solve(ins, mip_rel_gap=1e-9)
    b = solve_societal(ins, mip_rel_gap=1e-9)
    assert a.status == 0 and b.status == 0
    assert abs(a.obj - b.obj) <= 1e-6 * max(1.0, abs(a.obj)), (a.obj, b.obj)
    assert b.residual_max <= 1e-6


def test_curve_formulation_fixed_first_stage():
    ins = _with_eps(_phantom_instance(811), 0.4, 0)
    a = ts.solve(ins, mip_rel_gap=1e-9)
    fz = {kk: float(v) for kk, v in a.z.items()}
    e1 = ts.solve(ins, fixed_z=fz, fixed_q=a.q); e2 = solve_societal(ins, fixed_z=fz, fixed_q=a.q)
    assert abs(e1.obj - e2.obj) <= 1e-6 * max(1.0, abs(e1.obj))
    assert e2.residual_max <= 1e-6


def test_eps_binds_in_tests():
    """The societal constraint actually changes the optimum in these instances."""
    ins = _with_eps(random_instance(seed=700), 0.3, 0)
    free = ts.solve(ins.__class__(**{**ins.__dict__, "eps": None}))
    assert solve_societal(ins).obj < free.obj - 1e-6


def test_disjunction_is_exercised():
    """Some blocks keep more than one non-dominated open-lane set (binaries are needed)."""
    from rpascap.model.societal import block_curves
    ins = _phantom_instance(701)
    multi = sum(len(block_curves(cd, t, w, ins, 0)) > 1 for cd in ins.configs[0]
                for t in range(ins.T) for w in range(ins.W))
    assert multi > 0


@pytest.mark.parametrize("seed", range(6))
def test_block_form_milp_equals_structured_algorithm(seed):
    """Without the societal row the block-form MILP equals the structured algorithm."""
    from rpascap.model.fast import solve_fast
    ins = _phantom_instance(900 + seed) if seed % 2 else random_instance(seed=900 + seed, K=2, B=3, T=6, W=4)
    ins.max_switches = seed % 3
    a = solve_fast(ins); b = solve_societal(ins, mip_rel_gap=1e-9)
    assert b.status == 0
    assert abs(a.obj - b.obj) <= 1e-6 * max(1.0, abs(a.obj))
    assert b.residual_max <= 1e-6
