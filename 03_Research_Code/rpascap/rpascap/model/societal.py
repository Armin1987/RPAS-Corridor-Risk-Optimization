"""Block (curve) formulation of the two-stage model, with or without the hourly societal
constraint (7).  Without (7) each block reduces to its capacity F* (Proposition 4, step 2),
giving a compact MILP in the configuration binaries only; this is used to check the
structured algorithm at study scale, where the big-M extensive form does not solve.

Same model as two_stage.build() (open-lane semantics, constraint (7)); different, tighter
formulation.  Constraint (7) depends on the lane flows of block (k, c, t, w) only through
    s = sum_l coef_l x_l,   coef_l = tau_k (lambda_f + theta_A) C_l,
and the objective and balance rows only through the total F = sum_l x_l.  The feasible lane
flows of a block are the union over open-lane sets S of the polytopes X_S (fast.py, (b)), so
the block enters the problem only through the 2-D set  P = union_S P_S,
P_S = {(1'x, coef'x) : x in X_S}.  Because rows of X_S are homogeneous in x with non-negative
right-hand sides, every P_S contains the segment from (0,0) to each of its points, so for
the purposes of (7) (which only bounds s from above) only the lower boundary
    g_S(F) = min{coef'x : x in X_S, 1'x = F},   F in [0, F_S],
matters; g_S is convex and piecewise linear with g_S(0) = 0.  Hence a block is feasible with
(F, s) iff, for some S, F <= F_S and s >= g_S(F).  We compute the breakpoints of each g_S
exactly (parametric LP, 'sandwich' recursion), drop subsets whose curve is dominated by
another subset's, and model the remaining disjunction with one binary per (block, subset)
and the standard disjunctive (convex-hull) formulation:
    F_j <= F_Sj lam_j,  s_j >= a_i F_j + b_i lam_j (each linear piece i of g_Sj),
    sum_j lam_j <= z[k,c,b(t)].
When a block has a single relevant subset, lam is replaced by z (no new binary).
A lane-flow vector is recovered by interpolating the stored vertex flows of the active
piece (a convex combination of two points of X_S, so it lies in X_S and attains g_S(F)).
The recovered solution is checked independently with two_stage.check_solution.
Tests verify that the optimum equals the big-M extensive form (two_stage.solve).
"""
from __future__ import annotations
import time
from itertools import combinations
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy import sparse
from .two_stage import Instance, Result, usable, check_solution

TOL_REL = 1e-9


def _rows(S, K, C, ins):
    A, b = [], []
    for l in S:
        A.append(C[l] * K[l] / ins.theta_G); b.append(1.0 - ins.lambda_f * C[l] / ins.theta_G)
        A.append(K[l] / ins.theta_A); b.append(1.0)
    A = np.array(A); b = np.array(b)
    keep = np.abs(A).sum(axis=1) > 0
    return (A[keep], b[keep]) if keep.any() else (None, None)


def _lp(c, S, ub, A, b, Aeq=None, beq=None):
    n = ub.size
    bnd = [(0.0, ub[l] if l in S else 0.0) for l in range(n)]
    r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=bnd, method="highs")
    if r.status != 0:
        raise RuntimeError(f"block LP failed: {r.message}")
    return np.clip(r.x, 0, [hi for _, hi in bnd])


def subset_curve(S, ub, K, C, coef, ins):
    """Vertices [(F, s, x)] of g_S, from (0, 0) to (F_S, g_S(F_S)), F increasing."""
    S = set(S); n = ub.size
    A, b = _rows(S, K, C, ins)
    one = np.ones(n)
    scale = max(float(coef.max()), 1e-300)
    cn = coef / scale
    xF = _lp(-one, S, ub, A, b)
    FS = float(xF.sum())
    if FS <= 1e-12:
        return []
    xR = _lp(cn, S, ub, A, b, Aeq=one[None, :], beq=[FS * (1 - 1e-10)])
    V = [(0.0, 0.0, np.zeros(n)), (float(xR.sum()), float(cn @ xR), xR)]

    def rec(i, j, depth=0):
        (FA, sA, _), (FB, sB, _) = V[i], V[j]
        if FB - FA <= 1e-12 or depth > 30:
            return []
        a = (sB - sA) / (FB - FA)
        x = _lp(cn - a * one, S, ub, A, b)
        val = float((cn - a * one) @ x)
        if val < sA - a * FA - TOL_REL * max(1.0, abs(sA), abs(sB)):
            return [(float(x.sum()), float(cn @ x), x)]
        return []

    # iterative refinement until no segment can be improved
    changed = True
    while changed:
        changed = False
        out = [V[0]]
        for i in range(len(V) - 1):
            new = rec(i, i + 1)
            if new and V[i][0] + 1e-12 < new[0][0] < V[i + 1][0] - 1e-12:
                out.append(new[0]); changed = True
            out.append(V[i + 1])
        V = out
    return [(F, s * scale, x) for F, s, x in V]


def _interp(V, F):
    Fs = np.array([v[0] for v in V]); ss = np.array([v[1] for v in V])
    return np.interp(F, Fs, ss, right=np.inf)


def block_curves(cd, t, w, ins: Instance, k):
    """Relevant (non-dominated) subsets of block (k, c, t, w), each as a vertex list."""
    n = cd.n_lanes
    if n == 0:
        return []
    ub = np.array([cd.U[t, w, l] if usable(ins, cd, t, w, l) else 0.0 for l in range(n)])
    lanes = [l for l in range(n) if ub[l] > 0]
    if not lanes:
        return []
    K = (1.0 - ins.mitigation) * cd.kappa[t, w]
    C = cd.C[t, w]
    coef = ins.tau_h[k] * (ins.lambda_f + ins.theta_A) * C
    curves = []
    for r in range(len(lanes), 0, -1):
        for S in combinations(lanes, r):
            V = subset_curve(S, ub, K, C, coef, ins)
            if V:
                curves.append(V)
    # dominance: drop V if some other curve reaches at least as far and is never higher
    keep = []
    for i, V in enumerate(curves):
        FS = V[-1][0]
        dom = False
        for j, W in enumerate(curves):
            if j == i or W[-1][0] < FS - 1e-9:
                continue
            pts = sorted({v[0] for v in V} | {u[0] for u in W if u[0] <= FS})
            gv = _interp(V, np.array(pts)); gw = _interp(W, np.array(pts))
            tol = 1e-9 * max(float(np.max(np.abs(gv), initial=0.0)), 1e-300)
            if np.all(gw <= gv + tol + 1e-18):
                # tie-break identical curves by index so that exactly one survives
                if np.all(np.abs(gw - gv) <= tol + 1e-18) and W[-1][0] <= FS + 1e-9 and j > i:
                    continue
                dom = True
                break
        if not dom:
            keep.append(V)
    return keep


def capacity_curve(cd, t, w, ins: Instance):
    """Without the societal row only the capacity F* = max_S F_S of a block matters: one
    'curve' from (0, 0) to (F*, 0) with the maximising flow (step 2 of Proposition 4)."""
    from .fast import lane_capacity_lp
    F, x = lane_capacity_lp(cd, t, w, ins)
    if F <= 1e-12:
        return []
    return [[(0.0, 0.0, np.zeros(cd.n_lanes)), (float(F), 0.0, x)]]


def solve_societal(ins: Instance, time_limit=600, mip_rel_gap=1e-6, fixed_z=None, fixed_q=None,
                   cache=None, cap_key=None, time_sharing=False) -> Result:
    t0 = time.perf_counter()
    no_eps = ins.eps is None
    K, T, W, B = ins.K, ins.T, ins.W, ins.B
    n = 0
    iz, iw, iq, iy, io = {}, {}, {}, {}, {}
    for k, cfgs in enumerate(ins.configs):
        for c in range(len(cfgs)):
            for b in range(B):
                iz[k, c, b] = n; n += 1
                iw[k, c, b] = n; n += 1
    for k in range(K):
        for t in range(T):
            iq[k, t] = n; n += 1
            for w in range(W):
                iy[k, t, w] = n; n += 1
                io[k, t, w] = n; n += 1
    blocks = {}          # (k,c,t,w) -> list of (V, jF, js, jlam or None)
    n_curve_keys = 0
    for k, cfgs in enumerate(ins.configs):
        for c, cd in enumerate(cfgs):
            for t in range(T):
                for w in range(W):
                    fn = (lambda: capacity_curve(cd, t, w, ins)) if no_eps else (lambda: block_curves(cd, t, w, ins, k))
                    key = None if cap_key is None else ("cap1" if no_eps else "soc",) + tuple(cap_key(k, c, t, w))
                    if key is None or cache is None:
                        cur = fn(); n_curve_keys += 1
                    else:
                        cur = cache.get(key, fn)
                    ent = []
                    for V in cur:
                        jF = n; js = n + 1; n += 2
                        jl = None
                        if len(cur) > 1:
                            jl = n; n += 1
                        ent.append((V, jF, js, jl))
                    blocks[k, c, t, w] = ent
    cost = np.zeros(n); lb = np.zeros(n); ub = np.full(n, np.inf); integ = np.zeros(n)
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0

    def add(entries, l, h):
        nonlocal r
        for j, v in entries:
            rows.append(r); cols.append(j); vals.append(v)
        lo.append(l); hi.append(h); r += 1

    for (k, c, b), j in iz.items():
        cost[j] = ins.configs[k][c].fixed_cost; ub[j] = 1; integ[j] = 1
        if fixed_z is not None:
            lb[j] = ub[j] = fixed_z[k, c, b]; integ[j] = 0
    for (k, c, b), j in iw.items():
        cost[j] = ins.switch_cost if b >= 1 else 0.0; ub[j] = 1 if b >= 1 else 0
    for (k, t), j in iq.items():
        cost[j] = ins.cancel_penalty; ub[j] = ins.P[k, t]
        if fixed_q is not None:
            lb[j] = ub[j] = fixed_q[k, t]
    for (k, t, w), j in iy.items():
        cost[j] = -ins.p[w] * (ins.value + ins.cancel_penalty)
    for (k, t, w), j in io.items():
        cost[j] = -ins.p[w] * ins.value; ub[j] = ins.O[k, t, w]
    for k, cfgs in enumerate(ins.configs):
        for b in range(B):
            add([(iz[k, c, b], 1.0) for c in range(len(cfgs))], 1.0, 1.0)
        for c in range(len(cfgs)):
            for b in range(1, B):
                add([(iw[k, c, b], 1.0), (iz[k, c, b], -1.0), (iz[k, c, b - 1], 1.0)], 0.0, np.inf)
        add([(iw[k, c, b], 1.0) for c in range(len(cfgs)) for b in range(1, B)], -np.inf, ins.max_switches)
    for (k, c, t, w), ent in blocks.items():
        b = ins.block_of_t[t]
        lams = []
        for V, jF, js, jl in ent:
            # The load variable is measured in units of the hour's limit (sigma = s / eps_t), so
            # that variables and rows are O(1); raw loads (~1e-8) sit below solver tolerances.
            sw = 1.0 if no_eps else float(ins.eps[t])
            if no_eps:
                ub[js] = 0.0
            lam = jl if jl is not None else iz[k, c, b]
            if jl is not None:
                ub[jl] = 1; integ[jl] = 0 if time_sharing else 1; lams.append(jl)
            ub[jF] = V[-1][0]
            add([(jF, 1.0), (lam, -V[-1][0])], -np.inf, 0.0)
            for i in range(len(V) - 1 if not no_eps else 0):
                (F1, s1, _), (F2, s2, _) = V[i], V[i + 1]
                a = (s2 - s1) / (F2 - F1); bb = s1 - a * F1
                add([(js, 1.0), (jF, -a / sw), (lam, -bb / sw)], 0.0, np.inf)
        if lams:
            add([(j, 1.0) for j in lams] + [(iz[k, c, b], -1.0)], -np.inf, 0.0)
    for k, cfgs in enumerate(ins.configs):
        for t in range(T):
            for w in range(W):
                ent = [(e[1], 1.0) for c in range(len(cfgs)) for e in blocks[k, c, t, w]]
                add(ent + [(iy[k, t, w], -1.0), (io[k, t, w], -1.0)], 0.0, 0.0)
                add([(iy[k, t, w], 1.0), (iq[k, t], -1.0)], -np.inf, 0.0)
    for t in range(T if not no_eps else 0):
        for w in range(W):
            ent = [(e[2], 1.0) for (k, c, tt, ww), es in blocks.items() if tt == t and ww == w for e in es]
            if ent:
                add(ent, -np.inf, 1.0)
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n))
    t1 = time.perf_counter()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub), integrality=integ,
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False})
    if res.x is None:
        return Result(res.status, res.message, np.nan, {}, {}, {}, {}, {}, time.perf_counter() - t0, n, r,
                      int(integ.sum()), None)
    xv = res.x
    z = {kk: int(round(xv[j])) for kk, j in iz.items()}
    q = {kk: float(xv[j]) for kk, j in iq.items()}
    y = {kk: float(xv[j]) for kk, j in iy.items()}
    o = {kk: float(xv[j]) for kk, j in io.items()}
    x = {}
    for (k, c, t, w), ent in blocks.items():
        cd = ins.configs[k][c]
        xs = np.zeros(cd.n_lanes)
        for V, jF, js, jl in ent:
            F = float(xv[jF])
            if F <= 1e-12:
                continue
            Fs = np.array([v[0] for v in V])
            i = int(np.clip(np.searchsorted(Fs, F) - 1, 0, len(V) - 2))
            (F1, _, x1), (F2, _, x2) = V[i], V[i + 1]
            a = float(np.clip((F - F1) / (F2 - F1), 0, 1)) if F2 > F1 else 1.0
            xs += (1 - a) * x1 + a * x2
        for l in range(cd.n_lanes):
            x[k, c, l, t, w] = float(xs[l])
    out = Result(res.status, res.message + f" (curve formulation; {n_curve_keys} uncached blocks; build {t1 - t0:.1f} s)",
                 -res.fun, z, q, x, y, o, time.perf_counter() - t0, n, r, int(integ.sum()),
                 getattr(res, "mip_gap", None))
    db = getattr(res, "mip_dual_bound", None)
    out.dual_bound = float(-db) if db is not None and np.isfinite(db) else float(-res.fun)  # max form, valid upper bound
    out.residual_max = check_solution(ins, out)
    return out
