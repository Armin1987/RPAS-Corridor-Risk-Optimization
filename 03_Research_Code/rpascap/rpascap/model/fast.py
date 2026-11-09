"""Exact structured algorithm for the two-stage model without the societal constraint.

Structure (proved in the manuscript, Proposition 4; verified against the MILP in tests):
 (a) Corridors share no constraint -> solve each corridor separately.
 (b) Given configuration c in hour t and scenario w, the recourse depends on the lane
     flows only through their total.  Risk rows apply to open lanes only, so the feasible
     set is the union over open-lane sets S of the polytopes X_S = {x : x_l = 0 for l not in
     S, x <= U, rows (5),(6) for l in S}.  Each X_S is defined by rows homogeneous in x with
     non-negative right-hand sides (lanes with negative right-hand side are unusable), so
     its totals form [0, F_S] (scale its maximiser down; scaling keeps the support inside
     S).  Hence the feasible totals form [0, F*] with F* = max_S F_S.  F_S <= sum_{l in S}
     U_l, which prunes the enumeration (at most 2^6 - 1 subsets for the largest menu item).
 (c) With total capacity F, the hour value is
         V(q; F, O) = (v+pi) min(q, F) + v min(O, max(F - q, 0)),
     so the pre-approval value g_t(q) = -pi q + sum_w p_w V(q; F_w, O_w) is concave and
     piecewise linear in q; its maximum over [0, P_t] lies in the finite breakpoint set
     {0, P_t, F_w, F_w - O_w}.
 (d) Block values H(b, c) = sum_{t in b} max_q g_t(q) - f_c; the configuration plan with
     switching cost sigma and at most M switches is a dynamic programme over blocks.
The result is the exact optimum of the extensive-form MILP (same model, no approximation).
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import linprog
from .two_stage import Instance, Result, usable, check_solution


class CapacityCache:
    def __init__(self):
        self.d = {}

    def get(self, key, fn):
        if key not in self.d:
            self.d[key] = fn()
        return self.d[key]


def _lp_subset(S, ub, K, C, ins):
    n = ub.size
    rows, rhs = [], []
    for l in S:
        rows.append(C[l] * K[l] / ins.theta_G); rhs.append(1.0 - ins.lambda_f * C[l] / ins.theta_G)
        rows.append(K[l] / ins.theta_A); rhs.append(1.0)
    A = np.array(rows); b = np.array(rhs)
    keep = np.abs(A).sum(axis=1) > 0
    bnd = [(0.0, ub[l] if l in S else 0.0) for l in range(n)]
    res = linprog(-np.ones(n), A_ub=A[keep] if keep.any() else None, b_ub=b[keep] if keep.any() else None,
                  bounds=bnd, method="highs")
    if res.status != 0:
        raise RuntimeError(f"capacity LP failed: {res.message}")
    x = np.clip(res.x, 0, [hi for _, hi in bnd])
    return float(x.sum()), x


def lane_capacity_lp(cd, t, w, ins: Instance):
    """Max total flow and a maximising flow vector for configuration cd in (t, w), with
    per-aircraft risk rows imposed on the lanes that are open (carry traffic)."""
    from itertools import combinations
    n = cd.n_lanes
    if n == 0:
        return 0.0, np.zeros(0)
    ub = np.array([cd.U[t, w, l] if usable(ins, cd, t, w, l) else 0.0 for l in range(n)])
    if ub.max() <= 0:
        return 0.0, np.zeros(n)
    K = (1.0 - ins.mitigation) * cd.kappa[t, w]
    C = cd.C[t, w]
    lanes = [l for l in range(n) if ub[l] > 0]
    subsets = [S for r in range(len(lanes), 0, -1) for S in combinations(lanes, r)]
    subsets.sort(key=lambda S: -ub[list(S)].sum())
    best, bx = 0.0, np.zeros(n)
    for S in subsets:
        if ub[list(S)].sum() <= best + 1e-9:
            break                                    # remaining subsets cannot do better
        F, x = _lp_subset(set(S), ub, K, C, ins)
        if F > best + 1e-12:
            best, bx = F, x
    return best, bx


def hour_value(q, F, O, v, pi):
    return (v + pi) * np.minimum(q, F) + v * np.minimum(O, np.maximum(F - q, 0.0))


def best_q(P_t, F, O, p, v, pi):
    cand = np.unique(np.clip(np.concatenate([[0.0, P_t], F, F - O]), 0.0, P_t))
    vals = np.array([-pi * q + np.dot(p, hour_value(q, F, O, v, pi)) for q in cand])
    i = int(np.argmax(vals))
    return float(cand[i]), float(vals[i])


def solve_fast(ins: Instance, cache: CapacityCache | None = None, cap_key=None, fixed_z=None, fixed_q=None):
    """Exact solution (or evaluation of a fixed first stage).  cap_key(k, c, t, w) may return
    a hashable key under which capacity LPs are cached across instances (same coefficients)."""
    t0 = time.perf_counter()
    cache = cache or CapacityCache()
    K, T, W, B = ins.K, ins.T, ins.W, ins.B
    v, pi = ins.value, ins.cancel_penalty
    z, q, x, y, o = {}, {}, {}, {}, {}
    total = 0.0
    n_lp = 0
    for k in range(K):
        cfgs = ins.configs[k]
        Fcap = np.zeros((len(cfgs), T, W)); Xcap = {}
        for c, cd in enumerate(cfgs):
            for t in range(T):
                for w in range(W):
                    key = None if cap_key is None else cap_key(k, c, t, w)
                    if key is None:
                        F, xv = lane_capacity_lp(cd, t, w, ins); n_lp += 1
                    else:
                        F, xv = cache.get(key, lambda: lane_capacity_lp(cd, t, w, ins))
                    Fcap[c, t, w] = F; Xcap[c, t, w] = xv
        # hour values and best pre-approvals per configuration
        G = np.zeros((len(cfgs), T)); Q = np.zeros((len(cfgs), T))
        for c in range(len(cfgs)):
            for t in range(T):
                if fixed_q is not None:
                    qq = fixed_q[k, t]
                    G[c, t] = -pi * qq + np.dot(ins.p, hour_value(qq, Fcap[c, t], ins.O[k, t], v, pi)); Q[c, t] = qq
                else:
                    Q[c, t], G[c, t] = best_q(ins.P[k, t], Fcap[c, t], ins.O[k, t], ins.p, v, pi)
        H = np.zeros((B, len(cfgs)))
        for b in range(B):
            ts = np.flatnonzero(ins.block_of_t == b)
            H[b] = G[:, ts].sum(axis=1) - np.array([cd.fixed_cost for cd in cfgs])
        # dynamic programme over blocks: state (config, switches used)
        if fixed_z is not None:
            plan = [max(range(len(cfgs)), key=lambda c: fixed_z[k, c, b]) for b in range(B)]
            val = sum(H[b, plan[b]] for b in range(B)) - ins.switch_cost * sum(plan[b] != plan[b - 1] for b in range(1, B))
        else:
            M = ins.max_switches; nc = len(cfgs)
            V = np.full((nc, M + 1), -np.inf); V[:, 0] = H[0]
            back = [None]
            for b in range(1, B):
                Vn = np.full((nc, M + 1), -np.inf); bk = np.zeros((nc, M + 1, 2), int)
                for c in range(nc):
                    for m in range(M + 1):
                        best, arg = V[c, m], (c, m)                     # stay
                        if m >= 1:
                            prev = V[:, m - 1].copy(); prev[c] = -np.inf
                            j = int(np.argmax(prev))
                            if prev[j] - ins.switch_cost > best:
                                best, arg = prev[j] - ins.switch_cost, (j, m - 1)
                        Vn[c, m] = best + H[b, c]; bk[c, m] = arg
                V = Vn; back.append(bk)
            c, m = np.unravel_index(int(np.argmax(V)), V.shape)
            val = float(V[c, m]); plan = [0] * B; plan[-1] = int(c)
            for b in range(B - 1, 0, -1):
                c, m = back[b][c, m]; plan[b - 1] = int(c)
        total += val
        for c in range(len(cfgs)):
            for b in range(B):
                z[k, c, b] = 1 if plan[b] == c else 0
        for t in range(T):
            c = plan[ins.block_of_t[t]]
            q[k, t] = float(Q[c, t])
            for w in range(W):
                F = Fcap[c, t, w]
                yy = min(q[k, t], F); oo = min(ins.O[k, t, w], max(F - yy, 0.0))
                y[k, t, w] = yy; o[k, t, w] = oo
                xs = Xcap[c, t, w]
                scale = (yy + oo) / F if F > 0 else 0.0
                for l in range(len(xs)):
                    x[k, c, l, t, w] = float(xs[l] * scale)
    rt = time.perf_counter() - t0
    out = Result(0, f"exact structured algorithm ({n_lp} uncached capacity LPs)", total, z, q, x, y, o, rt,
                 0, 0, 0, None, np.nan)
    out.residual_max = check_solution(ins, out)       # independent check, not assumed
    return out
