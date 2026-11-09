"""Two-stage stochastic corridor-configuration and flight-admission model (extensive form).

FIRST STAGE (day-ahead; context known, realised weather unknown)
  z[k,c,b] in {0,1}   configuration c of corridor k in block b (c may be 'closed')
  w[k,c,b] in [0,1]   switch-on indicator (integral at optimum)
  q[k,t]   >= 0       pre-approved planned flights for hour t, q <= P[k,t] (planned
                      requests, known day-ahead)
SECOND STAGE (hour t, after observing hour-t wind and on-demand requests; scenario w)
  x[k,c,l,t,w] >= 0   admitted flights/h on lane l of configuration c
  y[k,t,w]     >= 0   pre-approved flights actually flown (y <= q)
  o[k,t,w]     >= 0   on-demand flights admitted (o <= O[k,t,w])
OBJECTIVE
  max  sum_w p_w sum_{k,t} [ v (y + o) - pi (q - y) ]  - sigma*sum w - sum f*z
CONSTRAINTS (all linear -> MILP; binaries z (first stage) and u (recourse))
  (1) sum_c z[k,c,b] = 1
  (2) w[k,c,b] >= z[k,c,b] - z[k,c,b-1]; sum_{c,b>=1} w <= M
  u[k,c,l,t,w] in {0,1} lane l open (carries traffic) in hour t, scenario w (recourse)
  (3) x <= U[k,c,t,w,l] * u ;  u <= z[k,c,b(t)]     (U = 0: lane unusable in (t,w))
  (4) sum_{c,l} x = y + o ;  y <= q ;  o <= O[k,t,w] ;  q <= P[k,t]
  (5) ground, per aircraft flight hour, imposed on OPEN lanes only (big-M on u):
        C_l * (1-m) sum_l' kappa_ll' x_l' <= theta_G - lambda_f C_l + M5_l (1 - u_l)
  (6) air, per aircraft flight hour, open lanes only:
              (1-m) sum_l' kappa_ll' x_l' <= theta_A + M6_l (1 - u_l)
      M_l = the largest value the left-hand side can take given x <= U (tight big-M).
      An empty lane carries no aircraft, so no per-aircraft target applies to it.
  (7) optional societal epsilon-constraint per hour (conservative linear bound):
        sum tau_k (lambda_f + theta_A) C_l x <= eps_t
 m = credited tactical mitigation (0 in the base case).
Hours are not coupled in the recourse, so the two-stage form is exactly nonanticipative.
Risk rows are divided by their targets (HiGHS uses absolute tolerances).
"""
from __future__ import annotations
import itertools
import time
import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy import sparse


@dataclass
class ConfigData:
    n_lanes: int
    fixed_cost: float
    U: np.ndarray        # (T, W, n) flights/h cap (0 = unusable)
    C: np.ndarray        # (T, W, n) fatalities per crash
    kappa: np.ndarray    # (T, W, n, n) collision involvements per FH per (fl/h)
    name: str = ""


@dataclass
class Instance:
    configs: list            # [k][c] ConfigData
    O: np.ndarray            # (K, T, W) on-demand requests per hour
    P: np.ndarray            # (K, T) planned requests per hour (known day-ahead)
    p: np.ndarray            # (W,) scenario weights
    block_of_t: np.ndarray   # (T,)
    tau_h: np.ndarray        # (K,) hours on corridor per flight
    value: float = 1.0
    cancel_penalty: float = 2.0
    switch_cost: float = 5.0
    max_switches: int = 2
    theta_G: float = 1e-6
    theta_A: float = 1e-5
    lambda_f: float = 1e-5
    mitigation: float = 0.0
    eps: np.ndarray | None = None

    @property
    def K(self): return len(self.configs)
    @property
    def T(self): return self.O.shape[1]
    @property
    def W(self): return self.O.shape[2]
    @property
    def B(self): return int(self.block_of_t.max()) + 1


class Index:
    def __init__(self, ins: Instance):
        n = 0
        self.z, self.w, self.q, self.x, self.y, self.o, self.u = {}, {}, {}, {}, {}, {}, {}
        for k, cfgs in enumerate(ins.configs):
            for c in range(len(cfgs)):
                for b in range(ins.B):
                    self.z[k, c, b] = n; n += 1
        for k, cfgs in enumerate(ins.configs):
            for c in range(len(cfgs)):
                for b in range(ins.B):
                    self.w[k, c, b] = n; n += 1
        for k in range(ins.K):
            for t in range(ins.T):
                self.q[k, t] = n; n += 1
        for k, cfgs in enumerate(ins.configs):
            for c, cd in enumerate(cfgs):
                for l in range(cd.n_lanes):
                    for t in range(ins.T):
                        for w in range(ins.W):
                            self.x[k, c, l, t, w] = n; n += 1
        for k in range(ins.K):
            for t in range(ins.T):
                for w in range(ins.W):
                    self.y[k, t, w] = n; n += 1
                    self.o[k, t, w] = n; n += 1
        for k, cfgs in enumerate(ins.configs):
            for c, cd in enumerate(cfgs):
                for l in range(cd.n_lanes):
                    for t in range(ins.T):
                        for w in range(ins.W):
                            if usable(ins, cd, t, w, l):
                                self.u[k, c, l, t, w] = n; n += 1
        self.n = n


@dataclass
class Result:
    status: int
    message: str
    obj: float
    z: dict
    q: dict
    x: dict
    y: dict
    o: dict
    runtime_s: float
    n_vars: int
    n_cons: int
    n_int: int
    mip_gap: float | None
    residual_max: float = field(default=np.nan)


ON_TOL = 1e-4   # flights/h; a lane carrying less is treated as empty (solver integrality tolerance)


def usable(ins: Instance, cd: ConfigData, t, w, l) -> bool:
    return cd.U[t, w, l] > 0 and ins.lambda_f * cd.C[t, w, l] <= ins.theta_G


def build(ins: Instance, fixed_z=None, fixed_q=None):
    I = Index(ins)
    cost = np.zeros(I.n)
    lb = np.zeros(I.n); ub = np.full(I.n, np.inf); integ = np.zeros(I.n)
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    mf = 1.0 - ins.mitigation

    def add(entries, l, h):
        nonlocal r
        for j, v in entries:
            rows.append(r); cols.append(j); vals.append(v)
        lo.append(l); hi.append(h); r += 1

    for (k, c, b), j in I.z.items():
        cost[j] = ins.configs[k][c].fixed_cost
        ub[j] = 1; integ[j] = 1
        if fixed_z is not None:
            lb[j] = ub[j] = fixed_z[k, c, b]; integ[j] = 0
    for (k, c, b), j in I.w.items():
        cost[j] = ins.switch_cost if b >= 1 else 0.0
        ub[j] = 1 if b >= 1 else 0
    for (k, t), j in I.q.items():
        cost[j] = ins.cancel_penalty            # -(-pi q) in minimisation form
        ub[j] = ins.P[k, t]
        if fixed_q is not None:
            lb[j] = ub[j] = fixed_q[k, t]
    for (k, t, w), j in I.y.items():
        cost[j] = -ins.p[w] * (ins.value + ins.cancel_penalty)
    for (k, t, w), j in I.o.items():
        cost[j] = -ins.p[w] * ins.value
        ub[j] = ins.O[k, t, w]

    for k, cfgs in enumerate(ins.configs):
        for b in range(ins.B):
            add([(I.z[k, c, b], 1.0) for c in range(len(cfgs))], 1.0, 1.0)            # (1)
        for c in range(len(cfgs)):
            for b in range(1, ins.B):
                add([(I.w[k, c, b], 1.0), (I.z[k, c, b], -1.0), (I.z[k, c, b - 1], 1.0)], 0.0, np.inf)
        add([(I.w[k, c, b], 1.0) for c in range(len(cfgs)) for b in range(1, ins.B)], -np.inf,
            ins.max_switches)                                                             # (2)

    for (k, c, l, t, w), j in I.u.items():
        ub[j] = 1; integ[j] = 1
    for k, cfgs in enumerate(ins.configs):
        for c, cd in enumerate(cfgs):
            for t in range(ins.T):
                b = ins.block_of_t[t]
                for w in range(ins.W):
                    Uok = np.array([cd.U[t, w, l] if usable(ins, cd, t, w, l) else 0.0 for l in range(cd.n_lanes)])
                    for l in range(cd.n_lanes):
                        j = I.x[k, c, l, t, w]
                        if Uok[l] <= 0:
                            ub[j] = 0.0
                            continue
                        ju = I.u[k, c, l, t, w]
                        add([(j, 1.0), (ju, -Uok[l])], -np.inf, 0.0)                     # (3) x <= U u
                        add([(ju, 1.0), (I.z[k, c, b], -1.0)], -np.inf, 0.0)               # (3) u <= z
                        Cl = cd.C[t, w, l]
                        kap = mf * cd.kappa[t, w, l]
                        nz = [(I.x[k, c, l2, t, w], kap[l2]) for l2 in range(cd.n_lanes) if kap[l2] > 0 and Uok[l2] > 0]
                        if nz:
                            lhs_max = float(sum(kap[l2] * Uok[l2] for l2 in range(cd.n_lanes) if Uok[l2] > 0))
                            M5 = Cl * lhs_max / ins.theta_G
                            add([(jj, Cl * v / ins.theta_G) for jj, v in nz] + [(ju, M5)], -np.inf,
                                1.0 - ins.lambda_f * Cl / ins.theta_G + M5)               # (5)
                            M6 = lhs_max / ins.theta_A
                            add([(jj, v / ins.theta_A) for jj, v in nz] + [(ju, M6)], -np.inf, 1.0 + M6)  # (6)

    for k, cfgs in enumerate(ins.configs):
        for t in range(ins.T):
            for w in range(ins.W):
                xs = [(I.x[k, c, l, t, w], 1.0) for c, cd in enumerate(cfgs) for l in range(cd.n_lanes)]
                add(xs + [(I.y[k, t, w], -1.0), (I.o[k, t, w], -1.0)], 0.0, 0.0)          # (4)
                add([(I.y[k, t, w], 1.0), (I.q[k, t], -1.0)], -np.inf, 0.0)

    if ins.eps is not None:                                                               # (7)
        for t in range(ins.T):
            for w in range(ins.W):
                ent = []
                for k, cfgs in enumerate(ins.configs):
                    for c, cd in enumerate(cfgs):
                        for l in range(cd.n_lanes):
                            coef = ins.tau_h[k] * (ins.lambda_f + ins.theta_A) * cd.C[t, w, l] / ins.eps[t]
                            if coef > 0:
                                ent.append((I.x[k, c, l, t, w], coef))
                if ent:
                    add(ent, -np.inf, 1.0)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, I.n))
    return I, cost, A, np.array(lo), np.array(hi), lb, ub, integ


def solve(ins: Instance, time_limit=600, mip_rel_gap=1e-6, fixed_z=None, fixed_q=None) -> Result:
    I, cost, A, lo, hi, lb, ub, integ = build(ins, fixed_z, fixed_q)
    t0 = time.perf_counter()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), bounds=Bounds(lb, ub), integrality=integ,
               options={"time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False})
    rt = time.perf_counter() - t0
    if res.x is None:
        return Result(res.status, res.message, np.nan, {}, {}, {}, {}, {}, rt, I.n, A.shape[0], int(integ.sum()), None)
    xv = res.x
    ax = A @ xv
    resid = float(max(np.max(np.maximum(lo - ax, 0), initial=0), np.max(np.maximum(ax - hi, 0), initial=0),
                      np.max(np.maximum(lb - xv, 0)), np.max(np.maximum(xv - ub, 0))))
    out = Result(res.status, res.message, -res.fun,
                  {kk: int(round(xv[j])) for kk, j in I.z.items()},
                  {kk: float(xv[j]) for kk, j in I.q.items()},
                  {kk: float(xv[j]) for kk, j in I.x.items()},
                  {kk: float(xv[j]) for kk, j in I.y.items()},
                  {kk: float(xv[j]) for kk, j in I.o.items()},
                  rt, I.n, A.shape[0], int(integ.sum()), getattr(res, "mip_gap", None), resid)
    out.matrix_residual = resid
    out.residual_max = max(resid, check_solution(ins, out))
    return out


def check_solution(ins: Instance, res: Result) -> float:
    """Independent feasibility check of a solution against the model's constraints, computed
    from the solution values and the instance data (not from the constraint matrix).
    Returns the largest violation; risk rows are measured relative to their targets, flow
    rows in flights/h.  Risk targets are checked on lanes that carry traffic."""
    mf = 1.0 - ins.mitigation
    v = 0.0
    for k, cfgs in enumerate(ins.configs):
        for b in range(ins.B):
            v = max(v, abs(sum(res.z.get((k, c, b), 0) for c in range(len(cfgs))) - 1))
        sw = sum(1 for b in range(1, ins.B)
                 if max(range(len(cfgs)), key=lambda c: res.z.get((k, c, b), 0)) !=
                 max(range(len(cfgs)), key=lambda c: res.z.get((k, c, b - 1), 0)))
        v = max(v, sw - ins.max_switches)
        for t in range(ins.T):
            b = ins.block_of_t[t]
            qq = res.q.get((k, t), 0.0)
            v = max(v, qq - ins.P[k, t], -qq)
            for w in range(ins.W):
                tot = 0.0
                for c, cd in enumerate(cfgs):
                    x = np.array([res.x.get((k, c, l, t, w), 0.0) for l in range(cd.n_lanes)])
                    if x.size == 0:
                        continue
                    tot += x.sum()
                    Uok = np.array([cd.U[t, w, l] if usable(ins, cd, t, w, l) else 0.0 for l in range(cd.n_lanes)])
                    v = max(v, float(np.max(x - Uok * res.z.get((k, c, b), 0))), float(np.max(-x)))
                    on = x > ON_TOL
                    if on.any():
                        rr = mf * cd.kappa[t, w] @ x
                        g = (ins.lambda_f + rr) * cd.C[t, w]
                        v = max(v, float((g[on] / ins.theta_G).max()) - 1, float((rr[on] / ins.theta_A).max()) - 1)
                yy, oo = res.y.get((k, t, w), 0.0), res.o.get((k, t, w), 0.0)
                v = max(v, abs(tot - yy - oo), yy - qq, oo - ins.O[k, t, w], -yy, -oo)
    if ins.eps is not None:
        for t in range(ins.T):
            for w in range(ins.W):
                s = sum(ins.tau_h[k] * (ins.lambda_f + ins.theta_A) * cd.C[t, w, l] * res.x.get((k, c, l, t, w), 0.0)
                        for k, cfgs in enumerate(ins.configs) for c, cd in enumerate(cfgs) for l in range(cd.n_lanes))
                v = max(v, s / ins.eps[t] - 1)
    return float(max(v, 0.0))


def plans_for(ins: Instance, k: int):
    out = []
    for plan in itertools.product(range(len(ins.configs[k])), repeat=ins.B):
        if sum(plan[b] != plan[b - 1] for b in range(1, ins.B)) <= ins.max_switches:
            out.append(plan)
    return out


def z_from_plans(ins: Instance, plans) -> dict:
    fz = {}
    for k, plan in enumerate(plans):
        for c in range(len(ins.configs[k])):
            for b in range(ins.B):
                fz[k, c, b] = 1.0 if plan[b] == c else 0.0
    return fz


def brute_force(ins: Instance):
    best, best_z = -np.inf, None
    for combo in itertools.product(*[plans_for(ins, k) for k in range(ins.K)]):
        fz = z_from_plans(ins, combo)
        r = solve(ins, fixed_z=fz)
        if r.status == 0 and r.obj > best:
            best, best_z = r.obj, fz
    return best, best_z


def recompute_risk(ins: Instance, res: Result):
    """Independent re-computation of per-lane risk from the solution (not from the
    constraint matrix).  Returns dict of maxima and per-hour societal expected fatalities
    (exact, including the bilinear collision term)."""
    mf = 1.0 - ins.mitigation
    max_g = max_a = 0.0
    F = np.zeros((ins.T, ins.W))
    for k, cfgs in enumerate(ins.configs):
        for c, cd in enumerate(cfgs):
            for t in range(ins.T):
                for w in range(ins.W):
                    x = np.array([res.x.get((k, c, l, t, w), 0.0) for l in range(cd.n_lanes)])
                    if x.size == 0 or x.max() <= ON_TOL:
                        continue
                    rr = mf * cd.kappa[t, w] @ x
                    g = (ins.lambda_f + rr) * cd.C[t, w]
                    F[t, w] += float((x * ins.tau_h[k] * g).sum())
                    on = x > ON_TOL
                    bad = np.array([not usable(ins, cd, t, w, l) for l in range(cd.n_lanes)])
                    gr = np.where(bad, np.inf, g)           # flown although unusable under this information
                    max_g = max(max_g, float((gr[on] / ins.theta_G).max()))
                    max_a = max(max_a, float((rr[on] / ins.theta_A).max()))
    return {"max_ground_ratio": max_g, "max_air_ratio": max_a, "F_per_hour": F}


def _sub(ins: Instance, k: int) -> Instance:
    import copy
    s = copy.copy(ins)
    s.configs = [ins.configs[k]]; s.O = ins.O[k:k + 1]; s.P = ins.P[k:k + 1]; s.tau_h = ins.tau_h[k:k + 1]
    return s


def solve_decomposed(ins: Instance, time_limit=600, mip_rel_gap=1e-6, fixed_z=None, fixed_q=None) -> Result:
    """Exact corridor-wise decomposition.  Without the societal constraint (eps is None)
    no constraint links corridors, so the extensive form separates into K independent
    MILPs; the optimum is the sum of the corridor optima (verified in tests)."""
    if ins.eps is not None or ins.K == 1:
        return solve(ins, time_limit, mip_rel_gap, fixed_z, fixed_q)
    parts = []
    for k in range(ins.K):
        fz = None if fixed_z is None else {(0, c, b): v for (kk, c, b), v in fixed_z.items() if kk == k}
        fq = None if fixed_q is None else {(0, t): v for (kk, t), v in fixed_q.items() if kk == k}
        parts.append(solve(_sub(ins, k), time_limit / ins.K, mip_rel_gap, fz, fq))
    def remap(d, k):
        return {(k,) + key[1:]: v for key, v in d.items()}
    st = max(p.status for p in parts)
    if any(p.status not in (0, 1) or not p.z and fixed_z is None for p in parts):
        bad = next(p for p in parts if p.status != 0)
        return Result(bad.status, bad.message, np.nan, {}, {}, {}, {}, {}, sum(p.runtime_s for p in parts),
                      sum(p.n_vars for p in parts), sum(p.n_cons for p in parts), sum(p.n_int for p in parts), None)
    out = Result(st, "; ".join(sorted({p.message for p in parts})), float(sum(p.obj for p in parts)),
                 {}, {}, {}, {}, {}, float(sum(p.runtime_s for p in parts)), sum(p.n_vars for p in parts),
                 sum(p.n_cons for p in parts), sum(p.n_int for p in parts),
                 max([p.mip_gap or 0.0 for p in parts]), float(max(p.residual_max for p in parts)))
    for k, p in enumerate(parts):
        out.z.update(remap(p.z, k)); out.q.update(remap(p.q, k)); out.x.update(remap(p.x, k))
        out.y.update(remap(p.y, k)); out.o.update(remap(p.o, k))
    return out
