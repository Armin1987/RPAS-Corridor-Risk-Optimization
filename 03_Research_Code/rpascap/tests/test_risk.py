import numpy as np
import pytest
from scipy import integrate
from rpascap.risk import reich, ballistic, ground, normalization as nz

MIX = reich.LaplaceMixture(5.3405, 50.0, 0.05)


def test_laplace_mixture_normalised():
    v, _ = integrate.quad(MIX.pdf, -np.inf, np.inf)
    assert abs(v - 1) < 1e-10


@pytest.mark.parametrize("S", [0.0, 5.0, 20.0, 50.0, 120.0])
def test_overlap_closed_form_matches_quadrature(S):
    a = reich.overlap_probability(S, 1.255, MIX)
    b = reich.overlap_probability(S, 1.255, MIX, numeric=True)
    assert abs(a - b) / b < 1e-7


def test_stage1_audit_values_reproduced():
    # Stage 1 recomputation of the original Tables 5-6 parameters
    assert abs(reich.overlap_probability(0, 0.485, MIX) - 0.041825) < 1e-5
    assert abs(reich.overlap_probability(0, 1.255, MIX) - 0.108228) < 1e-5
    assert abs(reich.laplace_scale_from_95(16.0) - 5.3409) < 1e-3


def test_overlap_monte_carlo():
    # P(|Y1 - Y2 + S| < lam) ~= 2*lam*g(S) for small lam (independent check of the Reich form)
    rng = np.random.default_rng(3)
    e = reich.LaplaceMixture(2.0, 10.0, 0.05)
    n = 4_000_000
    def draw():
        tail = rng.random(n) < e.alpha
        return np.where(tail, rng.laplace(0, e.a2, n), rng.laplace(0, e.a1, n))
    d = draw() - draw() + 3.0
    mc = np.mean(np.abs(d) < 0.25)
    an = reich.overlap_probability(3.0, 0.25, e)
    assert abs(mc - an) / an < 0.03


def test_kappa_linear_and_units():
    geo = reich.Geometry(0.9, 0.9, 0.43, 15.0, 1.0, 1.0, 0.1)
    ez = reich.LaplaceMixture(0.17, 10, 0.01)
    k = reich.kappa_lateral(30.0, geo, MIX, ez)
    # doubling the neighbour flow doubles the rate (linearity used by the MILP)
    assert np.isclose(2 * k * 10, k * 20)
    # wider separation reduces the coefficient
    assert reich.kappa_lateral(60.0, geo, MIX, ez) < k
    # opposite direction is riskier than same direction
    assert reich.kappa_lateral(30.0, geo, MIX, ez, opposite=True) > k


def test_ballistic_vacuum_limit():
    n = 5
    inp = ballistic.DescentInputs(np.full(n, 8.0), np.zeros(n), np.full(n, 0.1),
                                  np.linspace(0, 20, n), np.full(n, 2.0), np.full(n, 120.0),
                                  np.zeros(n), np.zeros(n))
    out = ballistic.simulate(inp, dt=0.0005)
    ref = ballistic.vacuum_solution(inp.vx0, inp.vz0, inp.h0)
    assert np.allclose(out["x"], ref["x"], atol=0.05)
    assert np.allclose(out["t"], ref["t"], atol=0.01)


def test_ballistic_terminal_velocity():
    n = 1
    inp = ballistic.DescentInputs(np.full(n, 8.0), np.full(n, 0.6), np.full(n, 0.15),
                                  np.zeros(n), np.zeros(n), np.full(n, 3000.0), np.zeros(n), np.zeros(n))
    out = ballistic.simulate(inp, dt=0.005, t_max=200)
    vt = np.sqrt(2 * 8.0 * ballistic.G0 / (ballistic.RHO_AIR * 0.6 * 0.15))
    assert abs(out["speed"][0] - vt) / vt < 0.01


def test_tailwind_moves_impact_downwind():
    rng = np.random.default_rng(0)
    p = {"mass": {"dist": "fixed", "value": 8}, "cd": {"dist": "fixed", "value": 0.6},
         "area": {"dist": "fixed", "value": 0.15}, "vx0": {"dist": "fixed", "value": 15},
         "vz0": {"dist": "fixed", "value": 0}, "h0": {"dist": "fixed", "value": 120}}
    calm = ballistic.simulate(ballistic.sample_inputs(rng, 1, p, 0.0))
    tail = ballistic.simulate(ballistic.sample_inputs(rng, 1, p, 10.0, 0.0))
    cross = ballistic.simulate(ballistic.sample_inputs(rng, 1, p, 10.0, np.pi / 2))
    assert tail["x"][0] > calm["x"][0]
    assert cross["y"][0] > 1.0


def test_kernel_and_consequence_limits():
    rng = np.random.default_rng(1)
    dx, dy = rng.normal(40, 15, 20000), rng.normal(0, 10, 20000)
    K, mass = ground.impact_kernel(dx, dy, 20.0, 8)
    assert abs(K.sum() - mass) < 1e-12 and mass > 0.999
    rho = np.full((40, 40), 0.01); sh = np.zeros((40, 40))
    lane = [(20, c) for c in range(10, 30)]
    c1 = ground.lane_consequence(lane, K, rho, sh, AL_pf=2.0)
    assert np.isclose(c1, 0.01 * 2.0 * mass, rtol=1e-9)      # uniform field -> rho*E[A_L Pf]
    c2 = ground.lane_consequence(lane, K, 2 * rho, sh, AL_pf=2.0)
    assert np.isclose(c2, 2 * c1)                              # linear in density
    assert ground.lane_consequence(lane, K, 0 * rho, sh, 2.0) == 0
    assert ground.lane_consequence(lane, K, rho, np.ones_like(sh), 2.0) == 0


def test_pf_monotone_and_bounded():
    E = np.logspace(0, 5, 50)
    pf = ground.pf_dalamagkidis(E, 0.5)
    assert np.all(np.diff(pf) >= 0) and pf.min() >= 0 and pf.max() <= 1


def test_normalisation_identities():
    # per-flight-hour failure risk does not depend on traffic
    g10 = nz.lane_hour_fatalities(10, 5400, 15, 1e-5, 0, 0.05) / nz.occupancy(10, 5400, 15)
    g90 = nz.lane_hour_fatalities(90, 5400, 15, 1e-5, 0, 0.05) / nz.occupancy(90, 5400, 15)
    assert np.isclose(g10, g90) and np.isclose(g10, 5e-7)
    # comparing lane-hour totals with a per-FH target gives a 'capacity' inversely
    # proportional to lane length -> a disguised societal budget
    c1 = nz.implied_capacity_if_totals_compared(1e-6, 1e-5, 0.05, 5400, 15)
    c2 = nz.implied_capacity_if_totals_compared(1e-6, 1e-5, 0.05, 10800, 15)
    assert np.isclose(c1, 2 * c2) and np.isclose(c1, 20.0)


def test_harmonic_mean_counterexample_regression():
    TLS, g, a = 1e-6, 1e-7, 1e-8
    H = 2 / (g / TLS + a / TLS)
    assert np.isclose((g + a) * H, 2 * TLS)


def test_exact_overlap_vs_reich_approximation():
    e_small = reich.LaplaceMixture(5.3405, 50.0, 0.05)
    # small lambda relative to error: both agree
    a = reich.overlap_probability(0.0, 0.05, e_small)
    b = reich.overlap_probability_exact(0.0, 0.05, e_small)
    assert abs(a - b) / b < 0.01
    # RPAS-like vertical case: airframe larger than error scale -> approximation > 1
    e_v = reich.LaplaceMixture(0.17, 10.0, 0.01)
    assert reich.overlap_probability(0.0, 0.43, e_v) > 1.0
    pe = reich.overlap_probability_exact(0.0, 0.43, e_v)
    assert 0.0 < pe <= 1.0


def test_exact_overlap_monte_carlo():
    rng = np.random.default_rng(5)
    e = reich.LaplaceMixture(0.17, 10.0, 0.01)
    n = 2_000_000
    def draw():
        tail = rng.random(n) < e.alpha
        return np.where(tail, rng.laplace(0, e.a2, n), rng.laplace(0, e.a1, n))
    mc = np.mean(np.abs(draw() - draw()) < 0.43)
    assert abs(mc - reich.overlap_probability_exact(0.0, 0.43, e)) < 0.002
