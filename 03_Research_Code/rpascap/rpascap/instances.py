"""Turn scenario days into optimisation instances, and the synthetic demand model.

DEMAND IS SYNTHETIC (no drone-delivery demand data exist for Athens).  It is generated
from a transparent, parameterised model and is identical for every policy:
  planned requests  P[k,t] = round(phi * Lambda_k * prof(t))         (known day-ahead)
  on-demand         O[k,t] ~ Poisson((1-phi) * Lambda_k * prof(t))    (seed = date)
prof(t): two-peak daily profile (late-morning/lunch and evening), max 1.
"""
from __future__ import annotations
import numpy as np
from .model.two_stage import ConfigData, Instance
from .tables import bins


def profile(hours):
    h = np.asarray(hours, float)
    p = 0.35 + 0.65 * np.exp(-0.5 * ((h - 12.5) / 1.6) ** 2) + 0.55 * np.exp(-0.5 * ((h - 19.5) / 1.5) ** 2)
    return p / p.max()


def planned(hours, Lam, phi):
    return np.rint(phi * np.outer(Lam, profile(hours)))


def on_demand(date_key: int, hours, Lam, phi, seed=2026):
    rng = np.random.default_rng([seed, int(date_key)])
    return rng.poisson((1 - phi) * np.outer(Lam, profile(hours))).astype(float)


def build_instance(tabs, menu, days, weights, P, O_list, par, block_of_t, tau_h, eps=None,
                   max_switches=None, menu_mask=None):
    """days: list of (speed[T], from_dir[T]); O_list: list of (K,T) arrays."""
    Ctab, Ktab = tabs["C"], tabs["K"]
    K, T, Wn = len(Ctab), len(block_of_t), len(days)
    sb = np.zeros((Wn, T), int); db = np.zeros((Wn, T), int); us = np.zeros((Wn, T), bool)
    for w, (spd, fdir) in enumerate(days):
        sb[w], db[w], us[w] = bins(np.asarray(spd), np.asarray(fdir))
    cap_factor = 3600.0 * par["V"]
    configs = []
    for k in range(K):
        row = []
        for c, cfg in enumerate(menu):
            if menu_mask is not None and not menu_mask[c]:
                continue
            n = cfg.n_lanes
            U = np.zeros((T, Wn, n)); C = np.zeros((T, Wn, n)); KA = np.zeros((T, Wn, n, n))
            if n:
                for w in range(Wn):
                    for t in range(T):
                        C[t, w] = Ctab[k][c][sb[w, t], db[w, t]]
                        KA[t, w] = Ktab[k][c][sb[w, t]]
                        U[t, w] = cap_factor / cfg.S_x if us[w, t] else 0.0
            row.append(ConfigData(n, cfg.fixed_cost, U, C, KA, cfg.name))
        configs.append(row)
    O = np.stack(O_list, axis=2)           # (K, T, W)
    return Instance(configs=configs, O=O, P=P, p=np.asarray(weights, float), block_of_t=block_of_t,
                    tau_h=tau_h, value=par["value"], cancel_penalty=par["cancel_penalty"],
                    switch_cost=par["switch_cost"],
                    max_switches=par["max_switches"] if max_switches is None else max_switches,
                    theta_G=par["theta_G"], theta_A=par["theta_A"], lambda_f=par["lambda_f"],
                    mitigation=par.get("mitigation", 0.0), eps=eps)
