"""Run every experiment of the study and write raw results to results/raw/*.csv.

Usage:  python -m scripts.run_experiments --processed <dir> --out results [--stride 1] [--jobs 2]
        [--only main,nav,soc,dyn,sens,rob,comp]
Regimes (fixed before any real-data result was seen; see manuscript Sect. 6.2):
  R1  nav model E1-precise, no tactical-mitigation credit
  R2  nav model E2-measured, tactical-mitigation credit m = 0.9 (ASSUMED); chosen on synthetic
      development inputs as a regime in which the collision constraint binds, before any
      official data were processed
"""
from __future__ import annotations
import argparse, json, os, sys, time, platform, traceback
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpascap.study import Study, first_stage, evaluate, summarize_first_stage, CAP  # noqa: E402
import pickle  # noqa: E402

POLICIES = ["Static", "Deterministic", "SAA", "SeasonalSAA", "kNN", "RF", "Robust90", "WaitAndSee"]
REGIMES = {"R1": {"nav_model": "E1-precise", "mitigation": 0.0},
           "R2": {"nav_model": "E2-measured", "mitigation": 0.9}}

SOC_NSCEN = 10      # scenarios in the coupled societal-frontier MILPs (stated in the manuscript)
SOC_FRACS = [None, 1.0, 0.5, 0.25, 0.1]   # eps_t as a fraction of the unconstrained RF plan's hourly risk
EXT_TIME_LIMIT = 300  # s, extensive-form runs in the computational experiment
SOC_TIME_LIMIT = 60  # s per strict MILP; the time-sharing relaxation gets twice this
_ST = None


def get_study(processed, cfg, cache):
    global _ST
    if _ST is None:
        _ST = Study(Path(processed), cfg, Path(cache), log=lambda *a: None)
    return _ST


def run_day(processed, cfg, cache, exp, setting, par, policies, d, static_plan=None, extra=None):
    st = get_study(processed, cfg, cache)
    st.build_tables(par["nav_model"], par.get("gamma", cfg["gamma"]), cfg["n_mc_kernel"])
    extra = dict(extra or {})
    rows = []
    eps_ref = None
    if extra.get("eps_frac") is not None:
        # Hourly societal limit from DECISION-TIME information only: a fraction of the
        # scenario-weighted expected societal risk of the unconstrained RF plan for this day
        # (its own forecast scenarios; no realised data).  The same limit applies to every
        # policy on that day so that the frontier points are comparable.
        from rpascap.model.two_stage import recompute_risk
        ins0, r0 = first_stage(st, "RF", d, par, extra.get("n_scen", cfg["n_scenarios"]))
        Fh = recompute_risk(ins0, r0)["F_per_hour"] @ np.asarray(ins0.p)
        eps_ref = Fh
        extra["eps"] = np.maximum(extra["eps_frac"] * Fh, 1e-12)
    for pol in policies:
        rec = {"experiment": exp, "setting": setting, "policy": pol, "date": str(d.date())}
        if eps_ref is not None:
            rec["eps_ref_day"] = float(eps_ref.sum()); rec["eps_day"] = float(np.sum(extra["eps"]))
        try:
            if pol == "Static" and static_plan is not None:
                z, q, fs = static_plan
            else:
                ins, r = first_stage(st, pol, d, par, extra.get("n_scen", cfg["n_scenarios"]),
                                     eps=extra.get("eps"), menu_mask=extra.get("menu_mask"),
                                     max_switches=extra.get("max_switches"), force_milp=extra.get("force_milp", False),
                                     time_limit=extra.get("time_limit", 300))
                fs = summarize_first_stage(st, ins, r); z, q = r.z, r.q
            rec.update(fs)
            if extra.get("verify_milp") and pol not in ("Static",) and ((int(d.strftime('%Y%m%d')) + POLICIES.index(pol)) % 40 == 0):
                ins_m, r_m = first_stage(st, pol, d, par, extra.get("n_scen", cfg["n_scenarios"]), force_milp=True,
                                         menu_mask=extra.get("menu_mask"), max_switches=extra.get("max_switches"))
                rec.update(milp_check_obj=r_m.obj, milp_check_runtime_s=r_m.runtime_s, milp_check_gap=r_m.mip_gap,
                           milp_check_absdiff=abs(r_m.obj - fs["fs_obj"]))
            ev, _ = evaluate(st, d, z, q, par, eps=extra.get("eps"), menu_mask=extra.get("menu_mask"),
                             source=extra.get("admit_source", "realised"),
                             risk_source=extra.get("risk_source"), risk_par=extra.get("risk_par"),
                             max_switches=extra.get("max_switches"))
            rec.update(ev)
        except Exception as e:
            rec.update(error=repr(e), tb=traceback.format_exc()[-800:])
        rows.append(rec)
    return rows


def static_plan_for(processed, cfg, cache, par, extra=None):
    """Static benchmark: one configuration for the whole day and the same plan every day,
    optimised once on training scenarios (no forecast)."""
    st = get_study(processed, cfg, cache)
    st.build_tables(par["nav_model"], par.get("gamma", cfg["gamma"]), cfg["n_mc_kernel"])
    extra = extra or {}
    d0 = st.train[-1]                       # seed only; scenarios are random training days
    ins, r = first_stage(st, "Static", d0, par, cfg["n_scenarios"] * 2, eps=extra.get("eps"),
                         menu_mask=extra.get("menu_mask"))
    return r.z, r.q, summarize_first_stage(st, ins, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th test day for MAIN")
    ap.add_argument("--sub-stride", type=int, default=1, help="test-day stride for secondary experiments")
    ap.add_argument("--soc-stride", type=int, default=16, help="test-day stride for the societal (MILP) frontier")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default="main,nav,soc,dyn,sens,rob,comp")
    ap.add_argument("--nav-models", default=None, help="comma-separated subset for the nav experiment (chunked runs; merge raw files afterwards)")
    ap.add_argument("--sens-keys", default=None, help="comma-separated subset of sensitivity parameters ('base' for the base case)")
    ap.add_argument("--tag", default="", help="suffix for raw result files (e.g. _lead2)")
    ap.add_argument("--context-lead", type=int, default=None,
                    help="forecast lead (days) for the decision context: 1 (default, 24 h) or 2 (48 h; strict day-ahead check)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.context_lead is not None:
        cfg["context_lead_days"] = a.context_lead
    out = Path(a.out); (out / "raw").mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    st = Study(Path(a.processed), cfg, cache)
    st.build_tables(cfg["base"]["nav_model"], cfg["gamma"], cfg["n_mc_kernel"])
    test = list(st.test)
    main_days = test[::a.stride]; sub_days = test[::a.sub_stride]
    runlog = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "python": sys.version,
              "platform": platform.platform(), "cpu_count": os.cpu_count(), "args": vars(a),
              "n_test": len(test), "n_main": len(main_days), "n_sub": len(sub_days)}
    prev = out / "raw" / f"run_log{a.tag}.json"
    if prev.exists() and a.only != "main,nav,soc,dyn,sens,rob,comp":
        # partial run (--only): keep the log entries of earlier calls, record this call's args
        old = json.load(open(prev))
        old.setdefault("calls", []).append({"args": vars(a), "started_utc": runlog["started_utc"]})
        runlog = {**old, **{k: v for k, v in runlog.items() if k not in ("started_utc", "args")}}
    only = set(a.only.split(","))
    P = Parallel(n_jobs=a.jobs, verbose=0)

    # Block-capacity / curve cache shared across calls (results are deterministic functions of
    # the instance data, so caching changes run time only).  Effective with --jobs 1.
    for f in sorted(cache.glob("capcache_*.pkl")):
        try:
            CAP.d.update(pickle.load(open(f, "rb")))
        except Exception as e:                       # a damaged cache file is ignored
            print(f"ignoring {f.name}: {e!r}", flush=True)
    capfile = cache / f"capcache_{a.only.replace(',', '-')}{a.tag}.pkl"

    def go(name, tasks):
        t0 = time.time()
        res = P(delayed(run_day)(a.processed, cfg, str(cache), *t) for t in tasks)
        rows = [r for rr in res for r in rr]
        name = name + a.tag
        pd.DataFrame(rows).to_csv(out / "raw" / f"{name}.csv", index=False)
        runlog[f"{name}_seconds"] = round(time.time() - t0, 1)
        runlog[f"{name}_rows"] = len(rows)
        print(f"{name}: {len(rows)} rows in {time.time()-t0:.0f} s", flush=True)
        json.dump(runlog, open(out / "raw" / f"run_log{a.tag}.json", "w"), indent=1)
        if a.jobs == 1:
            tmp = capfile.with_suffix(".tmp"); pickle.dump(CAP.d, open(tmp, "wb")); tmp.replace(capfile)

    base = dict(cfg["base"])
    if "main" in only:
        for R, ov in REGIMES.items():
            par = {**base, **ov}
            sp = static_plan_for(a.processed, cfg, str(cache), par)
            go(f"main_{R}", [("main", R, par, POLICIES, d, sp, {"verify_milp": True}) for d in main_days])
    if "nav" in only:
        tasks = []
        for nav in (a.nav_models.split(",") if a.nav_models else ["E1-precise", "E2-measured", "E3-degraded", "E4-KimBae"]):
            for m in [0.0, 0.9, 0.99]:
                par = {**base, "nav_model": nav, "mitigation": m}
                tasks += [("nav", f"{nav}|m={m}", par, ["SeasonalSAA", "RF", "WaitAndSee"], d) for d in sub_days]
        go("nav", tasks)
    if "soc" in only:
        par = {**base, **REGIMES["R2"]}
        tasks = []
        for f in SOC_FRACS:
            tasks += [("soc", "none" if f is None else f"{f}", par, ["RF", "WaitAndSee"], d, None,
                       {"eps_frac": f, "n_scen": SOC_NSCEN}) for d in test[::a.soc_stride]]
        runlog["soc_eps_rule"] = "eps_t = f * expected societal risk in hour t of the unconstrained RF plan (decision-time scenarios)"
        runlog["soc_n_scen"] = SOC_NSCEN
        runlog["soc_fracs"] = SOC_FRACS
        runlog["soc_time_limit_s"] = SOC_TIME_LIMIT
        runlog["soc_formulation"] = "strict open-lane curve formulation (model/societal.py); certified gap from min(own dual bound, time-sharing relaxation optimum bound)"
        go("soc", tasks)
    if "dyn" in only:
        tasks = []
        for R in ["R1", "R2"]:
            par0 = {**base, **REGIMES[R]}
            for M in [0, 1, 2, 3]:
                for sc in [0.0, 5.0, 20.0]:
                    par = {**par0, "switch_cost": sc}
                    tasks += [("dyn", f"{R}|M={M}|sigma={sc}", par, ["RF", "WaitAndSee"], d, None, {"max_switches": M}) for d in sub_days]
            for menu_name, mask in {"no-dense": [1, 1, 1, 1, 0, 0], "single-only": [1, 1, 0, 0, 0, 0],
                                    "no-single": [1, 0, 1, 1, 1, 1]}.items():
                tasks += [("dyn", f"{R}|menu={menu_name}", par0, ["RF", "WaitAndSee"], d, None, {"menu_mask": mask}) for d in sub_days]
        go("dyn", tasks)
    if "sens" in only:
        par0 = {**base, **REGIMES["R2"]}
        grid = {"lambda_f": [1e-6, 3e-6, 3e-5, 1e-4], "theta_G": [1e-7, 1e-5], "theta_A": [1e-6, 1e-4],
                "ped_density": [0.005, 0.05, 0.1], "offpeak_scale": [0.5], "demand_scale": [0.5, 1.5, 2.0],
                "cancel_penalty": [1.0, 4.0], "gamma": [0.0, 0.2], "mitigation": [0.5, 0.99]}
        keys = a.sens_keys.split(",") if a.sens_keys else ["base"] + list(grid)
        tasks = [("sens", "base", par0, ["SeasonalSAA", "RF", "WaitAndSee"], d) for d in sub_days] if "base" in keys else []
        for k, vals in grid.items():
            if k not in keys:
                continue
            for v in vals:
                par = {**par0, k: v}
                tasks += [("sens", f"{k}={v}", par, ["SeasonalSAA", "RF", "WaitAndSee"], d) for d in sub_days]
        go("sens", tasks)
    if "rob" in only:
        par = {**base, **REGIMES["R2"]}
        tasks = []
        if st.SR is not None:
            tasks += [("rob", "nowcast-shortrange", par, ["RF", "SeasonalSAA", "Deterministic"], d, None,
                       {"admit_source": "shortrange", "risk_source": "realised"}) for d in sub_days]
            tasks += [("rob", "nowcast-dayahead", par, ["RF"], d, None,
                       {"admit_source": "forecast", "risk_source": "realised"}) for d in sub_days]
        for nav in ["E3-degraded", "E4-KimBae"]:
            tasks += [("rob", f"misspec-{nav}", par, ["RF"], d, None,
                       {"risk_par": {**par, "nav_model": nav}}) for d in sub_days]
        for n in [10, 20, 50, 80]:
            tasks += [("rob", f"nscen={n}", par, ["RF", "SeasonalSAA"], d, None, {"n_scen": n}) for d in test[::8]]
        go("rob", tasks)
    if "comp" in only:
        par = {**base, **REGIMES["R2"]}
        tasks = []
        for n in [10, 30, 60, 100, 150]:
            tasks += [("comp", f"fast|nscen={n}", par, ["SAA"], d, None, {"n_scen": n}) for d in test[5:10]]
        for n in [10, 30, 60]:      # block-form MILP (same model; configuration binaries only)
            tasks += [("comp", f"milp|nscen={n}", par, ["SAA"], d, None, {"n_scen": n, "force_milp": True}) for d in test[5:10]]
        # big-M extensive form with lane indicators: documents that it does not solve at study scale
        tasks += [("comp", "extensive|nscen=10", par, ["SAA"], d, None,
                   {"n_scen": 10, "force_milp": "extensive", "time_limit": EXT_TIME_LIMIT}) for d in test[5:8]]
        runlog["extensive_time_limit_s"] = EXT_TIME_LIMIT
        go("comp", tasks)
    runlog["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(runlog, open(out / "raw" / "run_log.json", "w"), indent=1)


if __name__ == "__main__":
    main()
