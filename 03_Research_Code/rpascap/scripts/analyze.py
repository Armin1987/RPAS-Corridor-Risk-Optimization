"""Statistics and tables from results/raw/*.csv  ->  results/tables/*.csv (+ summary.json).

Paired comparisons use the same test days for both policies; 95 % CIs are percentile
intervals from a circular moving-block bootstrap over date-ordered test days (block
length 7 days, to respect serial dependence of weather; 10,000 resamples, fixed seed).
The independent-day bootstrap is reported alongside for comparison.  Policy means in the
main experiment are computed on the days on which every policy was solved.  No other tests.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import pandas as pd
from pathlib import Path

B = 10_000
RNG = np.random.default_rng(20261107)


BLOCK = 7


def boot_ci(x, block=BLOCK):
    """Circular moving-block bootstrap of the mean (block=1: independent resampling).
    x must be ordered by date."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    n = x.size
    if n < 2:
        return (np.nan, np.nan, np.nan)
    L = max(1, min(block, n))
    nb = int(np.ceil(n / L))
    starts = RNG.integers(0, n, (B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n] % n
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(df, a, b, metric, block=BLOCK):
    p = df.pivot_table(index="date", columns="policy", values=metric).sort_index()
    if a not in p or b not in p:
        return (np.nan,) * 4
    d = (p[a] - p[b]).dropna()
    m, lo, hi = boot_ci(d.values, block)
    return m, lo, hi, int(d.size)


def common_days(df):
    """Restrict to dates on which every policy present has a solved row."""
    npol = df.policy.nunique()
    ok = df.groupby("date").policy.nunique()
    return df[df.date.isin(ok[ok == npol].index)]


def load(raw: Path, name):
    f = raw / f"{name}.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        # chunked runs (run_staged.sh): <name>_<chunk>.csv with a chunk tag starting with the name
        parts = sorted(raw.glob(f"{name}_{name}*.csv"))
        if not parts:
            return None
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if "error" in df and df["error"].notna().any():
        print(f"WARNING {name}: {df['error'].notna().sum()} rows with errors")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    R = Path(a.results); raw = R / "raw"; out = R / "tables"; out.mkdir(parents=True, exist_ok=True)
    summary = {}

    # ---------------- main
    for reg in ["R1", "R2"]:
        df = load(raw, f"main_{reg}")
        if df is None:
            continue
        df = df[df.status == 0] if "status" in df else df
        n_all = df.date.nunique()
        df = common_days(df)
        summary[f"main_{reg}_days_dropped_not_common"] = int(n_all - df.date.nunique())
        g = df.groupby("policy").agg(days=("date", "nunique"), obj=("obj", "mean"), flown=("flown", "mean"),
                                     planned_flown=("planned_flown", "mean"), ondemand=("ondemand_flown", "mean"),
                                     preapproved=("preapproved", "mean"), cancelled=("cancelled", "mean"),
                                     switches=("switches", "mean"), max_ground=("max_ground_ratio", "max"),
                                     max_air=("max_air_ratio", "max"), F_day=("exp_fatalities_day", "mean"),
                                     fs_runtime=("fs_runtime_s", "mean"), fs_runtime_max=("fs_runtime_s", "max"),
                                     residual_max=("residual", "max"), fs_residual_max=("fs_residual", "max"))
        ws = g.loc["WaitAndSee", "obj"] if "WaitAndSee" in g.index else np.nan
        g["obj_pct_of_waitandsee"] = 100 * g.obj / ws
        g.to_csv(out / f"main_{reg}_by_policy.csv")
        comps = []
        for pol in ["RF", "kNN", "SAA", "Deterministic", "Robust90", "Static", "WaitAndSee"]:
            for ref in ["SeasonalSAA"]:
                for met in ["obj", "flown", "cancelled"]:
                    m, lo, hi, n = paired(df, pol, ref, met)
                    _, lo1, hi1, _ = paired(df, pol, ref, met, block=1)
                    comps.append({"policy": pol, "reference": ref, "metric": met, "mean_diff": m, "ci_lo": lo, "ci_hi": hi,
                                  "ci_lo_iid": lo1, "ci_hi_iid": hi1, "n_days": n})
        for met in ["obj", "flown", "cancelled"]:
            m, lo, hi, n = paired(df, "RF", "Deterministic", met)
            comps.append({"policy": "RF", "reference": "Deterministic", "metric": met, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "n_days": n})
            m, lo, hi, n = paired(df, "WaitAndSee", "RF", met)
            comps.append({"policy": "WaitAndSee", "reference": "RF", "metric": met, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "n_days": n})
            m, lo, hi, n = paired(df, "RF", "SAA", met)
            comps.append({"policy": "RF", "reference": "SAA", "metric": met, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "n_days": n})
            m, lo, hi, n = paired(df, "RF", "Static", met)
            comps.append({"policy": "RF", "reference": "Static", "metric": met, "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "n_days": n})
        pd.DataFrame(comps).to_csv(out / f"main_{reg}_paired.csv", index=False)
        # windy vs calm subgroups (days with any realised operating hour > 12 m/s)
        if (R / "tables" / "day_wind.csv").exists():
            wd = pd.read_csv(R / "tables" / "day_wind.csv")
            df2 = df.merge(wd[["date", "windy"]], on="date", how="left")
            rows = []
            for grp, sub in df2.groupby("windy"):
                for pol in ["RF", "kNN", "SAA", "Deterministic", "Robust90", "Static"]:
                    # subgroup days are not consecutive, so blocks are meaningless: independent-days bootstrap
                    m, lo, hi, n = paired(sub, pol, "SeasonalSAA", "obj", block=1)
                    rows.append({"windy_day": bool(grp), "policy": pol, "mean_diff_obj_vs_SeasonalSAA": m, "ci_lo": lo, "ci_hi": hi, "n_days": n})
            pd.DataFrame(rows).to_csv(out / f"main_{reg}_windy_split.csv", index=False)
        summary[f"main_{reg}"] = {"policies": g.reset_index().to_dict("records")}

    # ---------------- strict day-ahead context (forecast predicted 48 h before valid time)
    rows = []
    for reg in ["R1", "R2"]:
        d2 = load(raw, f"main_{reg}_lead2")
        if d2 is None:
            continue
        d2 = common_days(d2[d2.status == 0])
        for pol in ["RF", "kNN", "Deterministic", "Robust90"]:
            m, lo, hi, n = paired(d2, pol, "SeasonalSAA", "obj")
            rows.append({"regime": reg, "context": "48 h", "policy": pol, "mean_diff_obj_vs_SeasonalSAA": m,
                         "ci_lo": lo, "ci_hi": hi, "n_days": n,
                         "obj_mean": d2[d2.policy == pol].obj.mean(), "flown_mean": d2[d2.policy == pol].flown.mean()})
    if rows:
        pd.DataFrame(rows).to_csv(out / "strict_dayahead.csv", index=False)

    # ---------------- nav regime map
    df = load(raw, "nav")
    if df is not None:
        df = df[df.status == 0]
        rows = []
        for s, sub in df.groupby("setting"):
            nav, m = s.split("|")
            r = {"nav_model": nav, "mitigation": float(m.split("=")[1]), "days": sub.date.nunique()}
            for pol in ["WaitAndSee", "RF", "SeasonalSAA"]:
                q = sub[sub.policy == pol]
                r[f"flown_{pol}"] = q.flown.mean(); r[f"obj_{pol}"] = q.obj.mean()
            md, lo, hi, n = paired(sub, "RF", "SeasonalSAA", "obj")
            r.update(vci_obj=md, vci_lo=lo, vci_hi=hi)
            rows.append(r)
        pd.DataFrame(rows).sort_values(["nav_model", "mitigation"]).to_csv(out / "nav_regime_map.csv", index=False)

    # ---------------- societal frontier
    df = load(raw, "soc")
    if df is not None:
        df = df[df.status == 0]
        g = df.groupby(["setting", "policy"]).agg(days=("date", "nunique"), flown=("flown", "mean"), obj=("obj", "mean"),
                                                  F_day=("exp_fatalities_day", "mean"), F_day_max=("exp_fatalities_day", "max"),
                                                  cancelled=("cancelled", "mean"),
                                                  cert_gap_max=("fs_cert_gap", "max"),
                                                  cert_gap_mean=("fs_cert_gap", "mean"),
                                                  resid_max=("residual", "max")).reset_index()
        g.to_csv(out / "societal_frontier.csv", index=False)
        cg = df["fs_cert_gap"].dropna()
        summary["soc_cert_gap_max"] = float(cg.max()) if len(cg) else None
        summary["soc_cert_gap_median"] = float(cg.median()) if len(cg) else None
        summary["soc_fs_residual_max"] = float(df["fs_residual"].max())
        summary["soc_eval_residual_max"] = float(df["residual"].max())

    # ---------------- dynamic configuration
    df = load(raw, "dyn")
    if df is not None:
        df = df[df.status == 0]
        g = df.groupby(["setting", "policy"]).agg(days=("date", "nunique"), obj=("obj", "mean"), flown=("flown", "mean"),
                                                  switches=("switches", "mean"), cancelled=("cancelled", "mean")).reset_index()
        g.to_csv(out / "dynamic_configuration.csv", index=False)

    # ---------------- sensitivity
    df = load(raw, "sens")
    if df is not None:
        df = df[df.status == 0]
        rows = []
        for s, sub in df.groupby("setting"):
            r = {"setting": s, "days": sub.date.nunique()}
            for pol in ["RF", "SeasonalSAA", "WaitAndSee"]:
                q = sub[sub.policy == pol]
                r[f"obj_{pol}"] = q.obj.mean(); r[f"flown_{pol}"] = q.flown.mean()
                r[f"cancelled_{pol}"] = q.cancelled.mean()
            md, lo, hi, n = paired(sub, "RF", "SeasonalSAA", "obj"); r.update(vci=md, vci_lo=lo, vci_hi=hi)
            md, lo, hi, n = paired(sub, "WaitAndSee", "RF", "obj"); r.update(evpi=md, evpi_lo=lo, evpi_hi=hi)
            r["max_ground"] = sub.max_ground_ratio.max(); r["max_air"] = sub.max_air_ratio.max()
            rows.append(r)
        pd.DataFrame(rows).to_csv(out / "sensitivity.csv", index=False)

    # ---------------- robustness
    df = load(raw, "rob")
    if df is not None:
        df = df[df.status == 0]
        g = df.groupby(["setting", "policy"]).agg(days=("date", "nunique"), obj=("obj", "mean"), flown=("flown", "mean"),
                                                  exceed_h_ground=("exceed_hours_ground", "sum"),
                                                  exceed_h_air=("exceed_hours_air", "sum"),
                                                  days_with_exceed_g=("exceed_hours_ground", lambda s: int((s > 0).sum())),
                                                  days_with_exceed_a=("exceed_hours_air", lambda s: int((s > 0).sum())),
                                                  max_ground=("max_ground_ratio", "max"), max_air=("max_air_ratio", "max"),
                                                  fs_runtime=("fs_runtime_s", "mean")).reset_index()
        g.to_csv(out / "robustness.csv", index=False)

    # ---------------- computation
    names = ["main_R1", "main_R2", "nav", "soc", "dyn", "sens", "rob", "comp", "main_R1_lead2", "main_R2_lead2"]
    frames = [load(raw, n) for n in names]
    for n, f in zip(names, frames):
        if f is not None:
            f["_src"] = n
    allr = pd.concat([f for f in frames if f is not None], ignore_index=True)
    fs = allr.dropna(subset=["fs_runtime_s"])
    fs = fs[fs.fs_runtime_s > 0]
    # the static plan is solved once per run and its record copied to every day: count it once
    st_ = fs[fs.policy == "Static"].drop_duplicates(subset=["_src", "setting"])
    fs = pd.concat([fs[fs.policy != "Static"], st_], ignore_index=True)
    # first-stage problems solved by the structured algorithm (all but the societal frontier
    # and the MILP / extensive-form runs of the computational experiment)
    is_fast = ~fs.experiment.isin(["soc"]) & ~(fs.experiment.eq("comp") & ~fs.setting.astype(str).str.startswith("fast"))
    fs_all, fs = fs, fs[is_fast]
    comp = {"first_stage_solves": int(len(fs)), "status_counts": fs.fs_status.value_counts().to_dict(),
            "runtime_mean_s": float(fs.fs_runtime_s.mean()), "runtime_median_s": float(fs.fs_runtime_s.median()),
            "runtime_p95_s": float(fs.fs_runtime_s.quantile(.95)), "runtime_max_s": float(fs.fs_runtime_s.max()),
            "runtime_max_nonstatic_s": float(fs[fs.policy != "Static"].fs_runtime_s.max()),
            "runtime_max_static_s": float(fs[fs.policy == "Static"].fs_runtime_s.max()) if (fs.policy == "Static").any() else None,
            "gap_max": float(fs.fs_gap.max()), "fs_residual_max": float(fs.fs_residual.max()),
            "eval_residual_max": float(allr.residual.max()), "eval_status_counts": allr.status.value_counts().to_dict(),
            "max_ground_ratio_all_realised_eval": float(allr[allr.experiment != "rob"].max_ground_ratio.max()),
            "max_air_ratio_all_realised_eval": float(allr[allr.experiment != "rob"].max_air_ratio.max()),
            "vars_max": int(fs.n_vars.max()), "cons_max": int(fs.n_cons.max()), "ints_max": int(fs.n_int.max())}
    mains = [f for f in frames[:2] if f is not None]
    if mains and "milp_check_absdiff" in pd.concat(mains):
        mm = pd.concat(mains).dropna(subset=["milp_check_absdiff"])
        comp["milp_checks_main"] = int(len(mm)); comp["milp_checks_max_absdiff"] = float(mm.milp_check_absdiff.max())
        comp["milp_checks_runtime_mean_s"] = float(mm.milp_check_runtime_s.mean())
    c = load(raw, "comp")
    if c is not None:
        c = c[c.status == 0].copy()
        c["method"] = c.setting.str.split("|").str[0]; c["n_scen"] = c.setting.str.split("=").str[1].astype(int)
        g = c.groupby(["method", "n_scen"]).agg(days=("date", "nunique"), vars=("n_vars", "mean"), cons=("n_cons", "mean"),
                                                runtime=("fs_runtime_s", "mean"), runtime_max=("fs_runtime_s", "max"),
                                                gap=("fs_gap", "max"), obj=("fs_obj", "mean")).reset_index()
        g.to_csv(out / "computation_scaling.csv", index=False)
        # same days, same scenarios: objective agreement between methods
        pv = c.pivot_table(index=["date", "n_scen"], columns="method", values="fs_obj")
        if {"fast", "milp"} <= set(pv.columns):
            pm = pv.dropna(subset=["fast", "milp"])
            comp["fast_vs_milp_max_absdiff"] = float((pm.fast - pm.milp).abs().max()); comp["fast_vs_milp_pairs"] = int(len(pm))
        if {"fast", "extensive"} <= set(pv.columns):
            e = c[c.method == "extensive"]
            pe = pv.dropna(subset=["fast", "extensive"])
            comp["extensive_runs"] = int(len(e))
            comp["extensive_optimal"] = int((e.fs_status == 0).sum())
            comp["extensive_gap_min"] = float(e.fs_gap.min()); comp["extensive_gap_max"] = float(e.fs_gap.max())
            comp["extensive_runtime_max_s"] = float(e.fs_runtime_s.max())
            comp["extensive_shortfall_max"] = float(((pe.fast - pe.extensive) / pe.fast.abs()).max())
            comp["extensive_shortfall_min"] = float(((pe.fast - pe.extensive) / pe.fast.abs()).min())
    summary["computation"] = comp
    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (float, np.floating)) and not np.isfinite(o):
            return None                                  # JSON has no NaN/inf
        return o
    json.dump(_clean(summary), open(out / "summary.json", "w"), indent=1, default=float)
    print(json.dumps(comp, indent=1, default=float))


if __name__ == "__main__":
    main()
