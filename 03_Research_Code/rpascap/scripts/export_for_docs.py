"""Collect configuration, data summaries and result tables into one JSON consumed by the
Word-document builders (so that every number in the documents comes from saved outputs)."""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpascap.geometry import DEFAULT_MENU  # noqa

TAGS = {"value": ("flight-equivalent", "ASSUMED", "normalisation"), "cancel_penalty": ("flight-equivalents", "ASSUMED", "sensitivity 1-4"),
        "switch_cost": ("flight-equivalents", "ASSUMED", "sensitivity 0-20"), "max_switches": ("per corridor-day", "ASSUMED", "sensitivity 0-3"),
        "theta_G": ("fatalities per aircraft FH", "ASSUMED", "research criterion (Kim & Bae 2022); not regulatory"),
        "theta_A": ("collision involvements per aircraft FH", "ASSUMED", "= lambda_f; sensitivity 1e-6..1e-4"),
        "lambda_f": ("per FH", "ASSUMED", "sensitivity 1e-6..1e-4"), "mitigation": ("-", "ASSUMED", "0 (R1) or 0.9 (R2)"),
        "ped_density": ("persons/m2 open ground", "ASSUMED", "no observation exists; sensitivity 0.005-0.1"),
        "offpeak_scale": ("-", "ASSUMED", "road exposure outside 08-11 relative to observed"),
        "demand_scale": ("-", "SYNTHETIC", "multiplies synthetic demand"), "nav_model": ("-", "ASSUMED", "see Table 6"),
        "mip_gap": ("-", "setting", "HiGHS relative gap")}


def j(df):
    return json.loads(df.to_json(orient="records")) if df is not None else []


def rd(p):
    return pd.read_csv(p) if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); ap.add_argument("--config", default="configs/study.yaml")
    ap.add_argument("--out", default="results/doc_data.json")
    a = ap.parse_args()
    R = Path(a.results); T = R / "tables"; cfg = yaml.safe_load(open(a.config))
    s = json.load(open(T / "data_summary.json"))
    lodo = rd(T / "exposure_leave_one_date_out.csv")
    lodo_agg = []
    if lodo is not None and len(lodo):
        for k, g in lodo.groupby("class"):
            lodo_agg.append({"class": k, "corr_mean": g["corr"].mean(), "corr_min": g["corr"].min(), "corr_max": g["corr"].max(),
                             "rel_err_mean": g.rel_error_total.mean(), "n": len(g)})
    dw = rd(T / "day_wind.csv")
    W = pd.read_csv(R / "processed" / "wind_hourly_operating.csv.gz").merge(dw[["date", "set"]], on="date")
    params = []
    for k, v in cfg["base"].items():
        u, tag, note = TAGS.get(k, ("", "", ""))
        params.append([k, str(v), u, tag, note])
    for grp, tag in [("platform", "VERIFIED/ASSUMED"), ("kinematics", "VERIFIED/ASSUMED"), ("harm", "LITERATURE"), ("ps", "ASSUMED"),
                     ("occupancy", "ASSUMED"), ("corridors", "DESIGN"), ("demand", "SYNTHETIC")]:
        for k, v in cfg[grp].items():
            params.append([f"{grp}.{k}", json.dumps(v) if isinstance(v, dict) else str(v), "", tag, ""])
    ho = rd(T / "exposure_holdout_validation.csv")
    out = {"cfg": cfg, "summary": s, "lodo": lodo_agg, "holdout": j(ho) if ho is not None else [],
           "menu": [{"name": m.name, "n_lat": m.n_lat, "n_vert": m.n_vert, "n_lanes": m.n_lanes, "S_y": m.S_y, "S_z": m.S_z,
                     "S_x": m.S_x, "fixed_cost": m.fixed_cost} for m in DEFAULT_MENU],
           "corridors": j(rd(T / "corridors.csv")), "nav": j(rd(T / "nav_models.csv")), "params": params,
           "sens_ped": [0.005, 0.02, 0.05, 0.1],
           "frac_above12_test": float((W[W.set == "test"].W_realised_100m > 12).mean()),
           "frac_above12_train": float((W[W.set == "train"].W_realised_100m > 12).mean()),
           "windy_days_test": int(dw[(dw.set == "test")].windy.sum()), "n_test_days": int((dw.set == "test").sum()),
           "tables": {}}
    for f in sorted(T.glob("*.csv")):
        if f.stat().st_size < 3_000_000:
            out["tables"][f.stem] = j(pd.read_csv(f))
    for f in ["summary.json"]:
        if (T / f).exists():
            out["analysis"] = json.load(open(T / f))
    tl = R / "test_log.txt"
    if tl.exists():
        last = [l for l in tl.read_text().splitlines() if l.strip()]
        out["test_summary"] = last[-1].strip("= ") if last else None
    if (R / "raw" / "run_log.json").exists():
        out["run_log"] = json.load(open(R / "raw" / "run_log.json"))
    json.dump(out, open(a.out, "w"), default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
