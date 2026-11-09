"""Descriptive tables of the processed inputs and model coefficients -> results/tables/.
Also writes processed model inputs (exposure grids, daily wind matrices) to results/processed/.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpascap.study import Study, CLASSES  # noqa
from rpascap.geometry import lanes        # noqa
from rpascap import navmodels as nm       # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--config", default="configs/study.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    R = Path(a.out); T = R / "tables"; PR = R / "processed"; T.mkdir(parents=True, exist_ok=True); PR.mkdir(exist_ok=True)
    st = Study(Path(a.processed), cfg, R / "cache")
    st.build_tables(cfg["base"]["nav_model"], cfg["gamma"], cfg["n_mc_kernel"])
    meta = json.load(open(Path(a.processed) / "grid_meta.json"))
    info = {"dev_synthetic": bool(meta.get("dev_synthetic", False)), "grid_rows": st.nr, "grid_cols": st.nc,
            "cell_m": st.cell, "r0": st.r0, "c0": st.c0, "lat0": st.lat0, "lon0": st.lon0,
            "date_slots": st.n_date_slots, "osm_buildings": st.has_osm,
            "covered_area_km2": float(st.coverage.sum() * st.cell ** 2 / 1e6),
            "building_share_in_coverage": float(st.building_frac[st.coverage].mean()),
            "weather_days_complete": int(len(st.W)),
            "train_first_day": str(st.train.min().date()), "train_last_day": str(st.train.max().date()),
            "test_first_day": str(st.test.min().date()), "test_last_day": str(st.test.max().date()),
            "grid_point": {"lat": st.meta_real.get("latitude") if isinstance(st.meta_real, dict) else None,
                           "lon": st.meta_real.get("longitude") if isinstance(st.meta_real, dict) else None}, "train_days": int(len(st.train)), "test_days": int(len(st.test)),
            "weather_days_incomplete": int((~st.wqc.complete).sum()),
            "skill_train": st.skill_train, "skill_test": st.skill_test,
            "realised_meta": st.meta_real, "forecast_meta": st.meta_fc}
    # pNEUMA inventory summary
    inv = st.inventory
    info["pneuma_files"] = int(len(inv)); info["pneuma_dates"] = sorted(inv.date.dropna().astype(str).unique().tolist())
    if "duration_s" in inv:
        info["pneuma_duration_min_s"] = float(inv.duration_s.min()); info["pneuma_duration_max_s"] = float(inv.duration_s.max())
    info["pneuma_vehicles_total"] = int(inv.n_vehicles.sum()) if "n_vehicles" in inv else None
    vt = [c for c in inv.columns if c.startswith("n_") and c not in ("n_vehicles", "n_points", "n_cells")]
    inv_sum = inv.groupby("date")[["n_vehicles"] + vt].sum() if vt else inv.groupby("date")[["n_vehicles"]].sum()
    inv_sum.to_csv(T / "pneuma_inventory_by_date.csv")
    mv = st.mean_veh
    if st.exposure_sets:
        devk = {k for k, v in st.exposure_sets.items() if v == "dev"}
        mv = mv[[(a, b) in devk for a, b in zip(mv.date, mv.start)]]
        info["exposure_fit"] = "dev date-slots only (fit and freeze); testA/testB held out"
        info["pneuma_sets"] = inv.groupby("set").agg(files=("member", "count"), dates=("date", lambda x: sorted(set(map(str, x)))),
                                                     slots=("start", lambda x: sorted(set(map(str, x))))).reset_index().to_dict("records")
    by_type = mv.groupby("vtype").mean_veh.sum() / st.n_date_slots
    by_type.rename("mean_vehicles_present_in_study_area").to_csv(T / "pneuma_vehicles_by_type.csv")
    st.lodo.to_csv(T / "exposure_leave_one_date_out.csv", index=False)
    if len(st.holdout):
        st.holdout.to_csv(T / "exposure_holdout_validation.csv", index=False)
    # exposure grids
    np.savez_compressed(PR / "exposure_grids.npz", coverage=st.coverage, building_frac=st.building_frac,
                        ped_unit=st.ped_unit, **{f"road_{k}": v for k, v in st.rho_road.items()})
    cells = []
    for r in range(st.nr):
        for c in range(st.nc):
            if st.coverage[r, c]:
                cells.append({"row": r + st.r0, "col": c + st.c0,
                              "y_m": (r + st.r0 + 0.5) * st.cell, "x_m": (c + st.c0 + 0.5) * st.cell,
                              "lat": st.lat0 + np.rad2deg((r + st.r0 + 0.5) * st.cell / 6_371_000.0),
                              "lon": st.lon0 + np.rad2deg((c + st.c0 + 0.5) * st.cell / (6_371_000.0 * np.cos(np.deg2rad(st.lat0)))),
                              "building_frac": st.building_frac[r, c],
                              "road_moto_p_m2": st.rho_road["moto"][r, c], "road_vehicle_p_m2": st.rho_road["vehicle"][r, c],
                              "road_heavy_p_m2": st.rho_road["heavy"][r, c]})
    pd.DataFrame(cells).to_csv(PR / "exposure_cells.csv", index=False)
    # weather
    st.wqc.to_csv(T / "weather_qc.csv", index=False)
    windy = (st.W > 12).any(axis=1)
    dw = pd.DataFrame({"date": [str(d.date()) for d in st.W.index], "windy": windy.values,
                       "hours_above_12": (st.W > 12).sum(axis=1).values, "max_W": st.W.max(axis=1).values,
                       "mean_W": st.W.mean(axis=1).values, "mean_F": st.F.mean(axis=1).values,
                       "set": ["train" if d in st.train else ("test" if d in st.test else "other") for d in st.W.index]})
    dw.to_csv(T / "day_wind.csv", index=False)
    long = []
    for d in st.W.index:
        for h in st.hours:
            long.append({"date": str(d.date()), "hour": h, "W_realised_100m": st.W.at[d, h], "dir_realised_from": st.D.at[d, h],
                         "W_dayahead_100m": st.F.at[d, h], "dir_dayahead_from": st.FD.at[d, h],
                         "W_shortrange_100m": (st.SR.at[d, h] if st.SR is not None else np.nan)})
    pd.DataFrame(long).to_csv(PR / "wind_hourly_operating.csv.gz", index=False, compression="gzip")
    st.X.assign(date=[str(d.date()) for d in st.X.index]).to_csv(PR / "context_features.csv", index=False)
    mon = pd.DataFrame({"month": st.W.index.month, "frac_hours_above_12": (st.W > 12).mean(axis=1).values,
                        "set": dw.set.values}).groupby(["set", "month"]).frac_hours_above_12.mean()
    mon.to_csv(T / "wind_exceedance_by_month.csv")
    # corridors and lanes
    rows = []
    for k, cor in enumerate(st.corridors):
        for c, cfgm in enumerate(st.menu):
            for li, L in enumerate(lanes(cor, cfgm) if cfgm.n_lanes else []):
                Cmean = sum(np.mean(st.Ccls[cl][k][c][:, :, li]) * w for cl, w in
                            zip(CLASSES, [cfg["base"]["ped_density"], 1, 1, 1]))
                rows.append({"corridor": cor.name, "length_m": cor.length, "bearing_deg": cor.bearing,
                             "tau_h": st.tau_h[k], "config": cfgm.name, "lane": li, "offset_m": L["offset"],
                             "layer": L["layer"], "alt_m": L["alt"], "heading_deg": L["heading"],
                             "C_mean_over_wind_bins_base": Cmean,
                             "C_ped_unit_mean": float(np.mean(st.Ccls["ped"][k][c][:, :, li])),
                             "C_moto_mean": float(np.mean(st.Ccls["moto"][k][c][:, :, li])),
                             "C_vehicle_mean": float(np.mean(st.Ccls["vehicle"][k][c][:, :, li])),
                             "C_heavy_mean": float(np.mean(st.Ccls["heavy"][k][c][:, :, li]))})
    pd.DataFrame(rows).to_csv(T / "corridor_lanes.csv", index=False)
    pd.DataFrame([{"corridor": c.name, "x0_m": c.x0, "y0_m": c.y0, "x1_m": c.x1, "y1_m": c.y1, "length_m": c.length,
                   "bearing_deg": c.bearing, "base_alt_m": c.base_alt} for c in st.corridors]).to_csv(T / "corridors.csv", index=False)
    # full consequence table (long) for the workbook
    long = []
    for k, cor in enumerate(st.corridors):
        for c, cfgm in enumerate(st.menu):
            for li in range(cfgm.n_lanes):
                for s in range(13):
                    for di in range(12):
                        long.append({"corridor": cor.name, "config": cfgm.name, "lane": li, "speed_bin_ms": s,
                                     "wind_toward_deg": 30 * di,
                                     **{f"C_{cl}": float(st.Ccls[cl][k][c][s, di, li]) for cl in CLASSES}})
    pd.DataFrame(long).to_csv(T / "consequence_table_long.csv.gz", index=False, compression="gzip")
    pd.DataFrame(st.kernel_stats).to_csv(T / "descent_kernel_stats.csv", index=False)
    # collision coefficients for every nav model, configuration and wind bin (corridor-independent)
    kin = nm.Kinematics(**cfg["kinematics"]); rows = []
    for name, mdl in nm.MODELS.items():
        for cfgm in st.menu[1:]:
            for s in range(13):
                K = nm.kappa_matrix(cfgm, mdl, kin, float(s), cfg["gamma"])
                rows.append({"nav_model": name, "config": cfgm.name, "speed_bin_ms": s, "kappa_intrail": K[0, 0],
                             "kappa_rowsum_max": K.sum(axis=1).max(),
                             "max_uniform_flow_per_lane_thetaA_1e-5": 1e-5 / K.sum(axis=1).max(),
                             "max_uniform_flow_per_lane_thetaA_1e-5_m0.99": 1e-5 / (0.01 * K.sum(axis=1).max())})
    pd.DataFrame(rows).to_csv(T / "collision_coefficients.csv", index=False)
    pd.DataFrame([{"model": n, "a1_ref_m": m.a1_ref, "a2_m": ("= S_y" if m.a2 is None else m.a2), "alpha": m.alpha,
                   "a1z_m": m.a1z, "a2z_m": m.a2z, "alpha_z": m.alpha_z, "basis": m.basis} for n, m in nm.MODELS.items()]
                 ).to_csv(T / "nav_models.csv", index=False)
    json.dump(info, open(T / "data_summary.json", "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in info.items() if k not in ("realised_meta", "forecast_meta")}, indent=1, default=str))


if __name__ == "__main__":
    main()
