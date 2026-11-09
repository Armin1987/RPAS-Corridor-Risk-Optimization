"""Build Supplementary_Research_Dataset.xlsx from saved inputs/outputs only.
Usage: python -m scripts.make_workbook --processed <dir> --results results --out Supplementary_Research_Dataset.xlsx
"""
from __future__ import annotations
import argparse, json, sys
import pandas as pd
import yaml
from pathlib import Path
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rpascap import navmodels as nm  # noqa

SOURCES = [
 {"source": "pNEUMA dataset v1", "provider": "EPFL LUTS (Barmpounakis, Geroliminis)", "identifier": "https://doi.org/10.5281/zenodo.10491409",
  "licence": "CC BY 4.0 (Zenodo record); site footer states CC BY-NC 4.0 - conflict noted", "file": "pNEUMA_dataset.zip, 15.8 GB, MD5 4e842b2200bea2f80d41c6b785a9de25",
  "use_in_study": "road-vehicle positions -> road-occupant exposure", "type": "OBSERVED",
  "citation": "Barmpounakis E, Geroliminis N (2020) Transp Res Part C 111:50-71. https://doi.org/10.1016/j.trc.2019.11.023; acknowledgement 'Data source: pNEUMA - open-traffic.epfl.ch'"},
 {"source": "ERA5 reanalysis via Open-Meteo Historical Weather API", "provider": "ECMWF / Copernicus; served by Open-Meteo", "identifier": "https://archive-api.open-meteo.com/v1/archive",
  "licence": "see Open-Meteo licence page and Copernicus licence (to be confirmed before publication)", "file": "wind_era5_reanalysis.csv",
  "use_in_study": "realised hourly wind at 100 m (outcome)", "type": "REANALYSIS (model + observations)", "citation": "Hersbach et al. (2020) ERA5 (citation to be added at submission)"},
 {"source": "ECMWF IFS 0.25 day-ahead forecasts via Open-Meteo Previous Runs API", "provider": "ECMWF; served by Open-Meteo", "identifier": "https://previous-runs-api.open-meteo.com/v1/forecast",
  "licence": "see Open-Meteo licence page", "file": "wind_dayahead_previous_runs_ecmwf_ifs025.csv",
  "use_in_study": "context: forecast valid hour issued 24 h earlier (_previous_day1)", "type": "FORECAST", "citation": "Open-Meteo documentation"},
 {"source": "ECMWF IFS 0.25 stitched short-range forecasts via Open-Meteo Historical Forecast API", "provider": "ECMWF; Open-Meteo", "identifier": "https://historical-forecast-api.open-meteo.com/v1/forecast",
  "licence": "see Open-Meteo licence page", "file": "wind_historical_forecast_ecmwf_ifs025.csv",
  "use_in_study": "nowcast proxy in the robustness test", "type": "FORECAST", "citation": "Open-Meteo documentation"},
 {"source": "OpenStreetMap buildings", "provider": "OpenStreetMap contributors via osmnx/Overpass", "identifier": "https://www.openstreetmap.org",
  "licence": "ODbL 1.0", "file": "osm_buildings.geojson", "use_in_study": "open-ground mask for the pedestrian layer", "type": "OBSERVED (crowd-sourced)",
  "citation": "(c) OpenStreetMap contributors"},
 {"source": "DJI Matrice 300 RTK specifications", "provider": "DJI", "identifier": "https://www.dji.com/au/support/product/matrice-300",
  "licence": "manufacturer data", "file": "-", "use_in_study": "mass range, speed, descent rate, wind limit, size", "type": "VERIFIED SPECIFICATION", "citation": "DJI (retrieved 2026-11-07)"},
 {"source": "Geister et al. (2018) ICNS", "provider": "DLR", "identifier": "https://elib.dlr.de/119407/",
  "licence": "publication", "file": "-", "use_in_study": "measured lateral TSE RMS anchoring navigation core scales", "type": "LITERATURE (measured)", "citation": "Geister R, Limmer L, Rippl M, Dautermann T (2018) Total system error performance of drones for an unmanned PBN concept. ICNS 2018"},
 {"source": "Drone-delivery demand", "provider": "this study", "identifier": "rpascap/instances.py", "licence": "-", "file": "-",
  "use_in_study": "planned and on-demand requests", "type": "SYNTHETIC", "citation": "-"},
]

DICTIONARY = [
 ("row, col", "-", "grid cell index; cell covers y in [row*20, row*20+20) m, x in [col*20, col*20+20) m from reference point"),
 ("lat0, lon0", "deg", "grid reference point (WGS84) of the local equirectangular frame"),
 ("veh_s", "vehicle-seconds", "time-weighted presence of vehicles in a cell during one 30-min file (gaps > 1 s not counted)"),
 ("mean_veh", "vehicles", "veh_s / file duration; drones combined by per-cell maximum"),
 ("road_*_p_m2", "persons/m2", "occupants per m2 by class (moto / vehicle / heavy) using ASSUMED occupancy, 08:00-11:00 mean"),
 ("building_frac", "-", "share of cell covered by OSM building footprints"),
 ("W_realised_100m", "m/s", "ERA5 wind speed at 100 m, local time (Europe/Athens)"),
 ("W_dayahead_100m", "m/s", "ECMWF IFS forecast for that hour issued 24 h earlier"),
 ("dir_*_from", "deg", "meteorological direction the wind blows FROM"),
 ("C_*", "fatalities per crash (or m2 per unit density for C_ped)", "expected third-party fatalities per crash initiated on the lane, by exposure class"),
 ("kappa", "per flight hour per (flight/h)", "collision-involvement coefficient of a lane pair"),
 ("flown", "flights/day", "pre-approved plus on-demand flights flown on the evaluated test day"),
 ("cancelled", "flights/day", "pre-approved flights not flown"),
 ("obj", "flight-equivalents/day", "value of flights minus cancellation penalties, switching and configuration costs"),
 ("max_ground_ratio", "-", "max over lanes and hours of (lambda_f + r) C / theta_G, recomputed outside the solver"),
 ("max_air_ratio", "-", "max over lanes and hours of r / theta_A, recomputed outside the solver"),
 ("exp_fatalities_day", "fatalities/day", "exact expected third-party fatalities (including bilinear collision term)"),
 ("residual", "-", "max violation of any (scaled) constraint or bound in the solved LP/MILP"),
]


def style(ws, df):
    hdr = PatternFill("solid", fgColor="E7ECF3")
    for c in ws[1]:
        c.font = Font(bold=True); c.fill = hdr; c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, col in enumerate(df.columns, start=1):
        w = max([len(str(col))] + [len(str(v)) for v in df[col].head(200).tolist()]) + 2
        ws.column_dimensions[get_column_letter(i)].width = min(max(w, 8), 60)
    ws.freeze_panes = "A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True); ap.add_argument("--results", default="results")
    ap.add_argument("--config", default="configs/study.yaml"); ap.add_argument("--out", default="Supplementary_Research_Dataset.xlsx")
    a = ap.parse_args()
    P, R = Path(a.processed), Path(a.results); T, PR, RAW = R / "tables", R / "processed", R / "raw"
    cfg = yaml.safe_load(open(a.config)); info = json.load(open(T / "data_summary.json"))
    sheets = {}
    readme = [("Title", "Supplementary research dataset: data-driven corridor configuration and flight admission for urban drone delivery (Athens)"),
              ("Status", "DEVELOPMENT INPUTS - NOT RESULTS" if info.get("dev_synthetic") else "Processed official data and executed results"),
              ("Generated by", "scripts/make_workbook.py from results/ and the processed inputs; no value is typed by hand"),
              ("Grid", f"20 m cells, reference lat0={info['lat0']}, lon0={info['lon0']}, local equirectangular frame"),
              ("Periods", f"train {info.get('train_first_day', cfg['train'][0])}..{info.get('train_last_day', cfg['train'][1])} "
                          f"({info.get('train_days')} complete days; configured window {cfg['train'][0]}..{cfg['train'][1]}), "
                          f"test {info.get('test_first_day', cfg['test'][0])}..{info.get('test_last_day', cfg['test'][1])} ({info.get('test_days')} days)"),
              ("Type tags", "OBSERVED / REANALYSIS / FORECAST / VERIFIED / LITERATURE / ASSUMED / SYNTHETIC")]
    if (P / "MANIFEST.json").exists():
        for k, v in json.load(open(P / "MANIFEST.json")).items():
            readme.append((f"input file {k}", f"sha256 {v['sha256']}, {v['bytes']} bytes"))
    if (RAW / "run_log.json").exists():
        for k, v in json.load(open(RAW / "run_log.json")).items():
            readme.append((f"run_log {k}", str(v)))
    sheets["00_README_Reproducibility"] = pd.DataFrame(readme, columns=["item", "value"])
    sheets["01_Sources_Provenance"] = pd.DataFrame(SOURCES)
    inv = pd.read_csv(P / "pneuma_file_inventory.csv")
    sheets["02_pNEUMA_Inventory"] = inv
    if (P / "acquisition_log.jsonl").exists():
        sheets["02b_Acquisition_Log"] = pd.read_json(P / "acquisition_log.jsonl", lines=True).astype(str)
    sheets["03_Road_Exposure_Cells"] = pd.read_csv(PR / "exposure_cells.csv")
    sheets["03b_Exposure_Validation"] = pd.read_csv(T / "exposure_leave_one_date_out.csv")
    sheets["03c_Vehicles_By_Type"] = pd.read_csv(T / "pneuma_vehicles_by_type.csv")
    sheets["04_Corridors"] = pd.read_csv(T / "corridors.csv")
    sheets["04b_Lanes"] = pd.read_csv(T / "corridor_lanes.csv")
    sheets["05_Weather_Daily"] = pd.read_csv(T / "day_wind.csv")
    sheets["05b_Weather_Hourly"] = pd.read_csv(PR / "wind_hourly_operating.csv.gz")
    sheets["05c_Context_Features"] = pd.read_csv(PR / "context_features.csv")
    sheets["05d_Weather_QC"] = pd.read_csv(T / "weather_qc.csv")
    prm = []
    for k, v in cfg["base"].items():
        prm.append({"group": "base", "parameter": k, "value": v})
    for grp in ["platform", "kinematics", "harm", "ps", "occupancy", "corridors", "demand"]:
        for k, v in cfg[grp].items():
            prm.append({"group": grp, "parameter": k, "value": json.dumps(v) if isinstance(v, dict) else v})
    sheets["06_Model_Parameters"] = pd.DataFrame(prm)
    nav = pd.read_csv(T / "nav_models.csv")
    nav["sensitivity_role"] = ["R1 reference (light tail)", "R2 reference (with 99 % mitigation)", "sensitivity / misspecification", "sensitivity / misspecification"]
    sheets["07_NavError_Models"] = nav
    sheets["08_Collision_Coefficients"] = pd.read_csv(T / "collision_coefficients.csv")
    cl = pd.read_csv(T / "consequence_table_long.csv.gz")
    sheets["08b_Ground_Consequence"] = cl
    sheets["08c_Descent_Kernels"] = pd.read_csv(T / "descent_kernel_stats.csv")
    frames = {f.stem: pd.read_csv(f) for f in sorted(RAW.glob("*.csv"))}
    if frames:
        allr = pd.concat(frames.values(), ignore_index=True)
        sheets["09_Scenarios"] = allr[allr.experiment == "main"][["setting", "policy", "date", "scenarios"]]
        bench = pd.DataFrame([
            {"policy": "Static", "first_stage_information": "none", "scenarios": "2 x n training days, one plan for all days, no switches"},
            {"policy": "Deterministic", "first_stage_information": "day-ahead forecast", "scenarios": "single: forecast trajectory, mean on-demand"},
            {"policy": "SAA", "first_stage_information": "none", "scenarios": f"{cfg['n_scenarios']} random training days, equal weights"},
            {"policy": "SeasonalSAA", "first_stage_information": "calendar", "scenarios": f"{cfg['n_scenarios']} training days within +/-{cfg['seasonal_window']} days of year"},
            {"policy": "kNN", "first_stage_information": "forecast + calendar", "scenarios": f"{cfg['n_scenarios']} nearest training days in standardised context"},
            {"policy": "RF", "first_stage_information": "forecast + calendar", "scenarios": f"top {cfg['n_scenarios']} random-forest weights (200 trees, leaf >= 10)"},
            {"policy": "Robust90", "first_stage_information": "forecast + calendar", "scenarios": "single: hourly 90th percentile of kNN neighbours"},
            {"policy": "WaitAndSee", "first_stage_information": "realised day (bound)", "scenarios": "single: realised day"}])
        bench["solver"] = "SciPy milp -> HiGHS; mip_rel_gap 1e-4; time limit 300 s"
        sheets["10_Benchmarks_Settings"] = bench
        cols = ["experiment", "setting", "policy", "date", "plan", "switches", "q_total", "obj", "flown", "planned_flown",
                "ondemand_flown", "cancelled", "exp_fatalities_day", "fs_obj", "fs_runtime_s", "fs_status", "fs_gap", "n_vars", "n_cons", "n_int"]
        main = allr[allr.experiment == "main"]
        sheets["11_Results_Main_PerDay"] = main[[c for c in cols if c in main]]
        for f in ["main_R1_by_policy", "main_R2_by_policy", "main_R1_paired", "main_R2_paired", "main_R1_windy_split", "main_R2_windy_split"]:
            if (T / f"{f}.csv").exists():
                sheets[f"11_{f}"[:31]] = pd.read_csv(T / f"{f}.csv")
        sv = ["experiment", "setting", "policy", "date", "status", "fs_status", "fs_residual", "residual", "max_ground_ratio",
              "max_air_ratio", "exceed_hours_ground", "exceed_hours_air"]
        sheets["12_Safety_Verification"] = allr[[c for c in sv if c in allr]]
        for f, nm_ in [("nav_regime_map", "13_Nav_Regime_Map"), ("societal_frontier", "13b_Societal_Frontier"),
                       ("dynamic_configuration", "13c_Dynamic_Config"), ("sensitivity", "13d_Sensitivity"),
                       ("robustness", "13e_Robustness"), ("computation_scaling", "13f_Computation")]:
            if (T / f"{f}.csv").exists():
                sheets[nm_] = pd.read_csv(T / f"{f}.csv")
    sheets["14_Data_Dictionary"] = pd.DataFrame(DICTIONARY, columns=["variable", "unit", "definition"])
    with pd.ExcelWriter(a.out, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            style(xw.sheets[name[:31]], df)
    print(a.out, len(sheets), "sheets")


if __name__ == "__main__":
    main()
