"""DEVELOPMENT INPUTS ONLY - SYNTHETIC.  Mimics the file formats produced by
scripts/acquire_and_process.py so that the pipeline can be tested before the official data
arrive.  Nothing generated here may appear in the manuscript or the results package.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

LAT0, LON0, CELL, R = 37.975, 23.725, 20.0, 6_371_000.0


def _latlon(x, y):
    lat = LAT0 + np.rad2deg(y / R)
    lon = LON0 + np.rad2deg(x / (R * np.cos(np.deg2rad(LAT0))))
    return lat, lon


def make(out: Path, seed=0):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    (out / "DEV_SYNTHETIC_README.txt").write_text("SYNTHETIC development inputs - not data.\n")
    json.dump({"LAT0": LAT0, "LON0": LON0, "CELL_m": CELL, "dev_synthetic": True}, open(out / "grid_meta.json", "w"))
    # area 1100 m x 1200 m starting at (200 m, 300 m) from the reference
    X0, Y0, WX, WY = 200.0, 300.0, 1100.0, 1200.0
    roads_x = X0 + np.array([150, 420, 700, 950]); roads_y = Y0 + np.array([200, 520, 830, 1080])
    dates = ["20181024", "20181029", "20181030", "20181101"]
    slots = [("0800", "0830"), ("0830", "0900"), ("0900", "0930"), ("0930", "1000"), ("1000", "1030"), ("1030", "1100")]
    inv, rows = [], []
    tiles = [(X0 + i * WX / 5, Y0 + j * WY / 2, X0 + (i + 1) * WX / 5 + 30, Y0 + (j + 1) * WY / 2 + 30)
             for i in range(5) for j in range(2)]
    cr = lambda v: int(np.floor(v / CELL))
    for d in dates:
        for (s0, s1) in slots:
            for drone, (x0, y0, x1, y1) in enumerate(tiles, start=1):
                la0, lo0 = _latlon(x0, y0); la1, lo1 = _latlon(x1, y1)
                inv.append({"member": f"dev/{d}_d{drone}_{s0}_{s1}.csv", "date": d, "drone": drone, "start": s0,
                            "end": s1, "n_vehicles": 500, "duration_s": 1800.0, "median_dt_s": 0.04,
                            "lat_min": la0, "lat_max": la1, "lon_min": lo0, "lon_max": lo1})
                lvl = rng.uniform(0.7, 1.3)
                for rx in roads_x:
                    if x0 <= rx <= x1:
                        for yy in np.arange(y0, y1, CELL):
                            for vt, m in [("Car", 1.2), ("Taxi", 0.5), ("Motorcycle", 0.8), ("Bus", 0.05), ("Heavy Vehicle", 0.03)]:
                                rows.append((cr(yy), cr(rx), vt, rng.gamma(2, m / 2) * lvl * 1800, 20.0, d, drone, s0, s1))
                for ry in roads_y:
                    if y0 <= ry <= y1:
                        for xx in np.arange(x0, x1, CELL):
                            for vt, m in [("Car", 1.0), ("Taxi", 0.4), ("Motorcycle", 0.7)]:
                                rows.append((cr(ry), cr(xx), vt, rng.gamma(2, m / 2) * lvl * 1800, 20.0, d, drone, s0, s1))
    pd.DataFrame(inv).to_csv(out / "pneuma_file_inventory.csv", index=False)
    pd.DataFrame(rows, columns=["row", "col", "vtype", "veh_s", "mean_speed_kmh", "date", "drone", "start", "end"]
                 ).to_csv(out / "pneuma_cell_vehicle_seconds_by_drone.csv.gz", index=False, compression="gzip")
    # buildings: blocks between roads
    feats = []
    xs = np.concatenate([[X0], roads_x, [X0 + WX]]); ys = np.concatenate([[Y0], roads_y, [Y0 + WY]])
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            a, b, c, e = xs[i] + 15, xs[i + 1] - 15, ys[j] + 15, ys[j + 1] - 15
            if b - a > 20 and e - c > 20:
                ring = [_latlon(a, c), _latlon(b, c), _latlon(b, e), _latlon(a, e), _latlon(a, c)]
                feats.append({"type": "Feature", "properties": {"building": "yes"},
                              "geometry": {"type": "Polygon", "coordinates": [[[lo, la] for la, lo in ring]]}})
    json.dump({"type": "FeatureCollection", "features": feats}, open(out / "osm_buildings.geojson", "w"))
    # wind: hourly 2024-01-01..2026-10-31, AR(1) on log-speed with seasonal cycle
    t = pd.date_range("2024-01-01", "2026-10-31 23:00", freq="h")
    n = len(t); z = np.zeros(n)
    for i in range(1, n):
        z[i] = 0.97 * z[i - 1] + rng.normal(0, 0.25)
    seas = 0.25 * np.cos(2 * np.pi * (t.dayofyear - 200) / 365.25)
    spd = np.exp(np.log(6.5) + seas + z * 0.8) * (1 + 0.15 * np.sin(2 * np.pi * (t.hour - 15) / 24))
    dirn = (20 + 60 * np.tanh(z) + rng.normal(0, 25, n)) % 360
    fc = np.maximum(spd * np.exp(rng.normal(0, 0.25, n)), 0.1)
    fcd = (dirn + rng.normal(0, 25, n)) % 360
    hdr = "latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n37.98,23.73,100,7200,Europe/Athens,EET\n\n"
    tt = t.strftime("%Y-%m-%dT%H:%M")
    def write(name, cols):
        df = pd.DataFrame({"time": tt, **cols})
        (out / name).write_text(hdr + df.to_csv(index=False))
    write("wind_era5_reanalysis.csv", {"wind_speed_10m (m/s)": spd * 0.7, "wind_speed_100m (m/s)": spd,
                                       "wind_direction_100m (°)": dirn, "wind_gusts_10m (m/s)": spd * 1.3})
    sr = np.maximum(spd * np.exp(rng.normal(0, 0.1, n)), 0.1)
    fc2 = np.maximum(spd * np.exp(rng.normal(0, 0.32, n)), 0.1)      # longer lead: noisier (synthetic)
    fcd2 = (dirn + rng.normal(0, 32, n)) % 360
    write("wind_dayahead_previous_runs_ecmwf_ifs025.csv", {"wind_speed_10m (m/s)": spd * 0.7, "wind_speed_100m (m/s)": spd,
          "wind_direction_100m (°)": dirn, "wind_speed_10m_previous_day1 (m/s)": fc * 0.7,
          "wind_speed_100m_previous_day1 (m/s)": fc, "wind_direction_100m_previous_day1 (°)": fcd,
          "wind_speed_100m_previous_day2 (m/s)": fc2, "wind_direction_100m_previous_day2 (°)": fcd2})
    write("wind_historical_forecast_ecmwf_ifs025.csv", {"wind_speed_10m (m/s)": sr * 0.7, "wind_speed_100m (m/s)": sr,
          "wind_direction_100m (°)": (dirn + rng.normal(0, 10, n)) % 360, "wind_gusts_10m (m/s)": sr * 1.3})
    return out
