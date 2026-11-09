"""Reader and exposure builder for the official pNEUMA trajectory files.

Official source: pNEUMA dataset, Zenodo record 10.5281/zenodo.10491409 (v1, 2024-01-11,
CC BY 4.0, file pNEUMA_dataset.zip, md5 4e842b2200bea2f80d41c6b785a9de25) and
open-traffic.epfl.ch.  Required citation: Barmpounakis & Geroliminis (2020),
Transp. Res. Part C 111:50-71, doi:10.1016/j.trc.2019.11.023, plus the dataset DOI and
"Data source: pNEUMA - open-traffic.epfl.ch".

File layout per the official Downloads page: one row per vehicle; four fixed fields
(track id, vehicle type, traveled distance [m], average speed [km/h]) followed by a
repeating 6-field block per time step (latitude, longitude, speed [km/h], longitudinal
acceleration [m/s^2], lateral acceleration [m/s^2], time [s]).  Sampling is 0.04 s
(25 fps); one drone recorded at 23.97 fps, so time is taken from the time field, never
from the sample index.  The page does not state the delimiter or the order inside the
block, so both are detected and validated (the lat/lon ranges of central Athens)
instead of assumed.

What these data ARE: positions of road vehicles in ~1.3 km^2 of central Athens,
weekday mornings (08:00-11:00 slots), Oct-Nov 2018.
What they are NOT: pedestrian density, residential population, or drone demand.
They are converted here only into a ROAD-OCCUPANT exposure layer, with vehicle
occupancy and in-vehicle sheltering as explicit, separately reported assumptions.
"""
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

ATHENS_LAT = (37.90, 38.05)
ATHENS_LON = (23.65, 23.85)
FIXED = 4
BLOCK = 6
FILE_RE = re.compile(r"(\d{8})_d(\d+)_(\d{4})_(\d{4})")


@dataclass
class Layout:
    delimiter: str
    lat_idx: int = 0
    lon_idx: int = 1
    speed_idx: int = 2
    time_idx: int = 5


def parse_name(name: str) -> dict:
    m = FILE_RE.search(name)
    if not m:
        return {}
    d, drone, t0, t1 = m.groups()
    return {"date": d, "drone": int(drone), "start": t0, "end": t1}


def detect_layout(sample_lines: list[str]) -> Layout:
    """Detect delimiter and lat/lon order from data rows (header excluded)."""
    line = next(l for l in sample_lines if l.strip() and l.strip()[0].isdigit())
    delim = ";" if line.count(";") > line.count(",") else ","
    toks = [t.strip() for t in line.strip().strip(delim).split(delim)]
    blk = np.array([float(t) for t in toks[FIXED:FIXED + BLOCK]])
    a, b = blk[0], blk[1]
    in_lat = lambda v: ATHENS_LAT[0] <= v <= ATHENS_LAT[1]
    in_lon = lambda v: ATHENS_LON[0] <= v <= ATHENS_LON[1]
    if in_lat(a) and in_lon(b):
        return Layout(delim, 0, 1)
    if in_lat(b) and in_lon(a):
        return Layout(delim, 1, 0)
    raise ValueError(f"first block {blk[:2]} is not inside central Athens: layout unknown")


def iter_vehicles(text_or_path, layout: Layout | None = None):
    """Yield (track_id, vtype, arr[n_steps, 6]) for each vehicle row."""
    if isinstance(text_or_path, str) and "\n" in text_or_path:
        fh = io.StringIO(text_or_path)
    else:
        fh = open(text_or_path, "r", encoding="utf-8", errors="replace")
    with fh:
        lines = fh.readlines() if layout is None else None
        if layout is None:
            layout = detect_layout(lines[:50])
            src = lines
        else:
            src = fh
        for line in src:
            s = line.strip().strip(layout.delimiter)
            if not s or not s[0].isdigit():
                continue
            toks = [t.strip() for t in s.split(layout.delimiter)]
            vals = toks[FIXED:]
            nblk = len(vals) // BLOCK
            if nblk == 0:
                continue
            arr = np.asarray(vals[:nblk * BLOCK], float).reshape(nblk, BLOCK)
            yield int(float(toks[0])), toks[1], arr, layout


@dataclass
class Grid:
    lat0: float
    lon0: float
    cell: float = 20.0
    R: float = 6_371_000.0

    def to_xy(self, lat, lon):
        """Local equirectangular projection (m); error < 0.1 m over 2 km at this latitude."""
        x = np.deg2rad(lon - self.lon0) * self.R * np.cos(np.deg2rad(self.lat0))
        y = np.deg2rad(lat - self.lat0) * self.R
        return x, y

    def cell_index(self, lat, lon):
        x, y = self.to_xy(lat, lon)
        return np.floor(y / self.cell).astype(int), np.floor(x / self.cell).astype(int)


@dataclass
class ExposureAssumptions:
    """Vehicle -> people conversion.  ALL values are engineering assumptions to be
    replaced by sourced figures in Stage 4 and varied in sensitivity analysis."""
    occupancy: dict = field(default_factory=lambda: {
        "Car": 1.3, "Taxi": 1.5, "Motorcycle": 1.1, "Bus": 25.0,
        "Medium Vehicle": 1.2, "Heavy Vehicle": 1.1})
    sheltering: dict = field(default_factory=lambda: {
        "Car": 0.7, "Taxi": 0.7, "Motorcycle": 0.0, "Bus": 0.8,
        "Medium Vehicle": 0.8, "Heavy Vehicle": 0.9})


def vehicle_seconds_by_cell(path_or_text, grid: Grid, dt_out: float = 1.0):
    """Accumulate vehicle-seconds per (cell, vehicle type) for one file.

    Each sample is weighted by its own time step (diff of the time field), so the
    23.97-fps drone is handled exactly.  Returns a DataFrame and the file duration (s)."""
    rec = {}
    tmin, tmax = np.inf, -np.inf
    for tid, vtype, arr, lay in iter_vehicles(path_or_text):
        t = arr[:, lay.time_idx]
        if t.size < 2:
            continue
        w = np.diff(t, append=t[-1] + np.median(np.diff(t)))
        r, c = grid.cell_index(arr[:, lay.lat_idx], arr[:, lay.lon_idx])
        key = pd.MultiIndex.from_arrays([r, c])
        s = pd.Series(w, index=key).groupby(level=[0, 1]).sum()
        for (ri, ci), v in s.items():
            rec[(ri, ci, vtype)] = rec.get((ri, ci, vtype), 0.0) + float(v)
        tmin, tmax = min(tmin, t.min()), max(tmax, t.max())
    df = pd.DataFrame([(k[0], k[1], k[2], v) for k, v in rec.items()],
                      columns=["row", "col", "vtype", "veh_s"])
    return df, float(tmax - tmin)


def road_occupant_exposure(df: pd.DataFrame, duration_s: float, grid: Grid,
                           ass: ExposureAssumptions) -> pd.DataFrame:
    """Mean unsheltered-equivalent road occupants per m^2 per cell over the file window:
        rho_road = sum_type (veh_s/duration) * occupancy * (1 - sheltering) / cell_area."""
    d = df.copy()
    d["mean_veh"] = d["veh_s"] / duration_s
    d["people"] = d["mean_veh"] * d["vtype"].map(ass.occupancy).fillna(1.0)
    d["unsheltered"] = d["people"] * (1 - d["vtype"].map(ass.sheltering).fillna(0.5))
    g = d.groupby(["row", "col"], as_index=False)[["mean_veh", "people", "unsheltered"]].sum()
    g["rho_road_eff"] = g["unsheltered"] / grid.cell ** 2
    return g


def combine_drones(per_drone: list[pd.DataFrame], how: str = "max") -> pd.DataFrame:
    """Drone fields of view overlap at their borders, so the same vehicle can appear in
    two files with different track ids.  'max' takes the larger value per cell
    (no double counting, slight downward bias where coverage is partial); 'sum' is the
    naive alternative kept only for the sensitivity check."""
    allc = pd.concat(per_drone)
    agg = allc.groupby(["row", "col"], as_index=False)
    return agg.max(numeric_only=True) if how == "max" else agg.sum(numeric_only=True)
