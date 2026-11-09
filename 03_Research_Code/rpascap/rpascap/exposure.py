"""Ground-exposure layers on the 20 m grid.

Observed (pNEUMA):  vehicle-seconds per cell, vehicle type, date, 30-min slot, drone.
Derived:            mean vehicles present per cell (vehicle-seconds / slot duration),
                    combined over overlapping drone fields of view by the per-cell maximum.
Assumed:            occupants per vehicle and the sheltering class of each occupant group;
                    outdoor pedestrian density on non-building cells (no observation exists).
Exposure classes (each with a Dalamagkidis-type sheltering parameter p_s):
   open      motorcyclists + pedestrians    (assumed p_s = 0.25)
   vehicle   car, taxi, medium-vehicle, bus occupants (assumed p_s = 0.75)
   heavy     heavy-vehicle occupants        (assumed p_s = 1.0)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.path import Path as MPath

CLASS_OF_VTYPE = {"Motorcycle": "open", "Car": "vehicle", "Taxi": "vehicle", "Medium Vehicle": "vehicle",
                  "Bus": "vehicle", "Heavy Vehicle": "heavy"}


def load_pneuma_cells(processed: Path):
    cells = pd.read_csv(processed / "pneuma_cell_vehicle_seconds_by_drone.csv.gz",
                        dtype={"date": str, "start": str, "end": str})
    inv = pd.read_csv(processed / "pneuma_file_inventory.csv", dtype={"date": str, "start": str, "end": str})
    return cells, inv


def mean_vehicles_by_slot(cells: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    """Mean number of vehicles present per (date, slot, cell, vtype), drones combined by max."""
    dur = inv.dropna(subset=["duration_s"]).set_index(["date", "drone", "start"])["duration_s"]
    c = cells.copy()
    c["dur"] = dur.reindex(list(zip(c.date, c.drone, c.start))).values
    c = c[c.dur > 60]
    c["mean_veh"] = c.veh_s / c.dur
    g = c.groupby(["date", "start", "row", "col", "vtype"], as_index=False)["mean_veh"].max()
    return g


def occupant_density(mv: pd.DataFrame, occupancy: dict, cell_m: float) -> pd.DataFrame:
    """Persons per m^2 by exposure class, per (date, slot, cell)."""
    d = mv.copy()
    d["persons"] = d.mean_veh * d.vtype.map(occupancy).fillna(1.0)
    d["cls"] = d.vtype.map(CLASS_OF_VTYPE).fillna("vehicle")
    p = d.pivot_table(index=["date", "start", "row", "col"], columns="cls", values="persons",
                      aggfunc="sum", fill_value=0.0).reset_index()
    for k in ("open", "vehicle", "heavy"):
        if k not in p:
            p[k] = 0.0
        p[k] = p[k] / cell_m ** 2
    return p


def to_grids(df: pd.DataFrame, r0: int, c0: int, nr: int, nc: int, cols=("open", "vehicle", "heavy")):
    out = {k: np.zeros((nr, nc)) for k in cols}
    rr = df.row.values - r0; cc = df.col.values - c0
    ok = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
    for k in cols:
        np.add.at(out[k], (rr[ok], cc[ok]), df[k].values[ok])
    return out


def leave_one_date_out(occ: pd.DataFrame, r0, c0, nr, nc):
    """Empirical check of the exposure layer: predict each date's mean occupant density per
    cell by the mean of the other dates; report errors per date and class."""
    dates = sorted(occ.date.unique())
    per_date = {}
    for d in dates:
        sub = occ[occ.date == d]
        n_slots = sub.start.nunique()
        g = to_grids(sub, r0, c0, nr, nc)
        per_date[d] = {k: v / max(n_slots, 1) for k, v in g.items()}
    rows = []
    for d in dates:
        others = [o for o in dates if o != d]
        if not others:
            continue
        for k in ("open", "vehicle", "heavy"):
            obs = per_date[d][k].ravel()
            pred = np.mean([per_date[o][k] for o in others], axis=0).ravel()
            m = (obs > 0) | (pred > 0)
            if m.sum() < 5:
                continue
            o, p = obs[m], pred[m]
            rows.append({"held_out_date": d, "class": k, "n_cells": int(m.sum()),
                         "corr": float(np.corrcoef(o, p)[0, 1]) if o.std() > 0 and p.std() > 0 else np.nan,
                         "mae_per_m2": float(np.abs(o - p).mean()), "mean_obs_per_m2": float(o.mean()),
                         "rel_error_total": float((p.sum() - o.sum()) / o.sum()) if o.sum() > 0 else np.nan})
    return pd.DataFrame(rows), per_date


def building_fraction(geojson_path: Path, lat0, lon0, cell, r0, c0, nr, nc, sub=4, R=6_371_000.0):
    """Fraction of each cell covered by OSM building footprints (sub x sub point sampling)."""
    frac = np.zeros((nr, nc))
    if not Path(geojson_path).exists():
        return None
    gj = json.load(open(geojson_path))
    offs = (np.arange(sub) + 0.5) / sub
    yy, xx = np.meshgrid((np.arange(nr)[:, None] + offs[None, :]).ravel(),
                         (np.arange(nc)[:, None] + offs[None, :]).ravel(), indexing="ij")
    pts_y = (yy + r0) * cell; pts_x = (xx + c0) * cell
    pts = np.column_stack([pts_x.ravel(), pts_y.ravel()])
    hit = np.zeros(pts.shape[0], bool)
    k = np.cos(np.deg2rad(lat0))

    def rings(geom):
        if geom["type"] == "Polygon":
            yield geom["coordinates"][0]
        elif geom["type"] == "MultiPolygon":
            for poly in geom["coordinates"]:
                yield poly[0]
    for f in gj["features"]:
        g = f.get("geometry")
        if not g:
            continue
        for ring in rings(g):
            a = np.asarray(ring, float)
            x = np.deg2rad(a[:, 0] - lon0) * R * k; y = np.deg2rad(a[:, 1] - lat0) * R
            bb = (pts[:, 0] >= x.min()) & (pts[:, 0] <= x.max()) & (pts[:, 1] >= y.min()) & (pts[:, 1] <= y.max())
            if bb.any():
                hit[bb] |= MPath(np.column_stack([x, y])).contains_points(pts[bb])
    h = hit.reshape(nr * sub, nc * sub) if False else hit.reshape(nr, sub, nc, sub)
    return h.mean(axis=(1, 3))


def holdout_validation(occ: pd.DataFrame, sets: dict, r0, c0, nr, nc):
    """Fit-and-freeze check: the layer fitted on the 'dev' date-slots predicts the mean
    occupant density per cell of each held-out set ('testA': other days, same slots;
    'testB': same days, later slot).  `sets` maps (date, start) -> set label."""
    o = occ.copy()
    o["set"] = [sets.get((d, s), "unlisted") for d, s in zip(o.date, o.start)]
    def layer(sub):
        n = sub.groupby(["date", "start"]).ngroups
        g = to_grids(sub, r0, c0, nr, nc)
        return {k: v / max(n, 1) for k, v in g.items()}, n
    dev, n_dev = layer(o[o.set == "dev"])
    rows = []
    for lab in sorted(set(o.set) - {"dev", "unlisted"}):
        obs, n_obs = layer(o[o.set == lab])
        for k in ("open", "vehicle", "heavy"):
            a, b = obs[k].ravel(), dev[k].ravel()
            m = (a > 0) | (b > 0)
            if m.sum() < 5:
                continue
            a, b = a[m], b[m]
            rows.append({"test_set": lab, "class": k, "n_cells": int(m.sum()), "dev_date_slots": n_dev,
                         "test_date_slots": n_obs,
                         "corr": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan,
                         "mae_per_m2": float(np.abs(a - b).mean()), "mean_obs_per_m2": float(a.mean()),
                         "rel_error_total": float((b.sum() - a.sum()) / a.sum()) if a.sum() > 0 else np.nan})
    return pd.DataFrame(rows)
