"""Acquire and process the official open data for the Athens case (run in Google Colab).

Run:   !python scripts/acquire_and_process.py --out processed_inputs
Output: processed_inputs.zip (small: processed tables + logs + checksums) to upload back.

Steps (each step logs success/failure; a failed optional step never stops the run):
  1. pNEUMA (Zenodo 10.5281/zenodo.10491409, CC BY 4.0): resumable download of
     pNEUMA_dataset.zip, MD5 check, inventory of all trajectory CSVs, streamed processing
     of every CSV directly from the archive (no full extraction) into time-weighted
     vehicle-seconds per 20 m cell, vehicle type, date, 30-min slot and drone.
  2. Wind (Open-Meteo): day-ahead forecast archive (Previous Runs API, lead 24 h),
     stitched short-range forecasts (Historical Forecast API) and ERA5 reanalysis
     (Historical Weather API) for the study area, hourly, 2024-01-01..2026-10-31.
  3. OpenStreetMap buildings and roads for the study bounding box (optional; ODbL).
  4. Manifest with SHA-256 of every output file and the software environment.
Nothing here is synthetic.  Missing pieces are reported in acquisition_log.jsonl.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ZENODO_URL = "https://zenodo.org/records/10491409/files/pNEUMA_dataset.zip?download=1"
PNEUMA_MD5 = "4e842b2200bea2f80d41c6b785a9de25"           # from the Zenodo record page
LAT0, LON0, CELL, R_EARTH = 37.975, 23.725, 20.0, 6_371_000.0   # fixed grid reference
FILE_RE = re.compile(r"(\d{8})_d(\d+)_(\d{4})_(\d{4})")
SET_OF = {}   # file name -> design set (dev/testA/testB), from --file-list when it has a 'set' column
VTYPES = ["Car", "Taxi", "Motorcycle", "Bus", "Medium Vehicle", "Heavy Vehicle"]

def log(out: Path, **kw):
    kw["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(kw), flush=True)
    with open(out / "acquisition_log.jsonl", "a") as f:
        f.write(json.dumps(kw) + "\n")

def md5sum(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()

# ------------------------------------------------------------------ 1. pNEUMA
def download_resumable(url: str, dest: Path, out: Path):
    import requests
    pos = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, stream=True, headers=headers, timeout=120) as r:
        if r.status_code == 416:     # already complete
            return
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + pos
        mode = "ab" if pos and r.status_code == 206 else "wb"
        done, t0 = (pos if mode == "ab" else 0), time.time()
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 24):
                f.write(chunk); done += len(chunk)
                if time.time() - t0 > 30:
                    print(f"  {done/1e9:.2f}/{total/1e9:.2f} GB", flush=True); t0 = time.time()
    log(out, step="pneuma_download", url=url, bytes=dest.stat().st_size)

def parse_rows(fh, delim=None):
    """Yield (vtype, block-array[n,6]) from an official pNEUMA CSV stream."""
    order = None
    for raw in fh:
        line = raw.strip()
        if not line or not line[0].isdigit():
            continue
        if delim is None:
            delim = ";" if line.count(";") > line.count(",") else ","
        parts = line.split(delim, 4)
        if len(parts) < 5:
            continue
        vtype = parts[1].strip()
        vals = np.fromstring(parts[4], sep=delim)
        n = vals.size // 6
        if n < 2:
            continue
        a = vals[: n * 6].reshape(n, 6)
        if order is None:                      # detect lat/lon order from the first row
            if 37.85 < a[0, 0] < 38.10 and 23.60 < a[0, 1] < 23.90:
                order = (0, 1)
            elif 37.85 < a[0, 1] < 38.10 and 23.60 < a[0, 0] < 23.90:
                order = (1, 0)
            else:
                raise ValueError(f"unrecognised coordinates {a[0,:2]}")
        yield vtype, a[:, order[0]], a[:, order[1]], a[:, 2], a[:, 5]

def process_member(zf, name: str):
    """zf: an open ZipFile, or None to read `name` as a plain file path."""
    meta = {"member": name}
    m = FILE_RE.search(name)
    if m:
        meta.update(date=m.group(1), drone=int(m.group(2)), start=m.group(3), end=m.group(4))
    keys, w_all, v_all = [], [], []
    nveh = {t: 0 for t in VTYPES}; npts = 0; tmin, tmax = np.inf, -np.inf
    dts = []; lat_rng = [90, -90]; lon_rng = [180, -180]
    with (zf.open(name) if zf is not None else open(name, "rb")) as raw:
        fh = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        for vtype, lat, lon, spd, t in parse_rows(fh):
            nveh[vtype] = nveh.get(vtype, 0) + 1
            npts += t.size
            dt = np.diff(t); med = np.median(dt) if dt.size else 0.04
            dts.append(med)
            w = np.append(dt, med)
            w = np.clip(w, 0, 1.0)              # gaps > 1 s are not counted as presence
            x = np.deg2rad(lon - LON0) * R_EARTH * np.cos(np.deg2rad(LAT0))
            y = np.deg2rad(lat - LAT0) * R_EARTH
            r = np.floor(y / CELL).astype(np.int32); c = np.floor(x / CELL).astype(np.int32)
            ti = VTYPES.index(vtype) if vtype in VTYPES else len(VTYPES)
            keys.append((r.astype(np.int64) + 100000) * 10_000_000 + (c.astype(np.int64) + 100000) * 10 + ti)
            w_all.append(w); v_all.append(spd * w)
            tmin, tmax = min(tmin, t.min()), max(tmax, t.max())
            lat_rng = [min(lat_rng[0], lat.min()), max(lat_rng[1], lat.max())]
            lon_rng = [min(lon_rng[0], lon.min()), max(lon_rng[1], lon.max())]
    if not keys:
        return None, meta
    K = np.concatenate(keys); W = np.concatenate(w_all); V = np.concatenate(v_all)
    uk, inv = np.unique(K, return_inverse=True)
    veh_s = np.bincount(inv, weights=W); spd_s = np.bincount(inv, weights=V)
    ti = uk % 10; c = (uk // 10) % 1_000_000 - 100000; r = uk // 10_000_000 - 100000
    df = pd.DataFrame({"row": r, "col": c,
                       "vtype": [VTYPES[i] if i < len(VTYPES) else "Other" for i in ti],
                       "veh_s": veh_s, "mean_speed_kmh": spd_s / np.maximum(veh_s, 1e-9)})
    for k in ("date", "drone", "start", "end"):
        df[k] = meta.get(k)
    meta.update(n_vehicles=sum(nveh.values()), **{f"n_{k.replace(' ', '_')}": v for k, v in nveh.items()},
                n_points=int(npts), t_start=float(tmin), t_end=float(tmax), duration_s=float(tmax - tmin),
                median_dt_s=float(np.median(dts)), lat_min=lat_rng[0], lat_max=lat_rng[1],
                lon_min=lon_rng[0], lon_max=lon_rng[1], n_cells=int(df[["row", "col"]].drop_duplicates().shape[0]))
    return df, meta

def step_pneuma(work: Path, out: Path, max_files: int | None, zip_path: str | None = None, only: set | None = None):
    z = Path(zip_path) if zip_path else work / "pNEUMA_dataset.zip"
    if not zip_path:
        download_resumable(ZENODO_URL, z, out)
    got = md5sum(z)
    log(out, step="pneuma_md5", md5=got, expected=PNEUMA_MD5, ok=(got == PNEUMA_MD5))
    if got != PNEUMA_MD5:
        raise RuntimeError("MD5 mismatch - archive incomplete or different; rerun to resume")
    with zipfile.ZipFile(z) as zf:
        members = [i for i in zf.infolist() if i.filename.lower().endswith(".csv")]
        if only:
            members = [i for i in members if Path(i.filename).name in only]
            missing = sorted(only - {Path(i.filename).name for i in members})
            log(out, step="pneuma_selection", n_selected=len(members), missing_from_archive=missing)
        inv = pd.DataFrame([{"member": i.filename, "bytes_uncompressed": i.file_size,
                             "bytes_compressed": i.compress_size} for i in members])
        inv.to_csv(out / "pneuma_archive_members.csv", index=False)
        other = [i.filename for i in zf.infolist() if not i.filename.lower().endswith(".csv")]
        log(out, step="pneuma_inventory", n_csv=len(members), other_members=other[:50])
        frames, metas = [], []
        for k, info in enumerate(members[: max_files or None]):
            t0 = time.time()
            try:
                df, meta = process_member(zf, info.filename)
                if df is not None:
                    frames.append(df)
                meta["seconds"] = round(time.time() - t0, 1); metas.append(meta)
                print(f"[{k+1}/{len(members)}] {info.filename}: {meta.get('n_vehicles')} veh, {meta['seconds']} s", flush=True)
            except Exception as e:  # keep going, record the failure
                metas.append({"member": info.filename, "error": repr(e)})
                log(out, step="pneuma_file_error", member=info.filename, error=repr(e))
    inv = pd.DataFrame(metas)
    if SET_OF:
        inv["set"] = [SET_OF.get(Path(str(m)).name, "unlisted") for m in inv.member]
    inv.to_csv(out / "pneuma_file_inventory.csv", index=False)
    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(out / "pneuma_cell_vehicle_seconds_by_drone.csv.gz", index=False, compression="gzip")
    log(out, step="pneuma_processed", rows=len(allc), files=len(frames))

def step_pneuma_dir(src: Path, out: Path):
    """Selective download: process pNEUMA CSVs placed in a folder, as downloaded one by one
    from open-traffic.epfl.ch (loose CSVs and/or zip files containing them).  Same per-file
    processing as the archive route; the set of files used is recorded in the inventory."""
    items = []
    for p in sorted(src.rglob("*")):
        if p.suffix.lower() == ".csv" and FILE_RE.search(p.name):
            items.append((None, str(p), p.stat().st_size))
        elif p.suffix.lower() == ".zip":
            zf = zipfile.ZipFile(p)
            items += [(zf, i.filename, i.file_size) for i in zf.infolist()
                      if i.filename.lower().endswith(".csv") and FILE_RE.search(i.filename)]
    seen, uniq = set(), []
    for zf, name, size in items:                       # same file downloaded twice -> once
        key = FILE_RE.search(Path(name).name).group(0)
        if key not in seen:
            seen.add(key); uniq.append((zf, name, size))
    pd.DataFrame([{"member": Path(n).name, "bytes_uncompressed": b} for _, n, b in uniq]).to_csv(
        out / "pneuma_archive_members.csv", index=False)
    log(out, step="pneuma_inventory", mode="selective-folder", n_csv=len(uniq), source=str(src))
    frames, metas = [], []
    for k, (zf, name, _) in enumerate(uniq):
        t0 = time.time()
        try:
            df, meta = process_member(zf, name)
            meta["member"] = Path(name).name
            if df is not None:
                frames.append(df)
            meta["seconds"] = round(time.time() - t0, 1); metas.append(meta)
            print(f"[{k+1}/{len(uniq)}] {Path(name).name}: {meta.get('n_vehicles')} veh, {meta['seconds']} s", flush=True)
        except Exception as e:
            metas.append({"member": Path(name).name, "error": repr(e)})
            log(out, step="pneuma_file_error", member=name, error=repr(e))
    inv = pd.DataFrame(metas)
    if SET_OF:
        inv["set"] = [SET_OF.get(Path(str(m)).name, "unlisted") for m in inv.member]
    inv.to_csv(out / "pneuma_file_inventory.csv", index=False)
    allc = pd.concat(frames, ignore_index=True)
    allc.to_csv(out / "pneuma_cell_vehicle_seconds_by_drone.csv.gz", index=False, compression="gzip")
    log(out, step="pneuma_processed", rows=len(allc), files=len(frames))


# ------------------------------------------------------------------ 2. Wind
def step_wind(out: Path, lat: float, lon: float, start="2024-01-01", end="2026-10-31"):
    import requests
    common = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                  wind_speed_unit="ms", timezone="Europe/Athens", format="csv")
    jobs = {
        "wind_dayahead_previous_runs_ecmwf_ifs025.csv": ("https://previous-runs-api.open-meteo.com/v1/forecast", dict(
            hourly="wind_speed_10m,wind_speed_100m,wind_direction_100m,"
                   "wind_speed_10m_previous_day1,wind_speed_100m_previous_day1,wind_direction_100m_previous_day1,"
                   "wind_speed_100m_previous_day2,wind_direction_100m_previous_day2",
            models="ecmwf_ifs025")),
        "wind_historical_forecast_ecmwf_ifs025.csv": ("https://historical-forecast-api.open-meteo.com/v1/forecast", dict(
            hourly="wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_gusts_10m", models="ecmwf_ifs025")),
        "wind_era5_reanalysis.csv": ("https://archive-api.open-meteo.com/v1/archive", dict(
            hourly="wind_speed_10m,wind_speed_100m,wind_direction_100m,wind_gusts_10m", models="era5")),
    }
    for fname, (url, extra) in jobs.items():
        try:
            r = requests.get(url, params={**common, **extra}, timeout=180)
            log(out, step="wind", file=fname, url=r.url, status=r.status_code, bytes=len(r.content))
            if r.ok:
                (out / fname).write_bytes(r.content)
            else:
                (out / (fname + ".error.txt")).write_text(r.text[:5000])
        except Exception as e:
            log(out, step="wind", file=fname, error=repr(e))

# ------------------------------------------------------------------ 3. OSM
def step_osm(out: Path, bbox):
    try:
        import osmnx as ox
        north, south, east, west = bbox
        tags_b = {"building": True}
        try:
            b = ox.features_from_bbox(bbox=(west, south, east, north), tags=tags_b)
        except TypeError:
            b = ox.features_from_bbox(north, south, east, west, tags=tags_b)
        b = b[b.geometry.type.isin(["Polygon", "MultiPolygon"])]
        keep = [c for c in ["building", "building:levels", "height", "geometry"] if c in b.columns]
        b[keep].reset_index().to_file(out / "osm_buildings.geojson", driver="GeoJSON")
        try:
            g = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
        except TypeError:
            g = ox.graph_from_bbox(north, south, east, west, network_type="drive")
        edges = ox.graph_to_gdfs(g, nodes=False)
        keep = [c for c in ["highway", "lanes", "name", "length", "geometry"] if c in edges.columns]
        e2 = edges[keep].reset_index()
        for col in e2.columns:
            if e2[col].dtype == object and col != "geometry":
                e2[col] = e2[col].astype(str)
        e2.to_file(out / "osm_roads.geojson", driver="GeoJSON")
        log(out, step="osm", buildings=len(b), road_edges=len(edges), osmnx=ox.__version__)
    except Exception as e:
        log(out, step="osm", error=repr(e))

# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="processed_inputs")
    ap.add_argument("--work", default="/content/work" if Path("/content").exists() else "work")
    ap.add_argument("--max-files", type=int, default=None, help="debug: process only N CSVs")
    ap.add_argument("--skip-pneuma", action="store_true")
    ap.add_argument("--pneuma-dir", default=None, help="folder of pNEUMA CSVs downloaded selectively (no Zenodo archive)")
    ap.add_argument("--zip", default=None, help="local copy of pNEUMA_dataset.zip (MD5-checked; no download)")
    ap.add_argument("--file-list", default=None, help="CSV with a 'file' column: process only these archive members")
    ap.add_argument("--add-file", action="append", help="extra archive member to process (repeatable)")
    ap.add_argument("--add-file-set", default="testA", help="design set of the --add-file members")
    ap.add_argument("--skip-online", action="store_true", help="do not call Open-Meteo / OSM (files supplied manually)")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    log(out, step="start", python=sys.version, argv=sys.argv)
    if a.pneuma_dir:
        step_pneuma_dir(Path(a.pneuma_dir), out)
    elif not a.skip_pneuma:
        try:
            only = None
            if a.file_list:
                fl = pd.read_csv(a.file_list)
                only = set(fl["file"]) | set(a.add_file or [])
                if "set" in fl:
                    SET_OF.update(dict(zip(fl.file, fl.set)))
                    for f in a.add_file or []:
                        SET_OF.setdefault(f, a.add_file_set)
            step_pneuma(work, out, a.max_files, a.zip, only)
        except Exception as e:
            log(out, step="pneuma", error=repr(e))
    inv = out / "pneuma_file_inventory.csv"
    if inv.exists():
        d = pd.read_csv(inv)
        bbox = (d.lat_max.max() + 0.002, d.lat_min.min() - 0.002, d.lon_max.max() + 0.002, d.lon_min.min() - 0.002)
    else:
        bbox = (37.995, 37.970, 23.745, 23.715)
    lat_c, lon_c = (bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2
    log(out, step="study_bbox", north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3])
    if not a.skip_online:
        step_wind(out, round(lat_c, 4), round(lon_c, 4))
        step_osm(out, bbox)
    json.dump({"LAT0": LAT0, "LON0": LON0, "CELL_m": CELL, "projection": "local equirectangular",
               "bbox": dict(zip(["north", "south", "east", "west"], bbox))},
              open(out / "grid_meta.json", "w"), indent=1)
    (out / "pip_freeze.txt").write_text(subprocess.run([sys.executable, "-m", "pip", "freeze"],
                                                       capture_output=True, text=True).stdout)
    man = {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    json.dump(man, open(out / "MANIFEST.json", "w"), indent=1)
    zpath = Path(str(out) + ".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir():
            zf.write(p, arcname=p.name)
    print(f"\nDONE -> {zpath} ({zpath.stat().st_size/1e6:.1f} MB). Upload this file to the chat.")

if __name__ == "__main__":
    main()
