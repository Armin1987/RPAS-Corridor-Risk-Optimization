"""Assemble processed_inputs/ from the per-file parts and the manually downloaded files."""
import json, glob, hashlib, time, shutil, io, zipfile, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
import overpass_to_geojson as O

D = Path("..")                         # Downloads
OUT = Path("processed_inputs"); OUT.mkdir(exist_ok=True)
log = open(OUT / "acquisition_log.jsonl", "a")
def L(**k):
    k["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); log.write(json.dumps(k) + "\n"); print(k)
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()

L(step="pneuma_md5", file="pNEUMA_dataset.zip", md5="4e842b2200bea2f80d41c6b785a9de25",
  expected="4e842b2200bea2f80d41c6b785a9de25", ok=True, bytes=(D / "pNEUMA_dataset.zip").stat().st_size,
  note="md5sum run on the user's computer; matches the Zenodo record")
sel = pd.read_csv("pneuma_file_list.csv")
setmap = dict(zip(sel.file, sel.set)); setmap["20181024_d4_0830_0900.csv"] = "testA"
metas, frames = [], []
for m in sorted(glob.glob("parts/*.meta.json")):
    meta = json.load(open(m)); meta["member"] = Path(meta["member"]).name
    meta["set"] = setmap.get(meta["member"], "unlisted"); metas.append(meta)
    c = m.replace(".meta.json", ".cells.csv.gz")
    if Path(c).exists():
        frames.append(pd.read_csv(c, dtype={"date": str, "start": str, "end": str}))
inv = pd.DataFrame(metas); inv.to_csv(OUT / "pneuma_file_inventory.csv", index=False)
allc = pd.concat(frames, ignore_index=True)
allc.to_csv(OUT / "pneuma_cell_vehicle_seconds_by_drone.csv.gz", index=False, compression="gzip")
L(step="pneuma_processed", mode="selected members of the verified archive", files=len(metas),
  with_rows=len(frames), rows=len(allc), sets=inv.set.value_counts().to_dict(),
  added_beyond_list=["20181024_d4_0830_0900.csv"], errors=int(inv.get("error", pd.Series(dtype=str)).notna().sum()))
# wind files: identify by content
gust_files = []
for f in sorted(glob.glob(str(D / "open-meteo-*.csv"))):
    txt = open(f, encoding="utf-8").read(); body = txt.split("\n\n", 1)[1]; hdr = body.split("\n", 1)[0]
    if "previous_day1" in hdr:
        dst = "wind_dayahead_previous_runs_ecmwf_ifs025.csv"
        shutil.copy(f, OUT / dst); L(step="wind", source_file=Path(f).name, saved_as=dst, sha256=sha(f), header=txt.split("\n", 2)[1])
    elif "wind_gusts_10m" in hdr:
        miss = pd.read_csv(io.StringIO(body)).iloc[:, 1:].isna().mean().mean()
        gust_files.append((miss, f, txt))
# ERA5 reanalysis is complete over the period; the historical-forecast archive starts later
gust_files.sort()
for (miss, f, txt), dst in zip(gust_files, ["wind_era5_reanalysis.csv", "wind_historical_forecast_ecmwf_ifs025.csv"]):
    shutil.copy(f, OUT / dst); L(step="wind", source_file=Path(f).name, saved_as=dst, sha256=sha(f),
                                 missing_share=round(float(miss), 4), header=txt.split("\n", 2)[1])
O.main(str(D / "export.json"), str(OUT / "osm_buildings.geojson"))
L(step="osm", source_file="export.json", sha256=sha(D / "export.json"), saved_as="osm_buildings.geojson",
  query_bbox="37.965,23.710,38.000,23.750 (S,W,N,E)")
ok = inv.dropna(subset=["lat_min"])
bbox = (ok.lat_max.max() + 0.002, ok.lat_min.min() - 0.002, ok.lon_max.max() + 0.002, ok.lon_min.min() - 0.002)
json.dump({"LAT0": 37.975, "LON0": 23.725, "CELL_m": 20.0, "projection": "local equirectangular",
           "bbox": dict(zip(["north", "south", "east", "west"], bbox))}, open(OUT / "grid_meta.json", "w"), indent=1)
L(step="study_bbox", north=bbox[0], south=bbox[1], east=bbox[2], west=bbox[3])
log.close()
man = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "MANIFEST.json"}
json.dump(man, open(OUT / "MANIFEST.json", "w"), indent=1)
with zipfile.ZipFile("processed_inputs.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.iterdir(): z.write(p, arcname=p.name)
print("zip MB", Path("processed_inputs.zip").stat().st_size / 1e6)
