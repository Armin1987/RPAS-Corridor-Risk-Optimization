"""Download helpers for the official open sources (run in Colab or locally; the
authoring workspace had no outbound network, so these were NOT executed there).

Every download is checksummed or logged with retrieval time, URL and response size,
so the provenance report can be generated from the log.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

# Verified from the Zenodo record page (v1, 2024-01-11): file name, size 15.8 GB, md5.
PNEUMA_ZENODO_RECORD = "10491409"
PNEUMA_ZIP = "pNEUMA_dataset.zip"
PNEUMA_MD5 = "4e842b2200bea2f80d41c6b785a9de25"
# Standard Zenodo file-URL pattern; confirm on first run (the md5 check will fail loudly
# if the wrong object is retrieved).
PNEUMA_URL = f"https://zenodo.org/records/{PNEUMA_ZENODO_RECORD}/files/{PNEUMA_ZIP}?download=1"

OPEN_METEO_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
OPEN_METEO_HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def md5sum(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def log(entry: dict, logfile: Path):
    entry["retrieved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(logfile, "a") as f:
        f.write(json.dumps(entry) + "\n")


def download_pneuma(dest: Path, logfile: Path):
    import requests
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / PNEUMA_ZIP
    if not out.exists():
        with requests.get(PNEUMA_URL, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out, "wb") as f:
                for b in r.iter_content(1 << 24):
                    f.write(b)
    got = md5sum(out)
    log({"source": "pNEUMA", "url": PNEUMA_URL, "md5": got, "expected_md5": PNEUMA_MD5,
         "bytes": out.stat().st_size}, logfile)
    if got != PNEUMA_MD5:
        raise RuntimeError(f"pNEUMA md5 mismatch: {got} != {PNEUMA_MD5}")
    return out


def fetch_wind(lat: float, lon: float, start: str, end: str, logfile: Path,
               model: str = "ecmwf_ifs025"):
    """Day-ahead forecast (lead 24 h, '_previous_day1') and the matching stitched
    short-range series used as the realised-wind proxy, hourly, at 120 m and 10 m.
    Open-Meteo documents the Previous Runs archive from ~Jan 2024 for most models."""
    import requests
    vars_fc = ["wind_speed_120m_previous_day1", "wind_direction_120m_previous_day1",
               "wind_speed_10m_previous_day1"]
    vars_obs = ["wind_speed_120m", "wind_direction_120m", "wind_speed_10m", "wind_gusts_10m"]
    q = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
         "wind_speed_unit": "ms", "timezone": "Europe/Athens", "models": model}
    fc = requests.get(OPEN_METEO_PREVIOUS_RUNS, params={**q, "hourly": ",".join(vars_fc)}, timeout=60)
    ob = requests.get(OPEN_METEO_HIST_FORECAST, params={**q, "hourly": ",".join(vars_obs)}, timeout=60)
    for name, r in (("forecast_d1", fc), ("realised_proxy", ob)):
        log({"source": f"open-meteo:{name}", "url": r.url, "status": r.status_code,
             "bytes": len(r.content)}, logfile)
        r.raise_for_status()
    return fc.json(), ob.json()
