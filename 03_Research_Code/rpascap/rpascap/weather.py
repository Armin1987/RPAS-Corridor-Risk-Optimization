"""Open-Meteo CSV ingestion, daily wind matrices, quality control and context features.

Open-Meteo CSV layout: a metadata block (latitude, longitude, ..., timezone), a blank
line, then a header 'time,<var> (<unit>),...' and hourly rows in local time.
Realised wind (outcome):   ERA5 reanalysis, 100 m (wind_era5_reanalysis.csv)
Day-ahead context:         ECMWF IFS 0.25 forecast issued 24 h before valid time
                           (wind_speed_100m_previous_day1; Previous Runs API)
Short-range proxy (nowcast robustness test): Historical Forecast API, 100 m.
"""
from __future__ import annotations
import io
import re
import numpy as np
import pandas as pd
from pathlib import Path


def read_open_meteo_csv(path: Path) -> tuple[pd.DataFrame, dict]:
    txt = Path(path).read_text()
    lines = txt.splitlines()
    meta = {}
    if lines and lines[0].startswith("latitude"):
        keys = lines[0].split(","); vals = lines[1].split(",")
        meta = dict(zip(keys, vals))
    start = next(i for i, l in enumerate(lines) if l.startswith("time"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df.columns = [re.sub(r"\s*\(.*\)$", "", c).strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time"), meta


def daily_matrix(series: pd.Series, hours: list[int]) -> pd.DataFrame:
    """Rows = dates, columns = operating hours; NaN where missing."""
    s = series.copy()
    df = pd.DataFrame({"v": s.values, "date": s.index.normalize(), "h": s.index.hour})
    m = df[df.h.isin(hours)].pivot_table(index="date", columns="h", values="v", aggfunc="mean")
    return m.reindex(columns=hours)


def circular_mean_deg(a: np.ndarray, axis=None):
    r = np.deg2rad(a)
    return (np.rad2deg(np.arctan2(np.nanmean(np.sin(r), axis=axis), np.nanmean(np.cos(r), axis=axis))) + 360) % 360


def build_days(realised: pd.DataFrame, forecast: pd.DataFrame, hours: list[int],
               spd_col="wind_speed_100m", dir_col="wind_direction_100m",
               fspd_col="wind_speed_100m_previous_day1", fdir_col="wind_direction_100m_previous_day1"):
    """Return a dict of aligned daily matrices and a QC table (one row per date)."""
    W = daily_matrix(realised[spd_col], hours)
    D = daily_matrix(realised[dir_col], hours)
    F = daily_matrix(forecast[fspd_col], hours)
    FD = daily_matrix(forecast[fdir_col], hours) if fdir_col in forecast else D * np.nan
    idx = W.index.intersection(F.index)
    W, D, F, FD = W.loc[idx], D.loc[idx], F.loc[idx], FD.loc[idx]
    has_fd = fdir_col in forecast and FD.notna().any().any()
    qc = pd.DataFrame({"date": idx,
                       "missing_realised_h": W.isna().sum(axis=1).values + D.isna().sum(axis=1).values,
                       "missing_forecast_h": F.isna().sum(axis=1).values + (FD.isna().sum(axis=1).values if has_fd else 0)})
    qc["complete"] = (qc.missing_realised_h == 0) & (qc.missing_forecast_h == 0)
    ok = qc.complete.values
    return {"W": W[ok], "D": D[ok], "F": F[ok], "FD": FD[ok]}, qc


def context_features(F: pd.DataFrame, FD: pd.DataFrame, blocks: list[list[int]]) -> pd.DataFrame:
    """Day-ahead context per date: block mean and max forecast speed, vector-mean forecast
    direction (sin, cos) and day of year (sin, cos).  Known at decision time t0."""
    X = pd.DataFrame(index=F.index)
    for bi, (h0, h1) in enumerate(blocks):
        cols = [h for h in F.columns if h0 <= h <= h1]
        X[f"f_mean_b{bi}"] = F[cols].mean(axis=1)
        X[f"f_max_b{bi}"] = F[cols].max(axis=1)
    if FD.notna().any().any():                      # direction forecast available
        r = np.deg2rad(FD.values)
        X["f_dir_sin"] = np.nanmean(np.sin(r), axis=1)
        X["f_dir_cos"] = np.nanmean(np.cos(r), axis=1)
    doy = F.index.dayofyear.values
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return X


def forecast_skill(F: pd.DataFrame, W: pd.DataFrame) -> dict:
    e = (F.values - W.values).ravel()
    e = e[~np.isnan(e)]
    f, w = F.values.ravel(), W.values.ravel()
    m = ~np.isnan(f) & ~np.isnan(w)
    return {"bias_ms": float(e.mean()), "mae_ms": float(np.abs(e).mean()),
            "rmse_ms": float(np.sqrt((e ** 2).mean())), "corr": float(np.corrcoef(f[m], w[m])[0, 1]),
            "n_hours": int(m.sum())}
