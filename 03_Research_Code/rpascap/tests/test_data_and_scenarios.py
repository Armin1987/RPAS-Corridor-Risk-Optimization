import numpy as np
import pandas as pd
import pytest
from rpascap.data import pneuma
from rpascap.scenarios import contextual as cx


def _fake_pneuma(swap=False, delim="; ", fps=25.0, n_veh=3, seconds=10.0):
    """A synthetic file in the official pNEUMA layout (NOT real data)."""
    hdr = delim.join(["track_id", "type", "traveled_d", "avg_speed", "lat", "lon", "speed",
                      "lon_acc", "lat_acc", "time"])
    rows = [hdr]
    dt = 1.0 / fps
    for v in range(n_veh):
        t = np.arange(0, seconds, dt)
        lat = 37.98 + 1e-5 * v + 0 * t
        lon = 23.73 + 1e-6 * t * 10
        blk = []
        for i in range(t.size):
            a, b = (lon[i], lat[i]) if swap else (lat[i], lon[i])
            blk += [f"{a:.6f}", f"{b:.6f}", "20.0", "0.0", "0.0", f"{t[i]:.2f}"]
        rows.append(delim.join([str(v + 1), "Car", "50.0", "20.0"] + blk))
    return "\n".join(rows) + "\n"


def test_filename_parsing():
    assert pneuma.parse_name("20181024_d4_0830_0900.csv") == {
        "date": "20181024", "drone": 4, "start": "0830", "end": "0900"}


@pytest.mark.parametrize("swap,delim", [(False, "; "), (True, "; "), (False, ",")])
def test_layout_detection(swap, delim):
    txt = _fake_pneuma(swap=swap, delim=delim)
    lay = pneuma.detect_layout(txt.splitlines())
    assert lay.lat_idx == (1 if swap else 0)


def test_layout_rejects_non_athens():
    bad = "1; Car; 1; 1; 10.0; 10.0; 1; 0; 0; 0.0\n"
    with pytest.raises(ValueError):
        pneuma.detect_layout([bad])


@pytest.mark.parametrize("fps", [25.0, 23.97])
def test_vehicle_seconds_conserved(fps):
    txt = _fake_pneuma(fps=fps, n_veh=3, seconds=10.0)
    g = pneuma.Grid(37.98, 23.73, 20.0)
    df, dur = pneuma.vehicle_seconds_by_cell(txt, g)
    # 3 vehicles x 10 s each, time-weighted (independent of frame rate)
    assert abs(df["veh_s"].sum() - 30.0) < 0.2
    assert abs(dur - 10.0) < 0.1


def test_exposure_assumptions_applied():
    txt = _fake_pneuma(n_veh=2)
    g = pneuma.Grid(37.98, 23.73, 20.0)
    df, dur = pneuma.vehicle_seconds_by_cell(txt, g)
    e = pneuma.road_occupant_exposure(df, dur, g, pneuma.ExposureAssumptions())
    expected_people = df["veh_s"].sum() / dur * 1.3
    assert np.isclose(e["people"].sum(), expected_people, rtol=1e-6)
    assert np.isclose(e["unsheltered"].sum(), expected_people * 0.3, rtol=1e-6)


def test_combine_drones_no_double_count():
    a = pd.DataFrame({"row": [0, 1], "col": [0, 0], "people": [2.0, 1.0]})
    b = pd.DataFrame({"row": [0], "col": [0], "people": [2.0]})
    m = pneuma.combine_drones([a, b], "max")
    assert m.loc[(m.row == 0) & (m.col == 0), "people"].item() == 2.0


def test_weights():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    assert np.isclose(cx.saa_weights(100).sum(), 1)
    w = cx.knn_weights(X, X[5], k=10)
    assert np.isclose(w.sum(), 1) and w[5] == 0.1
    assert np.allclose(cx.knn_weights(X, X[0], k=100), cx.saa_weights(100))
    s = cx.seasonal_weights(np.arange(1, 366), 5, 10)
    assert np.isclose(s.sum(), 1) and s[360] > 0      # wraps around year end
    idx, ww = cx.reduce_scenarios(w, 5)
    assert idx.size == 5 and np.isclose(ww.sum(), 1)


def test_rf_weights_sum_to_one():
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4)); y = X[:, 0] + rng.normal(0, .1, 200)
    f = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, random_state=0).fit(X, y)
    w = cx.rf_weights(f, X, X[0])
    assert np.isclose(w.sum(), 1) and (w >= 0).all()


def test_weather_without_forecast_direction(tmp_path):
    import numpy as np, pandas as pd
    from rpascap import weather as wx
    t = pd.date_range("2024-01-01", "2024-01-10 23:00", freq="h")
    real = pd.DataFrame({"wind_speed_100m": np.linspace(1, 9, len(t)), "wind_direction_100m": 180.0}, index=t)
    fc = pd.DataFrame({"wind_speed_100m_previous_day1": np.linspace(1, 9, len(t))}, index=t)
    days, qc = wx.build_days(real, fc, list(range(7, 23)))
    assert len(days["W"]) == 10 and qc.complete.all()
    X = wx.context_features(days["F"], days["FD"], [[7, 10], [11, 14], [15, 18], [19, 22]])
    assert not X.isna().any().any() and "f_dir_sin" not in X


def test_open_meteo_csv_parser(tmp_path):
    from rpascap import weather as wx
    p = tmp_path / "w.csv"
    p.write_text("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n37.98,23.73,95.0,7200,Europe/Athens,GMT+2\n\n"
                 "time,wind_speed_100m (m/s),wind_direction_100m (°)\n2024-01-01T00:00,3.5,200\n2024-01-01T01:00,4.0,210\n")
    df, meta = wx.read_open_meteo_csv(p)
    assert list(df.columns) == ["wind_speed_100m", "wind_direction_100m"] and meta["timezone"] == "Europe/Athens"
