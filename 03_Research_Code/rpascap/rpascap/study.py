"""End-to-end study pipeline: setup from processed official inputs, policies, out-of-sample
evaluation.  Every number written to results/ comes from functions in this module.
"""
from __future__ import annotations
import json, time, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import correlate
from sklearn.ensemble import RandomForestRegressor

from . import exposure as ex, weather as wx, navmodels as nm, tables as tb
from .geometry import Corridor, DEFAULT_MENU, lanes, lane_points
from .instances import planned, on_demand, profile
from .model.two_stage import ConfigData, Instance, solve_decomposed as solve, recompute_risk
from .scenarios import contextual as cx
from .model.fast import solve_fast, CapacityCache
from .model.societal import solve_societal

CAP = CapacityCache()

CLASSES = ["ped", "moto", "vehicle", "heavy"]


# ------------------------------------------------------------------ setup
class Study:
    def __init__(self, processed: Path, cfg: dict, cache: Path, log=print):
        self.cfg, self.log, self.cache = cfg, log, Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.processed = Path(processed)
        g = json.load(open(self.processed / "grid_meta.json"))
        self.lat0, self.lon0, self.cell = g["LAT0"], g["LON0"], g["CELL_m"]
        self.hours = cfg["hours"]; self.blocks = cfg["blocks"]
        self.block_of_t = np.array([next(i for i, (a, b) in enumerate(self.blocks) if a <= h <= b) for h in self.hours])
        self._exposure(); self._corridors(); self._weather(); self._demand()

    # ---- exposure ----------------------------------------------------
    def _exposure(self):
        cells, inv = ex.load_pneuma_cells(self.processed)
        self.inventory = inv
        mv = ex.mean_vehicles_by_slot(cells, inv)
        occ = self.cfg["occupancy"]
        d = mv.copy()
        d["persons"] = d.mean_veh * d.vtype.map(occ).fillna(1.0)
        cls = {"Motorcycle": "moto", "Car": "vehicle", "Taxi": "vehicle", "Medium Vehicle": "vehicle",
               "Bus": "vehicle", "Heavy Vehicle": "heavy"}
        d["cls"] = d.vtype.map(cls).fillna("vehicle")
        self.mean_veh = mv
        # coverage from per-file bounding boxes (drone fields of view)
        ok = inv.dropna(subset=["lat_min"])
        R = 6_371_000.0; kx = np.cos(np.deg2rad(self.lat0))
        def xy(lat, lon):
            return np.deg2rad(lon - self.lon0) * R * kx, np.deg2rad(lat - self.lat0) * R
        boxes = []
        for _, f in ok.iterrows():
            x0, y0 = xy(f.lat_min, f.lon_min); x1, y1 = xy(f.lat_max, f.lon_max)
            boxes.append((x0, y0, x1, y1))
        self.boxes = boxes
        X0 = min(b[0] for b in boxes); Y0 = min(b[1] for b in boxes)
        X1 = max(b[2] for b in boxes); Y1 = max(b[3] for b in boxes)
        pad = 10
        self.r0 = int(np.floor(Y0 / self.cell)) - pad; self.c0 = int(np.floor(X0 / self.cell)) - pad
        self.nr = int(np.ceil(Y1 / self.cell)) + pad - self.r0; self.nc = int(np.ceil(X1 / self.cell)) + pad - self.c0
        cov = np.zeros((self.nr, self.nc), bool)
        for (x0, y0, x1, y1) in boxes:
            r_a, r_b = int(np.floor(y0 / self.cell)) - self.r0, int(np.ceil(y1 / self.cell)) - self.r0
            c_a, c_b = int(np.floor(x0 / self.cell)) - self.c0, int(np.ceil(x1 / self.cell)) - self.c0
            cov[max(r_a, 0):r_b, max(c_a, 0):c_b] = True
        self.coverage = cov
        # Fit-and-freeze design (pNEUMA file list): if the inventory labels files 'dev', the
        # road layer used by the model is fitted on the dev date-slots only; the held-out sets
        # are used for validation only.  Without labels all date-slots are used.
        self.exposure_sets = None
        if "set" in inv.columns and (inv.set == "dev").any():
            self.exposure_sets = {(r.date, r.start): r.set for r in inv.itertuples()}
            dev_keys = {k for k, v in self.exposure_sets.items() if v == "dev"}
            d = d[[(a, b) in dev_keys for a, b in zip(d.date, d.start)]]
        # per date x slot persons/m2 by class, then mean over the fitting date-slots
        n_ds = d.groupby(["date", "start"]).ngroups
        self.n_date_slots = n_ds
        self.rho_road = {}
        for k in ("moto", "vehicle", "heavy"):
            sub = d[d.cls == k].groupby(["row", "col"], as_index=False)["persons"].sum()
            grid = np.zeros((self.nr, self.nc))
            rr, cc = sub.row.values - self.r0, sub.col.values - self.c0
            m = (rr >= 0) & (rr < self.nr) & (cc >= 0) & (cc < self.nc)
            np.add.at(grid, (rr[m], cc[m]), sub.persons.values[m])
            self.rho_road[k] = grid / n_ds / self.cell ** 2
        # held-out-date validation of the road layer
        occd = ex.occupant_density(mv, occ, self.cell)
        self.lodo, _ = ex.leave_one_date_out(occd, self.r0, self.c0, self.nr, self.nc)
        self.holdout = (ex.holdout_validation(occd, self.exposure_sets, self.r0, self.c0, self.nr, self.nc)
                        if self.exposure_sets else pd.DataFrame())
        # buildings (OSM) -> open-ground mask for the pedestrian layer
        bf = ex.building_fraction(self.processed / "osm_buildings.geojson", self.lat0, self.lon0, self.cell,
                                  self.r0, self.c0, self.nr, self.nc)
        self.building_frac = bf if bf is not None else np.zeros((self.nr, self.nc))
        self.has_osm = bf is not None
        self.ped_unit = (1.0 - self.building_frac) * cov            # x ped density (persons/m2)
        self.log(f"exposure: grid {self.nr}x{self.nc}, date-slots {n_ds}, OSM buildings: {self.has_osm}")

    # ---- corridors ------------------------------------------------------
    def _corridors(self):
        c = self.cfg["corridors"]
        cov = self.coverage
        rows = np.where(cov.any(axis=1))[0]; cols = np.where(cov.any(axis=0))[0]
        m = int(np.ceil(c["margin_m"] / self.cell))
        y_lo, y_hi = (rows.min() + m + self.r0) * self.cell, (rows.max() - m + self.r0) * self.cell
        x_lo, x_hi = (cols.min() + m + self.c0) * self.cell, (cols.max() - m + self.c0) * self.cell
        yc, xc = (y_lo + y_hi) / 2, (x_lo + x_hi) / 2
        d = c["offset_m"]
        self.corridors = [
            Corridor("EW-north", x_lo, yc + d, x_hi, yc + d, c["alt_ew"]),
            Corridor("EW-south", x_hi, yc - d, x_lo, yc - d, c["alt_ew"]),
            Corridor("NS-east", xc + d, y_lo, xc + d, y_hi, c["alt_ns"]),
            Corridor("NS-west", xc - d, y_hi, xc - d, y_lo, c["alt_ns"]),
        ]
        self.menu = DEFAULT_MENU
        self.tau_h = np.array([cor.length / self.cfg["kinematics"]["V"] / 3600.0 for cor in self.corridors])
        self.log("corridors: " + ", ".join(f"{k.name} {k.length:.0f} m" for k in self.corridors))

    # ---- weather --------------------------------------------------------
    def _weather(self):
        p = self.processed
        real, self.meta_real = wx.read_open_meteo_csv(p / "wind_era5_reanalysis.csv")
        fc, self.meta_fc = wx.read_open_meteo_csv(p / "wind_dayahead_previous_runs_ecmwf_ifs025.csv")
        lead = int(self.cfg.get("context_lead_days", 1))       # 1: issued 24 h before valid time; 2: 48 h
        if lead not in (1, 2):
            raise ValueError("context_lead_days must be 1 or 2")
        fcol, dcol = f"wind_speed_100m_previous_day{lead}", f"wind_direction_100m_previous_day{lead}"
        if fcol not in fc:
            raise ValueError(f"{fcol} not in the forecast file; context_lead_days={lead} unavailable")
        self.context_lead_days = lead
        days, self.wqc = wx.build_days(real, fc, self.hours, fspd_col=fcol, fdir_col=dcol)
        self.W, self.D, self.F, self.FD = days["W"], days["D"], days["F"], days["FD"]
        sr = p / "wind_historical_forecast_ecmwf_ifs025.csv"
        if sr.exists():
            s, _ = wx.read_open_meteo_csv(sr)
            self.SR = wx.daily_matrix(s["wind_speed_100m"], self.hours).reindex(self.W.index)
            self.SRD = wx.daily_matrix(s["wind_direction_100m"], self.hours).reindex(self.W.index)
        else:
            self.SR = None
        self.X = wx.context_features(self.F, self.FD, self.blocks)
        tr0, tr1 = pd.Timestamp(self.cfg["train"][0]), pd.Timestamp(self.cfg["train"][1])
        te0, te1 = pd.Timestamp(self.cfg["test"][0]), pd.Timestamp(self.cfg["test"][1])
        idx = self.W.index
        self.train = idx[(idx >= tr0) & (idx <= tr1)]
        self.test = idx[(idx >= te0) & (idx <= te1)]
        Xt = self.X.loc[self.train]
        self.x_mu, self.x_sd = Xt.mean(), Xt.std().replace(0, 1)
        self.skill_train = wx.forecast_skill(self.F.loc[self.train], self.W.loc[self.train])
        self.skill_test = wx.forecast_skill(self.F.loc[self.test], self.W.loc[self.test])
        # random-forest weights: predict realised block means from context (training only)
        Y = np.column_stack([self.W.loc[self.train, [h for h in self.hours if a <= h <= b]].mean(axis=1)
                             for (a, b) in self.blocks])
        self.rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10, random_state=0, n_jobs=1)
        self.rf.fit(((Xt - self.x_mu) / self.x_sd).values, Y)
        self.log(f"weather: {len(idx)} complete days; train {len(self.train)}, test {len(self.test)}; "
                 f"day-ahead skill (test) MAE {self.skill_test['mae_ms']:.2f} m/s, r {self.skill_test['corr']:.2f}")

    # ---- demand (synthetic) ------------------------------------------
    def _demand(self):
        d = self.cfg["demand"]
        self.Lam = np.full(len(self.corridors), float(d["peak_per_corridor"]))
        self.phi = d["planned_share"]

    def P(self, scale=1.0):
        return planned(self.hours, self.Lam * scale, self.phi)

    def O(self, date, scale=1.0):
        return on_demand(int(date.strftime("%Y%m%d")), self.hours, self.Lam * scale, self.phi)

    # ---- risk tables ----------------------------------------------------
    def build_tables(self, nav_name: str, gamma: float, n_mc: int):
        # the consequence tables depend on the exposure layers: key the cache on their content so
        # that a cache built from other inputs is never reused
        import hashlib
        h = hashlib.sha256()
        for arr in [self.ped_unit] + [self.rho_road[k] for k in sorted(self.rho_road)]:
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        h.update(repr((self.cfg["platform"], self.cfg["ps"], self.cfg["harm"], self.cfg["seed"])).encode())
        key = f"C_tables_n{n_mc}_{h.hexdigest()[:12]}.pkl"
        f = self.cache / key
        heading_alt = sorted({(L["heading"], L["alt"]) for cor in self.corridors for cfg in self.menu
                              for L in (lanes(cor, cfg) if cfg.n_lanes else [])})
        if f.exists():
            z = pickle.load(open(f, "rb"))
            self.Ccls, self.Cmax, self.kernel_stats = z["Ccls"], z["Cmax"], z["stats"]
        else:
            rng = np.random.default_rng(self.cfg["seed"])
            plat = self.cfg["platform"]; ps = self.cfg["ps"]
            t0 = time.time()
            kern, stats = tb.harm_kernels(rng, plat, heading_alt, self.cell,
                                          {"open": ps["open"], "vehicle": ps["vehicle"], "heavy": ps["heavy"]},
                                          n_mc, (self.cfg["harm"]["alpha_J"], self.cfg["harm"]["beta_J"]),
                                          self.cfg["harm"]["r_uas"])
            self.kernel_stats = stats
            rho_unit = {"ped": ("open", self.ped_unit), "moto": ("open", self.rho_road["moto"]),
                        "vehicle": ("vehicle", self.rho_road["vehicle"]), "heavy": ("heavy", self.rho_road["heavy"])}
            # convolution maps per kernel and class, then sample lane points
            Ccls = {j: [] for j in CLASSES}; Cmax = []
            conv_cache = {}
            for (h, a, s, di), K in kern.items():
                for j, (kcls, rho) in rho_unit.items():
                    conv_cache[(h, a, s, di, j)] = correlate(rho, K[kcls], mode="same", method="direct")
            for k, cor in enumerate(self.corridors):
                rowj = {j: [] for j in CLASSES}; rowm = []
                for cfg in self.menu:
                    ln = lanes(cor, cfg) if cfg.n_lanes else []
                    tabs = {j: np.zeros((13, 12, len(ln))) for j in CLASSES}
                    tmax = np.zeros((13, 12, len(ln)))
                    for li, L in enumerate(ln):
                        xs, ys = lane_points(cor, L["offset"], self.cell)
                        rr = (np.floor(ys / self.cell) - self.r0).astype(int); cc = (np.floor(xs / self.cell) - self.c0).astype(int)
                        for s in range(13):
                            for di in range(12):
                                tot = 0
                                for j in CLASSES:
                                    v = conv_cache[(L["heading"], L["alt"], s, di, j)][rr, cc]
                                    tabs[j][s, di, li] = v.mean(); tot = tot + v
                                tmax[s, di, li] = np.max(tot)
                    for j in CLASSES:
                        rowj[j].append(tabs[j])
                    rowm.append(tmax)
                for j in CLASSES:
                    Ccls[j].append(rowj[j])
                Cmax.append(rowm)
            self.Ccls, self.Cmax = Ccls, Cmax
            pickle.dump({"Ccls": Ccls, "Cmax": Cmax, "stats": stats}, open(f, "wb"))
            self.log(f"harm kernels + consequence tables: {time.time()-t0:.0f} s")
        self.ktab(nav_name, gamma)
        self.nav_name, self.gamma = nav_name, gamma

    def ktab(self, nav_name, gamma):
        if not hasattr(self, "Ktabs"):
            self.Ktabs = {}
        key = (nav_name, float(gamma))
        if key not in self.Ktabs:
            kf = self.cache / f"K_{nav_name}_g{float(gamma)}.pkl"
            if kf.exists():
                self.Ktabs[key] = pickle.load(open(kf, "rb"))
            else:
                kin = nm.Kinematics(**self.cfg["kinematics"])
                self.Ktabs[key] = tb.kappa_tables(self.corridors, self.menu, nm.MODELS[nav_name], kin, gamma)
                pickle.dump(self.Ktabs[key], open(kf, "wb"))
        return self.Ktabs[key]

    def class_scales(self, ped_density, offpeak_scale):
        """(T, classes) multipliers: pedestrians = assumed density; road classes = observed
        08:00-11:00 mean, scaled by offpeak_scale outside the observed window."""
        S = np.ones((len(self.hours), len(CLASSES)))
        for t, h in enumerate(self.hours):
            road = 1.0 if 8 <= h <= 10 else offpeak_scale
            S[t] = [ped_density, road, road, road]
        return S

    # ---- instance --------------------------------------------------------
    def instance(self, days, weights, O_list, par, eps=None, menu_mask=None, max_switches=None):
        T, Wn = len(self.hours), len(days)
        S = self.class_scales(par["ped_density"], par["offpeak_scale"])
        sb = np.zeros((Wn, T), int); db = np.zeros((Wn, T), int); us = np.zeros((Wn, T), bool)
        for w, (spd, fdir) in enumerate(days):
            sb[w], db[w], us[w] = tb.bins(np.asarray(spd, float), np.asarray(fdir, float))
        cap = 3600.0 * self.cfg["kinematics"]["V"]
        Ktab = self.ktab(par["nav_model"], par.get("gamma", self.cfg["gamma"]))
        configs = []
        for k in range(len(self.corridors)):
            row = []
            for c, cfg in enumerate(self.menu):
                if menu_mask is not None and not menu_mask[c]:
                    continue
                n = cfg.n_lanes
                U = np.zeros((T, Wn, n)); C = np.zeros((T, Wn, n)); KA = np.zeros((T, Wn, n, n))
                if n:
                    sbT, dbT = sb.T, db.T                                   # (T, W)
                    C = sum(S[:, j][:, None, None] * self.Ccls[cl][k][c][sbT, dbT] for j, cl in enumerate(CLASSES))
                    KA = Ktab[k][c][sbT]                                    # (T, W, n, n)
                    U = np.where(us.T[:, :, None], cap / cfg.S_x, 0.0) * np.ones((1, 1, n))
                row.append(ConfigData(n, cfg.fixed_cost, U, C, KA, cfg.name))
            configs.append(row)
        ins = Instance(configs=configs, O=np.stack(O_list, axis=2), P=self.P(par["demand_scale"]),
                        p=np.asarray(weights, float), block_of_t=self.block_of_t, tau_h=self.tau_h,
                        value=par["value"], cancel_penalty=par["cancel_penalty"], switch_cost=par["switch_cost"],
                        max_switches=par["max_switches"] if max_switches is None else max_switches,
                        theta_G=par["theta_G"], theta_A=par["theta_A"], lambda_f=par["lambda_f"],
                        mitigation=par["mitigation"], eps=eps)
        ins.sb, ins.db, ins.us = sb, db, us
        ins.cmap = [c for c in range(len(self.menu)) if menu_mask is None or menu_mask[c]]
        ins.hclass = np.array([1 if 8 <= h <= 10 else 0 for h in self.hours])
        ins.parkey = (par["nav_model"], float(par.get("gamma", self.cfg["gamma"])), par["ped_density"], par["offpeak_scale"],
                      par["lambda_f"], par["theta_G"], par["theta_A"], par["mitigation"])
        return ins

    def cap_key(self, ins):
        return lambda k, c, t, w: (ins.parkey, k, ins.cmap[c], int(ins.sb[w, t]), int(ins.db[w, t]),
                                   bool(ins.us[w, t]), int(ins.hclass[t]))

    # ---- scenario sets for each policy -----------------------------------
    def day(self, d, source="realised"):
        if source == "realised":
            return (self.W.loc[d].values, self.D.loc[d].values)
        if source == "forecast":
            fd = self.FD.loc[d].values
            # no leakage: if the forecast direction is missing, use the training climatology
            clim = wx.circular_mean_deg(self.D.loc[self.train].values, axis=0)
            return (self.F.loc[d].values, np.where(np.isnan(fd), clim, fd))
        if source == "shortrange":
            # Missing short-range hours fall back to the day-ahead forecast (information that
            # is also available at admission time); the number of filled hours is logged.
            fs, fd = self.day(d, "forecast")
            sr, srd = self.SR.loc[d].values.astype(float), self.SRD.loc[d].values.astype(float)
            miss = np.isnan(sr) | np.isnan(srd)
            self.sr_filled_hours = getattr(self, "sr_filled_hours", 0) + int(miss.sum())
            return (np.where(miss, fs, sr), np.where(miss, fd, srd))

    def scenarios(self, policy, d, n, par):
        tr = self.train
        rng = np.random.default_rng([self.cfg["seed"], int(d.strftime("%Y%m%d"))])
        xs = ((self.X.loc[[d]] - self.x_mu) / self.x_sd).values[0]
        Xtr = ((self.X.loc[tr] - self.x_mu) / self.x_sd).values
        if policy in ("SAA", "Static"):
            idx = rng.choice(len(tr), size=min(n, len(tr)), replace=False); w = np.full(len(idx), 1 / len(idx))
        elif policy == "SeasonalSAA":
            ws = cx.seasonal_weights(tr.dayofyear.values, d.dayofyear, self.cfg["seasonal_window"])
            cand = np.flatnonzero(ws > 0)
            idx = rng.choice(cand, size=min(n, len(cand)), replace=False); w = np.full(len(idx), 1 / len(idx))
        elif policy == "kNN":
            ww = cx.knn_weights(Xtr, xs, n); idx, w = cx.reduce_scenarios(ww, n)
        elif policy == "RF":
            ww = cx.rf_weights(self.rf, Xtr, xs); idx, w = cx.reduce_scenarios(ww, n)
        elif policy == "Deterministic":
            self.last_scen = {"dates": [f"forecast:{d.date()}"], "weights": [1.0]}
            return [self.day(d, "forecast")], [1.0], [self.O_mean(par)]
        elif policy == "WaitAndSee":
            self.last_scen = {"dates": [f"realised:{d.date()}"], "weights": [1.0]}
            return [self.day(d, "realised")], [1.0], [self.O(d, par["demand_scale"])]
        elif policy == "Robust90":
            ww = cx.knn_weights(Xtr, xs, n); idx, _ = cx.reduce_scenarios(ww, n)
            Wn = np.nanpercentile(self.W.loc[tr[idx]].values, 90, axis=0)
            Dn = wx.circular_mean_deg(self.D.loc[tr[idx]].values, axis=0)
            self.last_scen = {"dates": ["p90-of:" + ",".join(str(x.date()) for x in tr[idx])], "weights": [1.0]}
            return [(Wn, Dn)], [1.0], [self.O_mean(par)]
        else:
            raise ValueError(policy)
        dates = tr[idx]
        self.last_scen = {"dates": [str(x.date()) for x in dates], "weights": [round(float(v), 6) for v in w]}
        return ([self.day(x) for x in dates], list(w), [self.O(x, par["demand_scale"]) for x in dates])

    def O_mean(self, par):
        return (1 - self.phi) * np.outer(self.Lam * par["demand_scale"], profile(self.hours))


# ------------------------------------------------------------------ run one policy-day
def first_stage(st: Study, policy, d, par, n, eps=None, menu_mask=None, max_switches=None, time_limit=300,
                force_milp=False, soc_time_limit=60):
    days, w, O = st.scenarios(policy, d, n, par)
    ms = 0 if policy == "Static" else max_switches
    ins = st.instance(days, w, O, par, eps=eps, menu_mask=menu_mask, max_switches=ms)
    ins.scen = getattr(st, "last_scen", None)
    if eps is None and not force_milp:
        r = solve_fast(ins, CAP, st.cap_key(ins))
    elif eps is None and force_milp == "extensive":
        r = solve(ins, time_limit=time_limit, mip_rel_gap=par.get("mip_gap", 1e-4))
    elif eps is None:
        # block-form MILP (configuration binaries; one capacity row per block), solved by HiGHS
        r = solve_societal(ins, time_limit=time_limit, mip_rel_gap=par.get("mip_gap", 1e-4),
                           cache=CAP, cap_key=st.cap_key(ins))
    elif eps is not None and not force_milp:
        # strict open-lane model (feasible, independently checked) plus its intra-hour
        # time-sharing relaxation, whose optimum bound certifies the strict solution's gap
        r = solve_societal(ins, time_limit=soc_time_limit, mip_rel_gap=par.get("mip_gap", 1e-4),
                           cache=CAP, cap_key=st.cap_key(ins))
        rts = solve_societal(ins, time_limit=2 * soc_time_limit, mip_rel_gap=par.get("mip_gap", 1e-4),
                             cache=CAP, cap_key=st.cap_key(ins), time_sharing=True)
        bound = min(getattr(r, "dual_bound", np.inf), rts.dual_bound)
        r.ts_obj, r.ts_status, r.ts_runtime_s = rts.obj, rts.status, rts.runtime_s
        r.cert_bound = bound
        r.cert_gap = (bound - r.obj) / max(1.0, abs(bound)) if np.isfinite(r.obj) else np.nan
    else:
        r = solve(ins, time_limit=time_limit, mip_rel_gap=par.get("mip_gap", 1e-4))
    return ins, r


def evaluate(st: Study, d, z, q, par, eps=None, menu_mask=None, source="realised", risk_source=None,
             risk_par=None, max_switches=None):
    """Fix (z, q); re-optimise hour-wise admissions on the realised day.  Optionally admit
    with information from `source` (e.g. short-range forecast) but measure risk with the
    realised wind (`risk_source`) and/or another parameter set (`risk_par`)."""
    O = [st.O(d, par["demand_scale"])]
    ins = st.instance([st.day(d, source)], [1.0], O, par, eps=eps, menu_mask=menu_mask, max_switches=max_switches)
    zz = {k: float(v) for k, v in z.items()}
    if eps is None:
        r = solve_fast(ins, CAP, st.cap_key(ins), fixed_z=zz, fixed_q=q)
    else:
        r = solve_societal(ins, fixed_z=zz, fixed_q=q, cache=CAP, cap_key=st.cap_key(ins))
    out = {"status": r.status, "obj": r.obj, "residual": r.residual_max, "eval_runtime_s": r.runtime_s}
    if r.status != 0:
        return out, r
    ins_r = ins
    if risk_source is not None or risk_par is not None:
        ins_r = st.instance([st.day(d, risk_source or source)], [1.0], O, risk_par or par, eps=eps,
                            menu_mask=menu_mask, max_switches=max_switches)
    rk = recompute_risk(ins_r, r)
    flown = sum(r.y.values()) + sum(r.o.values())
    out.update(flown=flown, planned_flown=sum(r.y.values()), ondemand_flown=sum(r.o.values()),
               preapproved=sum(q.values()), cancelled=sum(q.values()) - sum(r.y.values()),
               max_ground_ratio=rk["max_ground_ratio"], max_air_ratio=rk["max_air_ratio"],
               exp_fatalities_day=float(rk["F_per_hour"].sum()),
               exceed_hours_ground=0, exceed_hours_air=0)
    if ins_r is not ins:
        # count hours in which realised per-flight-hour risk exceeds a target
        eh_g = eh_a = 0
        mf = 1.0 - ins_r.mitigation
        for t in range(ins_r.T):
            worst_g = worst_a = 0.0
            for k, cfgs in enumerate(ins_r.configs):
                for c, cd in enumerate(cfgs):
                    x = np.array([r.x.get((k, c, l, t, 0), 0.0) for l in range(cd.n_lanes)])
                    if x.size == 0 or x.max() <= 1e-9:
                        continue
                    rr = mf * cd.kappa[t, 0] @ x
                    g = (ins_r.lambda_f + rr) * cd.C[t, 0]
                    if cd.U[t, 0].min() <= 0:      # lane flown although realised wind > limit
                        g = np.where(cd.U[t, 0] <= 0, np.inf, g)
                    on = x > 1e-9
                    worst_g = max(worst_g, float((g[on] / ins_r.theta_G).max()))
                    worst_a = max(worst_a, float((rr[on] / ins_r.theta_A).max()))
            eh_g += worst_g > 1 + 1e-9; eh_a += worst_a > 1 + 1e-9
        out.update(exceed_hours_ground=int(eh_g), exceed_hours_air=int(eh_a))
    return out, r


def summarize_first_stage(st: Study, ins: Instance, r):
    plan = {}
    for k, cor in enumerate(st.corridors):
        names = [cd.name for cd in ins.configs[k]]
        plan[cor.name] = "|".join(names[max(range(len(names)), key=lambda c: r.z.get((k, c, b), 0))] for b in range(ins.B))
    sw = sum(1 for k in range(ins.K) for b in range(1, ins.B)
             if max(range(len(ins.configs[k])), key=lambda c: r.z.get((k, c, b), 0)) !=
             max(range(len(ins.configs[k])), key=lambda c: r.z.get((k, c, b - 1), 0)))
    return {"plan": json.dumps(plan), "switches": sw, "q_total": sum(r.q.values()),
            "fs_obj": r.obj, "fs_runtime_s": r.runtime_s, "fs_status": r.status, "fs_gap": r.mip_gap,
            "fs_residual": r.residual_max,
            "fs_cert_gap": getattr(r, "cert_gap", np.nan), "fs_cert_bound": getattr(r, "cert_bound", np.nan),
            "fs_ts_obj": getattr(r, "ts_obj", np.nan), "fs_ts_status": getattr(r, "ts_status", np.nan), "n_vars": r.n_vars, "n_cons": r.n_cons, "n_int": r.n_int,
            "scenarios": json.dumps(getattr(ins, "scen", None))}
