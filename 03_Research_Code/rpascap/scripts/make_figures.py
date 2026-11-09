"""All manuscript figures from results/{tables,processed,raw}  ->  results/figures/*.png|pdf.
Every figure is drawn from saved files only.  Development inputs are watermarked.
"""
from __future__ import annotations
import argparse, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from pathlib import Path

PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
POL_ORDER = ["Static", "Deterministic", "SAA", "SeasonalSAA", "kNN", "RF", "Robust90", "WaitAndSee"]
POL_LABEL = {"Static": "Static", "Deterministic": "Deterministic", "SAA": "SAA", "SeasonalSAA": "Seasonal SAA",
             "kNN": "kNN contextual", "RF": "RF contextual", "Robust90": "Robust (90th pct)", "WaitAndSee": "Wait-and-see"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "figure.dpi": 100})
DEV = False


def save(fig, out, name):
    if DEV:
        fig.text(0.5, 0.5, "DEVELOPMENT INPUTS - NOT RESULTS", ha="center", va="center", rotation=25,
                 fontsize=26, color="#e34948", alpha=0.25, weight="bold")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_framework(out):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); ax.set_xlim(-1, 101); ax.set_ylim(-1.5, 58); ax.axis("off"); ax.grid(False)
    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1", fc=fc, ec=INK2, lw=0.7))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.4, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.0, body, ha="center", va="top", fontsize=6.4, color=INK, linespacing=1.25)
    def arr(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.7, color=INK2))
    c1, c2, c3, c4 = "#e8eef7", "#e9f3ea", "#fdf1e3", "#f2e8f5"
    for x, t in [(12.5, "Observed / official data"), (38.5, "Model inputs"), (64, "Two-stage model"), (88.5, "Out-of-sample test")]:
        ax.text(x, 55.5, t, fontsize=7.4, weight="bold", color=INK2, ha="center")
    box(1, 39, 23, 13, "pNEUMA (Zenodo)", "road-vehicle trajectories\ncentral Athens, 2018\nweekday mornings", c1)
    box(1, 25, 23, 12, "OpenStreetMap", "building footprints\n(open-ground mask)", c1)
    box(1, 11, 23, 12, "Open-Meteo", "ERA5 100 m wind (realised)\nECMWF day-ahead forecast", c1)
    box(1, 0, 23, 9.5, "Tagged assumptions", "harm model, occupancy,\nnavigation-error models", c1)
    box(27, 37, 23, 15, "Exposure layers", "road occupants by class\n+ assumed pedestrians\n(20 m cells)", c2)
    box(27, 19, 23, 16, "Risk coefficients", "harm kernels by wind -> C\nexact Reich overlap -> κ\nlane usable if W ≤ 12 m/s", c2)
    box(27, 1, 23, 16, "Scenarios", "day-ahead context ->\nkNN / random-forest\nweights on past days", c2)
    box(53, 30, 22, 22, "Stage 1 (day-ahead)", "corridor configuration\nper block, switches,\npre-approved flights", c3)
    box(53, 1, 22, 26, "Stage 2 (each hour)", "admissions per lane\nsubject to ground risk\nand collision rate per\naircraft flight hour", c3)
    box(78, 30, 21, 22, "Evaluation", "2026 test days\nfirst stage fixed,\nrecourse re-solved\non realised wind", c4)
    box(78, 1, 21, 26, "Metrics", "flights, cancellations\nrisk residuals\nexpected fatalities\nvalue of context\n(paired bootstrap)", c4)
    for a, b in [((24, 45), (27, 44)), ((24, 31), (27, 42)), ((24, 17), (27, 26)), ((24, 5), (27, 24)), ((24, 15), (27, 10)),
                 ((50, 44), (53, 41)), ((50, 27), (53, 38)), ((50, 9), (53, 38)), ((50, 27), (53, 15)),
                 ((64, 30), (64, 27)), ((75, 41), (78, 41)), ((75, 14), (78, 14)), ((88.5, 30), (88.5, 27))]:
        arr(a, b)
    save(fig, out, "Fig01_framework")


def fig_study_area(out, R, tables):
    z = np.load(R / "processed" / "exposure_grids.npz")
    road = z["road_moto"] + z["road_vehicle"] + z["road_heavy"]
    info = json.load(open(tables / "data_summary.json"))
    cell, r0, c0 = info["cell_m"], info["r0"], info["c0"]
    cor = pd.read_csv(tables / "corridors.csv"); lanes_ = pd.read_csv(tables / "corridor_lanes.csv")
    fig, ax = plt.subplots(figsize=(6.6, 5.6)); ax.grid(False)
    ext = [c0 * cell, (c0 + road.shape[1]) * cell, r0 * cell, (r0 + road.shape[0]) * cell]
    ax.imshow(z["building_frac"], origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=2.2, interpolation="nearest")
    m = np.ma.masked_where(road <= 0, road)
    im = ax.imshow(m, origin="lower", extent=ext, cmap="Blues", norm=matplotlib.colors.LogNorm(vmin=max(m.min(), 1e-4), vmax=m.max()),
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cb.set_label("Road occupants (persons per m², development-set mean)")
    for i, r in cor.iterrows():
        ax.plot([r.x0_m, r.x1_m], [r.y0_m, r.y1_m], color=PAL[1], lw=2.0)
        ax.annotate("", xy=(r.x1_m, r.y1_m), xytext=(r.x1_m - 0.08 * (r.x1_m - r.x0_m), r.y1_m - 0.08 * (r.y1_m - r.y0_m)),
                    arrowprops=dict(arrowstyle="-|>", color=PAL[1], lw=1.5))
        ax.text(r.x0_m, r.y0_m, f" {r.corridor}", fontsize=7, color=INK, va="bottom")
    ax.set_xlabel("Easting from reference point (m)"); ax.set_ylabel("Northing from reference point (m)")
    ax.set_aspect("equal")
    ax.set_title("Study area: observed road-occupant exposure, buildings (grey) and corridor centrelines", fontsize=8.5, loc="left")
    save(fig, out, "Fig02_study_area")


def fig_wind(out, R, tables):
    w = pd.read_csv(R / "processed" / "wind_hourly_operating.csv.gz")
    dw = pd.read_csv(tables / "day_wind.csv"); w = w.merge(dw[["date", "set"]], on="date")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    bins = np.arange(0, 26, 1)
    for i, s in enumerate(["train", "test"]):
        axs[0].hist(w[w.set == s].W_realised_100m, bins=bins, density=True, histtype="step", lw=1.6, color=PAL[i],
                    label=f"{s} ({'2024–25' if s == 'train' else '2026'})")
    axs[0].axvline(12, color=INK2, lw=0.8, ls="--"); axs[0].text(12.3, axs[0].get_ylim()[1] * 0.9, "12 m/s limit", fontsize=7, color=INK2)
    axs[0].set_xlabel("Realised wind at 100 m (m/s)"); axs[0].set_ylabel("Density"); axs[0].legend(fontsize=7)
    t = w[w.set == "test"]
    axs[1].hexbin(t.W_dayahead_100m, t.W_realised_100m, gridsize=35, cmap="Blues", mincnt=1, linewidths=0)
    mx = np.nanmax([t.W_dayahead_100m.max(), t.W_realised_100m.max()])
    axs[1].plot([0, mx], [0, mx], color=INK2, lw=0.8); axs[1].grid(False)
    axs[1].set_xlabel("Day-ahead forecast (m/s)"); axs[1].set_ylabel("Realised (m/s)"); axs[1].set_title("Test hours", fontsize=8, loc="left")
    mon = pd.read_csv(tables / "wind_exceedance_by_month.csv")
    for i, s in enumerate(["train", "test"]):
        q = mon[mon.set == s]
        axs[2].plot(q.month, 100 * q.frac_hours_above_12, marker="o", ms=4, lw=1.6, color=PAL[i], label=s)
    axs[2].set_xlabel("Month"); axs[2].set_ylabel("Operating hours > 12 m/s (%)"); axs[2].set_xticks(range(1, 13, 2)); axs[2].legend(fontsize=7)
    fig.tight_layout(); save(fig, out, "Fig03_wind")


def fig_harm(out, tables):
    L = pd.read_csv(tables / "consequence_table_long.csv.gz")
    q = L[(L.config == "std-3")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cors = q.corridor.unique()
    for i, c in enumerate(cors):
        s = q[(q.corridor == c) & (q.lane == 1) & (q.speed_bin_ms == 8)].sort_values("wind_toward_deg")
        tot = s.C_moto + s.C_vehicle + s.C_heavy
        axs[0].plot(s.wind_toward_deg, tot * 1e3, marker="o", ms=3, lw=1.5, color=PAL[i], label=c)
    axs[0].set_xlabel("Wind blowing toward (deg), 8 m/s"); axs[0].set_ylabel("Road-occupant fatalities per crash (×10⁻³)")
    axs[0].legend(fontsize=6.5, ncol=2); axs[0].set_xticks(range(0, 360, 90))
    for i, c in enumerate(cors):
        s = q[(q.corridor == c) & (q.lane == 1)].groupby("speed_bin_ms").C_ped.mean()
        axs[1].plot(s.index, s.values, marker="o", ms=3, lw=1.5, color=PAL[i], label=c)
    axs[1].set_xlabel("Wind speed bin (m/s)"); axs[1].set_ylabel("Pedestrian term per unit density (m²)")
    fig.tight_layout(); save(fig, out, "Fig04_harm")


def fig_nav(out, tables):
    k = pd.read_csv(tables / "collision_coefficients.csv")
    k = k[k.speed_bin_ms == 5]
    cfgs = ["single", "wide-2", "std-3", "dense-2x2", "dense-3x2"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for j, (col, ttl) in enumerate([("max_uniform_flow_per_lane_thetaA_1e-5", "no mitigation credit"),
                                    ("max_uniform_flow_per_lane_thetaA_1e-5_m0.99", "99 % mitigation credit")]):
        for i, m in enumerate(["E1-precise", "E2-measured", "E3-degraded", "E4-KimBae"]):
            s = k[k.nav_model == m].set_index("config").reindex(cfgs)
            axs[j].plot(range(len(cfgs)), s[col], marker="o", ms=4, lw=1.4, color=PAL[i], label=m)
        axs[j].set_yscale("log"); axs[j].set_xticks(range(len(cfgs))); axs[j].set_xticklabels(cfgs, rotation=30, fontsize=7)
        axs[j].axhline(60, color=INK2, lw=0.8, ls="--"); axs[j].set_title(ttl, fontsize=8, loc="left")
    axs[0].set_ylabel("Max. flow per lane at θ_A = 10⁻⁵ /FH (fl/h)"); axs[1].legend(fontsize=6.5, loc="lower left")
    axs[0].text(0, 75, "peak demand per corridor", fontsize=6.5, color=INK2)
    fig.tight_layout(); save(fig, out, "Fig05_navigation_models")


def fig_main(out, tables):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for j, reg in enumerate(["R1", "R2"]):
        f = tables / f"main_{reg}_paired.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f); p = p[(p.reference == "SeasonalSAA") & (p.metric == "obj")]
        pols = [x for x in ["Static", "Deterministic", "SAA", "kNN", "RF", "Robust90", "WaitAndSee"] if x in set(p.policy)]
        for i, pol in enumerate(pols):
            r = p[p.policy == pol].iloc[0]
            axs[j].errorbar(r.mean_diff, i, xerr=[[r.mean_diff - r.ci_lo], [r.ci_hi - r.mean_diff]], fmt="o", ms=5,
                            color=PAL[POL_ORDER.index(pol)], ecolor=INK2, elinewidth=1, capsize=2)
        axs[j].axvline(0, color=INK2, lw=0.8); axs[j].set_yticks(range(len(pols)))
        axs[j].set_yticklabels([POL_LABEL[x] for x in pols]); axs[j].set_title(f"Regime {reg}", fontsize=8.5, loc="left")
        axs[j].set_xlabel("Objective difference vs seasonal SAA\n(flight-equivalents per day, 95 % CI)")
    fig.tight_layout(); save(fig, out, "Fig06_benchmark_paired")


def fig_windy(out, tables):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for j, reg in enumerate(["R1", "R2"]):
        f = tables / f"main_{reg}_windy_split.csv"
        if not f.exists():
            continue
        p = pd.read_csv(f)
        pols = ["Deterministic", "SAA", "kNN", "RF", "Robust90"]
        for g, off, lab in [(False, -0.15, "calm days"), (True, 0.15, "windy days (any hour > 12 m/s)")]:
            for i, pol in enumerate(pols):
                r = p[(p.policy == pol) & (p.windy_day == g)]
                if r.empty or np.isnan(r.mean_diff_obj_vs_SeasonalSAA.iloc[0]):
                    continue
                r = r.iloc[0]
                axs[j].errorbar(r.mean_diff_obj_vs_SeasonalSAA, i + off,
                                xerr=[[r.mean_diff_obj_vs_SeasonalSAA - r.ci_lo], [r.ci_hi - r.mean_diff_obj_vs_SeasonalSAA]],
                                fmt="o" if g else "s", ms=4.5, color=PAL[1] if g else PAL[0], ecolor=INK2, elinewidth=0.9,
                                capsize=2, label=lab if i == 0 else None)
        axs[j].axvline(0, color=INK2, lw=0.8); axs[j].set_yticks(range(len(pols))); axs[j].set_yticklabels([POL_LABEL[x] for x in pols])
        axs[j].set_title(f"Regime {reg}", fontsize=8.5, loc="left"); axs[j].set_xlabel("Objective difference vs seasonal SAA")
    axs[1].legend(fontsize=6.5, loc="lower right"); fig.tight_layout(); save(fig, out, "Fig07_windy_split")


def fig_regime(out, tables):
    f = tables / "nav_regime_map.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    mods = ["E1-precise", "E2-measured", "E3-degraded", "E4-KimBae"]; ms = sorted(d.mitigation.unique())
    M = np.array([[d[(d.nav_model == m) & (d.mitigation == x)].flown_WaitAndSee.mean() for x in ms] for m in mods])
    fig, ax = plt.subplots(figsize=(4.6, 2.8)); ax.grid(False)
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M), aspect="auto")
    for i in range(len(mods)):
        for j in range(len(ms)):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.6 * np.nanmax(M) else INK)
    ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"m = {x:g}" for x in ms]); ax.set_yticks(range(len(mods))); ax.set_yticklabels(mods)
    cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("Flights per day (wait-and-see)")
    ax.set_title("Capacity by navigation-error model and mitigation credit", fontsize=8.5, loc="left")
    save(fig, out, "Fig08_regime_map")


def fig_frontier(out, tables):
    f = tables / "societal_frontier.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, pol in enumerate(["RF", "WaitAndSee"]):
        s = d[d.policy == pol].sort_values("F_day")
        ax.plot(s.F_day * 1e6, s.flown, marker="o", ms=4, lw=1.6, color=PAL[[5, 7][i]], label=POL_LABEL[pol])
    ax.set_xlabel("Expected third-party fatalities per day (×10⁻⁶)"); ax.set_ylabel("Flights flown per day")
    ax.legend(fontsize=7); ax.set_title("Societal-risk frontier (regime R2)", fontsize=8.5, loc="left")
    save(fig, out, "Fig09_societal_frontier")


def fig_dyn(out, tables):
    f = tables / "dynamic_configuration.csv"
    if not f.exists():
        return
    d = pd.read_csv(f); d = d[d.policy == "RF"]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=False)
    for j, reg in enumerate(["R1", "R2"]):
        for i, sc in enumerate([0.0, 5.0, 20.0]):
            pts = []
            for M in range(4):
                r = d[d.setting == f"{reg}|M={M}|sigma={sc}"]
                if not r.empty:
                    pts.append((M, r.obj.iloc[0]))
            if pts:
                x, y = zip(*pts); axs[j].plot(x, y, marker="o", ms=4, lw=1.5, color=PAL[i], label=f"switch cost {sc:g}")
        axs[j].set_xlabel("Maximum switches per corridor-day"); axs[j].set_title(f"Regime {reg}", fontsize=8.5, loc="left")
        axs[j].set_xticks(range(4))
    axs[0].set_ylabel("Mean objective (RF policy)"); axs[1].legend(fontsize=6.5)
    fig.tight_layout(); save(fig, out, "Fig10_dynamic_configuration")


def fig_sens(out, tables):
    f = tables / "sensitivity.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    d = d.set_index("setting"); order = ["base"] + [s for s in d.index if s != "base"]
    d = d.loc[order]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), sharey=True)
    y = np.arange(len(d))
    axs[0].errorbar(d.vci, y, xerr=[d.vci - d.vci_lo, d.vci_hi - d.vci], fmt="o", ms=4, color=PAL[5], ecolor=INK2, elinewidth=0.9, capsize=2)
    axs[0].axvline(0, color=INK2, lw=0.8); axs[0].set_xlabel("Value of context: RF − seasonal SAA\n(objective per day, 95 % CI)")
    axs[1].plot(d.flown_WaitAndSee, y, "o", ms=4, color=PAL[7], label="wait-and-see")
    axs[1].plot(d.flown_RF, y, "s", ms=4, color=PAL[5], label="RF contextual")
    axs[1].set_xlabel("Flights flown per day"); axs[1].legend(fontsize=6.5)
    axs[0].set_yticks(y); axs[0].set_yticklabels(d.index, fontsize=7); axs[0].invert_yaxis()
    fig.tight_layout(); save(fig, out, "Fig11_sensitivity")


def fig_comp(out, tables):
    f = tables / "computation_scaling.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    for i, (m, lab) in enumerate([("milp", "block-form MILP (HiGHS)"), ("fast", "structured algorithm (Prop. 4)"),
                                  ("extensive", "extensive form, lane indicators (time limit)")]):
        s_ = d[d.method == m].sort_values("n_scen")
        if len(s_):
            ax.errorbar(s_.n_scen, s_.runtime, yerr=[np.zeros(len(s_)), s_.runtime_max - s_.runtime], marker="o", ms=4, lw=1.6,
                        color=PAL[i], ecolor=INK2, capsize=2, label=lab)
    ax.set_yscale("log"); ax.set_xlabel("Scenarios in the first-stage problem"); ax.set_ylabel("Solve time (s), mean and max")
    ax.legend(fontsize=7)
    save(fig, out, "Fig12_computation")


def main():
    global DEV
    ap = argparse.ArgumentParser(); ap.add_argument("--results", default="results"); a = ap.parse_args()
    R = Path(a.results); T = R / "tables"; out = R / "figures"; out.mkdir(parents=True, exist_ok=True)
    DEV = json.load(open(T / "data_summary.json")).get("dev_synthetic", False)
    for f in [lambda: fig_framework(out), lambda: fig_study_area(out, R, T), lambda: fig_wind(out, R, T), lambda: fig_harm(out, T),
              lambda: fig_nav(out, T), lambda: fig_main(out, T), lambda: fig_windy(out, T), lambda: fig_regime(out, T),
              lambda: fig_frontier(out, T), lambda: fig_dyn(out, T), lambda: fig_sens(out, T), lambda: fig_comp(out, T)]:
        try:
            f()
        except Exception as e:
            print("figure failed:", repr(e))
    print(sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
