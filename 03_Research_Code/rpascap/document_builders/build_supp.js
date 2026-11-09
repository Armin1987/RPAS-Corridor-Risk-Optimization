// Build 02_Supplementary_Materials.docx.  Usage: node build_supp.js <results_dir> <out.docx>
const fs = require("fs"), path = require("path");
const L = require("./lib");
const { d, P, H1, H2, table, CAP, styles, numbering, FONT, image } = L;
const { Document, Packer, Paragraph, TextRun, Footer, Header, PageNumber, AlignmentType } = d;
const refs = require("./refs");
const RES = process.argv[2], OUT = process.argv[3];
const D = JSON.parse(fs.readFileSync(path.join(RES, "doc_data.json")));
D.fig = n => path.join(RES, "figures", n + ".png");
require("./build_order_refs")(D);
const f = (x, k = 3) => (x === null || x === undefined || Number.isNaN(+x)) ? "–" : (Math.abs(+x) < 1e-3 && +x !== 0 ? Number(x).toExponential(2) : Number(x).toFixed(k));
const T = n => D.tables[n] || [];
const tbl = (rows, cols, widths, size = 14) => table(cols.map(c => c[1]), rows.map(r => cols.map(c => typeof r[c[0]] === "number" ? f(r[c[0]]) : String(r[c[0]] ?? "–"))), widths, { size });

const ext = [
  H1("S8 Extended data and results"),
  H2("S8.1 pNEUMA inventory"),
  tbl(T("pneuma_inventory_by_date"), Object.keys((T("pneuma_inventory_by_date")[0]) || {}).map(k => [k, k]), Object.keys((T("pneuma_inventory_by_date")[0]) || {}).map(() => Math.floor(9300 / Math.max(1, Object.keys(T("pneuma_inventory_by_date")[0] || {}).length)))),
  CAP("Table S8.1 Processed pNEUMA files and vehicle tracks by date"),
  tbl(T("pneuma_vehicles_by_type"), [["vtype", "Vehicle type"], ["mean_vehicles_present_in_study_area", "Mean vehicles present (development set, 08:30–09:30)"]], [4000, 5300]),
  CAP("Table S8.2 Mean number of vehicles present in the processed area by type"),
  H2("S8.2 Validation of the exposure layer on held-out data"),
  tbl(T("exposure_holdout_validation"), [["test_set", "Held-out set"], ["class", "Class"], ["n_cells", "Cells"], ["corr", "Correlation"], ["mae_per_m2", "MAE (persons/m²)"], ["mean_obs_per_m2", "Mean observed"], ["rel_error_total", "Relative error of total"]], [1500, 1100, 900, 1300, 1600, 1500, 1400]),
  CAP("Table S8.3a Fit-and-freeze validation: layer fitted on the development set (29–30 Oct 2018, 08:30–09:30) against held-out days (testA: 24 Oct, 1 Nov, same slots) and held-out time (testB: 29–30 Oct, 10:00–10:30)"),
  tbl(T("exposure_leave_one_date_out"), [["held_out_date", "Held-out date"], ["class", "Class"], ["n_cells", "Cells"], ["corr", "Correlation"], ["mae_per_m2", "MAE (persons/m²)"], ["mean_obs_per_m2", "Mean observed"], ["rel_error_total", "Relative error of total"]], [1500, 1100, 900, 1300, 1600, 1500, 1400]),
  CAP("Table S8.3b Leave-one-date-out check over all processed files (each date predicted by the mean of the other dates)"),
  H2("S8.3 Descent kernels"),
  P("Summary statistics of the Monte Carlo descent kernels (impact energy, reach, lethal area) by heading, altitude and wind are in worksheet 08c_Descent_Kernels; Fig. S8.1 shows the dependence of the per-crash consequence on wind direction and speed."),
  image(D.fig("Fig04_harm"), 2100, 780, 6.2),
  CAP("Fig. S8.1 Expected road-occupant fatalities per crash by wind direction (8 m/s, centre lane of configuration std-3) and pedestrian term per unit density by wind speed"),
  H2("S8.4 Navigation regime map (full)"),
  tbl(T("nav_regime_map"), [["nav_model", "Model"], ["mitigation", "m"], ["days", "Days"], ["flown_WaitAndSee", "Flown (WS)"], ["flown_RF", "Flown (RF)"], ["flown_SeasonalSAA", "Flown (seasonal SAA)"], ["vci_obj", "Value of context"], ["vci_lo", "CI low"], ["vci_hi", "CI high"]], [1300, 600, 700, 1100, 1100, 1300, 1100, 1000, 1100]),
  CAP("Table S8.4 Regime map"),
  H2("S8.4b Paired differences: block and independent-day bootstrap"),
  P("Mean paired differences against the reference policy, with 95 % intervals from the circular moving-block bootstrap (block length 7 days; used in the main text) and from the bootstrap that treats days as independent."),
  ...["R1", "R2"].flatMap(reg => [
    tbl(T(`main_${reg}_paired`), [["policy", "Policy"], ["reference", "Reference"], ["metric", "Metric"], ["mean_diff", "Mean diff."], ["ci_lo", "Block CI low"], ["ci_hi", "Block CI high"], ["ci_lo_iid", "Indep. CI low"], ["ci_hi_iid", "Indep. CI high"], ["n_days", "Days"]], [1200, 1200, 900, 1000, 1000, 1000, 1000, 1000, 700]),
    CAP(`Table S8.4b-${reg} Paired differences, regime ${reg}`)]),
  H2("S8.5 Robustness (full)"),
  tbl(T("robustness"), [["setting", "Test"], ["policy", "Policy"], ["days", "Days"], ["obj", "Objective"], ["flown", "Flown"], ["exceed_h_ground", "Ground exc. h"], ["exceed_h_air", "Coll. exc. h"], ["max_ground", "Max ground"], ["max_air", "Max coll."]], [1800, 1100, 700, 1000, 900, 1000, 1000, 900, 900]),
  CAP("Table S8.5 Robustness tests"),
  H2("S8.6 Weather quality control and exceedance by month"),
  tbl(T("wind_exceedance_by_month"), [["set", "Set"], ["month", "Month"], ["frac_hours_above_12", "Share of operating hours > 12 m/s"]], [2500, 2500, 4300]),
  CAP("Table S8.6 Share of operating hours above the 12 m/s platform limit"),
  H2("S8.7 Corridors and lanes"),
  tbl(T("corridors"), [["corridor", "Corridor"], ["length_m", "Length (m)"], ["bearing_deg", "Bearing (deg)"], ["base_alt_m", "Lower-layer altitude (m)"], ["x0_m", "x₀"], ["y0_m", "y₀"], ["x1_m", "x₁"], ["y1_m", "y₁"]], [1400, 1100, 1100, 1500, 1050, 1050, 1050, 1050]),
  CAP("Table S8.7 Corridor geometry (local frame, metres from the grid reference)"),
];

const verif = [
  H1("S9 Reference verification"),
  P("Each reference was checked in this project on the page listed. Google Scholar, Scopus and Web of Science were not accessed. DOIs are given only where they were displayed on a publisher, DOI-registry, repository or indexing page."),
  table(["#", "Reference", "Link"], refs.list().map(r => [String(r.n), r.text, r.link]), [500, 6100, 2700], { size: 14 }),
];

const doc = new Document({ styles, numbering, sections: [{
  properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1200, bottom: 1200, left: 1250, right: 1250 } } },
  headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Supplementary Materials (Online Resource 1)", size: 16, color: "777777", font: FONT })] })] }) },
  footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT })] })] }) },
  children: [
    new Paragraph({ style: "Title", children: [new TextRun({ text: "Supplementary Materials", font: FONT })] }),
    P("Data-driven corridor configuration and flight admission for urban drone delivery under weather uncertainty and third-party risk constraints. OR Spectrum (submitted). Authors: A. Mahmoodi, S. M. Easa, S. M. Sajadi, M. Arunmozhi. Corresponding author: A. Mahmoodi, Toronto Metropolitan University."),
    ...(D.summary.dev_synthetic ? [P("**DEVELOPMENT BUILD ON SYNTHETIC INPUTS – NOT RESULTS**", { run: { color: "C00000" } })] : []),
    ...require("./supp_core")(D), ...ext, ...verif],
}] });
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); console.log("wrote", OUT); });
