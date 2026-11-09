// Results-dependent parts of the audit, checklist and reproducibility guide.
const { P, H1, H2, B, table, CAP } = require("./lib");
const f = (x, d = 3) => (x === null || x === undefined || Number.isNaN(+x)) ? "n/a" : Number(x).toFixed(d);
const fe = x => (x === null || x === undefined || Number.isNaN(+x)) ? "n/a" : Number(x).toExponential(1);

module.exports = (D) => {
  const c = (D.analysis || {}).computation || {};
  const s = D.summary;
  const real = !s.dev_synthetic;
  const audit = [
    H1("6 Results of the verification"),
    table(["Check", "Result"], [
      ["Automated test suite", D.test_summary || "see test_log.txt"],
      ["First-stage solves (all experiments)", `${c.first_stage_solves ?? "n/a"}; status counts ${JSON.stringify(c.status_counts || {})}`],
      ["Evaluation solves", `status counts ${JSON.stringify(c.eval_status_counts || {})}`],
      ["Structured algorithm vs block-form MILP (main experiments)", `${c.milp_checks_main ?? 0} policy-days; max |difference| ${fe(c.milp_checks_max_absdiff)}`],
      ["Structured algorithm vs block-form MILP (computational experiment)", `${c.fast_vs_milp_pairs ?? 0} pairs; max |difference| ${fe(c.fast_vs_milp_max_absdiff)}`],
      ["Big-M extensive form (computational experiment)", `${c.extensive_runs ?? 0} problems; ${c.extensive_optimal ?? 0} proved optimal within ${f(c.extensive_runtime_max_s, 0)} s; MIP gaps ${f(100 * c.extensive_gap_min, 0)}–${f(100 * c.extensive_gap_max, 0)} %`],
      ["Societal frontier (curve formulation)", `largest certified gap ${fe((D.analysis || {}).soc_cert_gap_max)}; largest residual ${fe((D.analysis || {}).soc_fs_residual_max)} (planning), ${fe((D.analysis || {}).soc_eval_residual_max)} (evaluation)`],
      ["Maximum constraint residual (first stage / evaluation)", `${fe(c.fs_residual_max)} / ${fe(c.eval_residual_max)}`],
      ["Maximum recomputed ground ratio (evaluations on realised information)", f(c.max_ground_ratio_all_realised_eval, 4)],
      ["Maximum recomputed collision ratio (evaluations on realised information)", f(c.max_air_ratio_all_realised_eval, 4)],
      ["Exposure layer, fit-and-freeze (r on held-out days / held-out slot)", ["open", "vehicle", "heavy"].map(k => { const a = (D.holdout || []).find(r => r.test_set === "testA" && r.class === k) || {}, b = (D.holdout || []).find(r => r.test_set === "testB" && r.class === k) || {}; return `${k}: ${f(a.corr, 2)} / ${f(b.corr, 2)}`; }).join("; ")],
      ["Exposure layer, leave one date out", (D.lodo || []).map(r => `${r.class}: mean r = ${f(r.corr_mean, 2)}`).join("; ") || "n/a"],
      ["Weather data", `${s.weather_days_complete} complete days (${s.weather_days_incomplete} excluded); day-ahead MAE ${f(s.skill_test?.mae_ms, 2)} m/s in test`],
      ["Inputs", real ? "processed official inputs; checksums in workbook sheet 00" : "DEVELOPMENT (synthetic) inputs – not valid for publication"],
    ], [4200, 5160], { size: 16 }),
    CAP("Table 3 Verification results of the final run"),
    ...(D.review || []),
  ];
  return {
    audit,
    realDataStatus: real ? "Done" : "Open",
    realDataEvidence: real ? "pNEUMA (official archive, MD5 verified), ERA5 and ECMWF forecasts, OSM; demand synthetic" : "official inputs not yet processed",
    abstractStatus: D.abstract_words ? (D.abstract_words >= 150 && D.abstract_words <= 250 ? "Done" : "Open") : "Open",
    abstractEvidence: D.abstract_words ? `${D.abstract_words} words` : "pending",
    lengthEvidence: D.main_pages ? `${D.main_pages} pages in the review build (A4, single column)` : "pending",
    reproLimits: [
      "Step 0 requires internet access and the availability of the external services (Zenodo, Open-Meteo, OpenStreetMap/Overpass). Open-Meteo forecast archives and OpenStreetMap content can change over time; checksums of the inputs used in this study are recorded so that differences can be detected.",
      "The analysis environment used to produce the reported results had no outbound network access; Step 0 was executed by the corresponding author in Google Colab and the resulting processed_inputs.zip was supplied to the analysis.",
      "Raw pNEUMA trajectories are not redistributed (they are available from Zenodo under CC BY 4.0); the processed cell-level vehicle-seconds are included.",
    ],
    checklistExtra: [],
  };
};
