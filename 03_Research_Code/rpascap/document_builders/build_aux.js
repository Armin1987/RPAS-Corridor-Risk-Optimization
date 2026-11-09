// Builds 06_Reproducibility_Guide.docx (converted to PDF), 07_Final_Scientific_Audit.docx and
// 08_Submission_Readiness_Checklist.docx.  Usage: node build_aux.js <results_dir> <out_dir>
const fs = require("fs"), path = require("path");
const L = require("./lib");
const { d, P, H1, H2, B, N, table, CAP, styles, numbering, FONT, EQ } = L;
const { Document, Packer, Paragraph, TextRun, Footer, PageNumber, AlignmentType } = d;
const RES = process.argv[2], OUT = process.argv[3];
const D = JSON.parse(fs.readFileSync(path.join(RES, "doc_data.json")));
try {   // abstract length, computed from the same builder as the manuscript
  const absTxt = require("./ms_results")({ ...D, fig: name => path.join(RES, "figures", name + ".png") }).abstract;
  D.abstract_words = absTxt.split(/\s+/).filter(Boolean).length;
} catch (e) { console.log("abstract length not computed:", e.message); }
if (process.env.MAIN_PAGES) D.main_pages = Number(process.env.MAIN_PAGES);
const A = require("./aux_results")(D);    // results-dependent parts (written after the experiments)

function docOf(title, subtitle, children, landscape = false) {
  return new Document({ styles, numbering, sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1200, bottom: 1200, left: 1250, right: 1250 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT })] })] }) },
    children: [new Paragraph({ style: "Title", children: [new TextRun({ text: title, font: FONT })] }), P(subtitle), ...children] }] });
}
async function write(doc, name) { fs.writeFileSync(path.join(OUT, name), await Packer.toBuffer(doc)); console.log("wrote", name); }

const repro = [
  H1("1 What this guide reproduces"),
  P("Every table, figure and number in the manuscript, the Supplementary Materials and the workbook Supplementary_Research_Dataset.xlsx is produced by the code in 03_Research_Code from (i) the processed official inputs and (ii) the configuration file configs/study.yaml. Nothing is entered by hand."),
  H1("2 Requirements"),
  B("Python 3.11 or later; packages and versions in requirements.txt (NumPy, SciPy ≥ 1.9 with HiGHS-based milp, pandas, scikit-learn, matplotlib, PyYAML, openpyxl, joblib, pytest)."),
  B("Node.js with the 'docx' package only if the Word documents are to be rebuilt."),
  B(`Hardware used for the reported run: ${D.run_log ? D.run_log.platform + ", " + D.run_log.cpu_count + " logical CPUs" : "see run_log.json"}. Total experiment time is reported in run_log.json.`),
  H1("3 Step 0 – acquire and process the official data"),
  P("Route used for the reported results (manual downloads, local processing):"),
  N("Download pNEUMA_dataset.zip from Zenodo (https://zenodo.org/records/10491409/files/pNEUMA_dataset.zip?download=1; 15.76 GB) and check its MD5: 4e842b2200bea2f80d41c6b785a9de25. The selective-download form of open-traffic.epfl.ch returned files containing only 'Empty Dataset' when tried on 8 November 2026 and was not used.", "num"),
  N("Download, in a browser, the three Open-Meteo CSVs (ERA5 archive; ECMWF IFS 0.25° Previous Runs with 24 h and 48 h leads; ECMWF IFS historical forecast) for 2024-01-01 to 2026-10-31 at 37.9825° N, 23.73° E, and the OpenStreetMap buildings for the box 37.965–38.000° N, 23.710–23.750° E from the Overpass API. The exact URLs are in scripts/local_processing/README.txt.", "num"),
  N("Run scripts/local_processing/run_parts.py (resumable; processes configs/pneuma_file_list.csv plus 20181024_d4_0830_0900.csv directly from the zip) and then finalize.py, which identifies the wind files by content, converts the Overpass export (overpass_to_geojson.py), records SHA-256 hashes and writes processed_inputs.zip with MANIFEST.json. The same result is obtained in one call with: python scripts/acquire_and_process.py --zip pNEUMA_dataset.zip --file-list configs/pneuma_file_list.csv --add-file 20181024_d4_0830_0900.csv --skip-online, after placing the downloaded wind and building files in the output folder.", "num"),
  N("Compare the checksums with those in worksheet 00_README_Reproducibility to confirm that you reproduced the same inputs.", "num"),
  P("Alternative: notebooks/01_acquire_official_data_colab.ipynb downloads and processes everything in Google Colab; to reproduce the reported design, pass --file-list configs/pneuma_file_list.csv --add-file 20181024_d4_0830_0900.csv. Open-Meteo archives can be revised or extended by the provider; if checksums differ, record the differences."),
  H1("4 Step 1 – run the study"),
  EQ("unzip processed_inputs.zip -d processed_inputs"),
  EQ("bash run_all.sh processed_inputs results"),
  P("run_all.sh executes, in order: the automated tests; describe_data (exposure layers, wind matrices, corridor and coefficient tables); run_experiments (all experiments, with every test day in the main comparison, every second test day in the secondary experiments and every 25th test day in the societal frontier (five settings; curve formulation with certified gaps); raw results in results/raw); a second run_experiments call that repeats the main comparison with the 48-h forecast as context (files tagged _lead2); analyze (paired differences with moving-block bootstrap intervals, block length 7 days, fixed seed 20261107); make_figures; export_for_docs; make_workbook."),
  P("run_staged.sh runs the same experiments in nine stages (prep, main, navA, navB, soc, misc, sensA, sensB, post), which can run as parallel processes, for environments that limit the length of a single job; partial runs append to results/raw/run_log.json, chunked raw files (for example nav_navA.csv, nav_navB.csv) are merged by analyze, and the capacity cache in results/cache is shared between stages (it changes run time only)."),
  H1("5 Mapping from manuscript to files"),
  table(["Manuscript item", "Produced by", "File"], [
    ["Fig. 1", "make_figures.fig_framework", "results/figures/Fig01_framework.*"],
    ["Fig. 2, Table 4, Sect. 5.1–5.2", "describe_data, make_figures", "results/processed/exposure_grids.npz; tables/corridors.csv; tables/exposure_holdout_validation.csv; Fig02_study_area.*"],
    ["Fig. 3, Sect. 5.3", "describe_data", "processed/wind_hourly_operating.csv.gz; tables/day_wind.csv; Fig03_wind.*"],
    ["Fig. 4", "describe_data", "tables/consequence_table_long.csv.gz; Fig04_harm.*"],
    ["Fig. 5, Table 5", "describe_data", "tables/collision_coefficients.csv; tables/nav_models.csv; Fig05_navigation_models.*"],
    ["Table 6", "configuration", "configs/study.yaml; rpascap/study.py (scenarios)"],
    ["Tables 7–8, Figs. 6–7", "run_experiments (main), analyze", "raw/main_R1.csv, raw/main_R2.csv; tables/main_*"],
    ["Fig. 8", "run_experiments (nav), analyze", "raw/nav_navA.csv, raw/nav_navB.csv; tables/nav_regime_map.csv"],
    ["Table 9, Fig. 9", "run_experiments (dyn), analyze", "raw/dyn.csv; tables/dynamic_configuration.csv"],
    ["Table 10, Fig. 10", "run_experiments (soc), analyze", "raw/soc.csv; tables/societal_frontier.csv"],
    ["Table 11, Fig. 11", "run_experiments (sens), analyze", "raw/sens_sensA.csv, raw/sens_sensB.csv; tables/sensitivity.csv"],
    ["Table 12", "run_experiments (rob), analyze", "raw/rob.csv; tables/robustness.csv"],
    ["Table 13", "run_experiments (main, --context-lead 2), analyze", "raw/main_R1_lead2.csv, raw/main_R2_lead2.csv; tables/strict_dayahead.csv"],
    ["Table 14, Fig. 12, Sect. 7.6", "run_experiments (comp), analyze", "raw/comp.csv; tables/computation_scaling.csv; tables/summary.json"],
  ], [2400, 3000, 3960], { size: 16 }),
  H1("6 Determinism"),
  P("Random elements and their seeds: Monte Carlo descent kernels (seed 20261107), random-forest weights (random_state 0), scenario sampling (seed derived from the configuration seed and the test date), synthetic demand (seed derived from the date), bootstrap (seed 20261107). HiGHS is deterministic for a given version and thread count; tiny differences across versions can change ties between equivalent configurations without changing objective values."),
  H1("7 Known limitations of reproducibility"),
  ...A.reproLimits.map(t => B(t)),
];

const checklist = [
  P("Status as of the delivery of this package. 'Done' items were checked in this project; 'Author' items need information or confirmation that only the authors can provide; 'Open' items remain to be completed before submission."),
  table(["Requirement (OR Spectrum / Special Issue)", "Status", "Evidence or action"], [
    ["Practical problem in an SI domain (mobility / last-mile delivery)", "Done", "drone-delivery corridor operations, Athens"],
    ["Data integrated into the OR method (predictions, stochasticity)", "Done", "context-weighted scenarios in a two-stage model; value of context measured out of sample"],
    ["Real-world data used", A.realDataStatus, A.realDataEvidence],
    ["Abstract 150–250 words, no undefined abbreviations", A.abstractStatus, A.abstractEvidence],
    ["4–6 keywords", "Done", "6 keywords"],
    ["Numbered citations in order; full DOI links", "Done", (() => { try { const r = JSON.parse(fs.readFileSync(path.join(OUT, "_refcount.json"))); return `generated by refs.js; ${r.cited} references cited in the manuscript, from a registry of ${r.registry} checked entries (Supp. S9)`; } catch (e) { return "generated by refs.js"; } })()],
    ["References verified (title, authors, year, venue, DOI)", "Done with exceptions", "Supp. S9; Sadana et al. (2025): DOI not displayed on the pages accessed (RePEc link given); Geister et al. (2018), Jung et al. (2016) and Dalamagkidis et al. (2008): no DOI located; grey literature (ICAO WP, DJI, Open-Meteo) cited by URL"],
    ["Statements and Declarations section", "Author", "funding (NSERC grant to be verified), competing interests and CRediT contributions must be confirmed by all authors"],
    ["Data availability statement", "Done", "manuscript; workbook; acquisition notebook"],
    ["Code availability statement", "Done (deposit open)", "code supplied; repository with DOI to be created at acceptance"],
    ["Documented LLM use (journal policy)", "Done – author review needed", "statement included in the declarations"],
    ["Title page with affiliations, ORCID, corresponding author e-mail", "Author", "placeholders in the manuscript"],
    ["Figures: vector or ≥ 300/600 dpi, captions in text, no titles inside figures", "Partly done", "PNG 300 dpi and PDF vector supplied; some figures contain panel titles, which the journal discourages – remove if the editor requests"],
    ["Figures readable in black and white", "Partly done", "markers differ by shape in key figures; colour-only distinctions remain in some line plots"],
    ["Permissions for previously published figures", "Done", "all figures are original; the original manuscript's schematic Figs. 2–3 are not reused"],
    ["Co-author approval of withdrawal of original results and new contribution", "Author", "required before submission"],
    ["Pre-submission inquiry to a guest editor (optional)", "Author", "recommended in Stage 2"],
    ["Main text length", "Done", A.lengthEvidence],
    ["Independent external review of the package", "Open", "planned by the authors"],
  ], [3400, 1600, 4360], { size: 16 }),
  ...A.checklistExtra,
];

(async () => {
  await write(docOf("Reproducibility Guide", "Data-driven corridor configuration and flight admission for urban drone delivery (Athens case)", repro), "06_Reproducibility_Guide.docx");
  await write(docOf("Final Scientific Audit", "Verification record, remaining assumptions and independent review of the reconstructed study", [...require("./audit_core")(D), ...A.audit, ...require("./review_passes")(D.review_extra ? [] : [])]), "07_Final_Scientific_Audit.docx");
  await write(docOf("Submission Readiness Checklist", "OR Spectrum, Special Issue “Data-Driven Optimization” (deadline 31 December 2026)", checklist), "08_Submission_Readiness_Checklist.docx");
})();
