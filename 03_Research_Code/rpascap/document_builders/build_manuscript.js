// Build 01_Final_Manuscript.docx from section modules and results/doc_data.json.
// Usage: node build_manuscript.js <results_dir> <out.docx>
const fs = require("fs"), path = require("path");
const L = require("./lib");
const { d, P, H1, H2, styles, numbering, table, CAP, FONT } = L;
const { Document, Packer, Paragraph, TextRun, Footer, Header, PageNumber, AlignmentType, ExternalHyperlink } = d;
const refs = require("./refs");

const RES = process.argv[2] || "results";
const OUT = process.argv[3] || "01_Final_Manuscript.docx";
const D = JSON.parse(fs.readFileSync(path.join(RES, "doc_data.json")));
D.fig = name => path.join(RES, "figures", name + ".png");
D.dev = !!D.summary.dev_synthetic;

const core = require("./ms_core")(D);
// Sections are evaluated in document order so that citation numbers follow first appearance.
const body = [...core.intro(), ...core.lit(), ...core.problem(), ...core.model(), ...require("./ms_data")(D), ...core.method()];
const res = require("./ms_results")(D);

const title = "Data-driven corridor configuration and flight admission for urban drone delivery under weather uncertainty and third-party risk constraints";
const front = [
  new Paragraph({ style: "Title", children: [new TextRun({ text: title, font: FONT })] }),
  ...(D.dev ? [P("**DEVELOPMENT BUILD ON SYNTHETIC INPUTS – NOT A MANUSCRIPT**", { run: { color: "C00000", size: 28 } })] : []),
  P("Armin Mahmoodi¹ · Said M. Easa¹ · Seyed Mojtaba Sajadi² · Manimuthu Arunmozhi³"),
  P("¹ Department of Civil Engineering, Toronto Metropolitan University, Toronto, Canada. ² [affiliation to be confirmed by the author]. ³ [affiliation to be confirmed by the author]."),
  P("Corresponding author: Armin Mahmoodi, [email to be confirmed]. ORCID iDs: [to be added]."),
  P("**Abstract** " + res.abstract),
  P("**Keywords** " + res.keywords.join(" · ")),
];

const decl = [
  H1("Statements and declarations"),
  P("**Funding** [To be confirmed by the authors. The original manuscript reported partial support by an NSERC Discovery Grant; the grant holder and number must be verified before submission.]"),
  P("**Competing interests** [To be confirmed by all authors.]"),
  P("**Author contributions** [To be confirmed by all authors (CRediT). The previous contribution statement must be revised to reflect the reconstructed study.]"),
  P("**Use of large language models** A large language model (Claude, Anthropic) was used to assist with code development, mathematical derivation, data processing, computational experiments and drafting of this manuscript. All analyses were executed and verified as documented in the supplementary audit; the authors take full responsibility for the content."),
  H1("Data availability"),
  P("The pNEUMA dataset is openly available from Zenodo (https://doi.org/10.5281/zenodo.10491409) under CC BY 4.0. ERA5 and ECMWF IFS forecast series were retrieved from Open-Meteo; OpenStreetMap data are available under ODbL. All processed inputs and results used in this paper are provided in the supplementary workbook Supplementary_Research_Dataset.xlsx and in the results archive, together with the scripts (and an equivalent Colab notebook) that reproduce the processed inputs from the original sources."),
  H1("Code availability"),
  P("The complete Python code (rpascap), configuration files, automated tests and the scripts that generate every table and figure are provided as supplementary material and will be deposited in a public repository with a persistent identifier upon acceptance."),
  H1("Acknowledgements"),
  P("Data source: pNEUMA – open-traffic.epfl.ch. Map data © OpenStreetMap contributors."),
];

const sections = [...front, ...body, ...res.results, ...res.implications, ...res.limitations, ...res.conclusions, ...decl];
// references last (after all cite() calls have run)
const refList = [H1("References"), ...refs.list().map(r => new Paragraph({
  spacing: { after: 80 }, indent: { left: 420, hanging: 420 },
  children: [new TextRun({ text: `${r.n}. ${r.text}. `, font: FONT, size: 19 }),
             new ExternalHyperlink({ link: r.link, children: [new TextRun({ text: r.link, style: "Hyperlink", font: FONT, size: 19 })] })] }))];

const doc = new Document({
  creator: "Authors", title, styles, numbering,
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1300, right: 1300 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: D.dev ? "DEVELOPMENT BUILD – NOT RESULTS" : "Manuscript submitted to OR Spectrum (Special Issue: Data-Driven Optimization)", size: 16, color: "777777", font: FONT })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT })] })] }) },
    children: [...sections, ...refList],
  }],
});
Packer.toBuffer(doc).then(b => { fs.writeFileSync(OUT, b); fs.writeFileSync(path.join(path.dirname(OUT), "_refcount.json"), JSON.stringify({ cited: refs.list().length, registry: Object.keys(refs.R).length })); console.log("wrote", OUT, "references:", refs.list().length); });
