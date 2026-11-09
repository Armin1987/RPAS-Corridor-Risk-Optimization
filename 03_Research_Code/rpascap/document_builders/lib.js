// Shared helpers for the Stage 2 Word documents (docx-js)
const d = require("docx");
const { Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, WidthType,
        ShadingType, AlignmentType, BorderStyle, ImageRun, PageBreak, LevelFormat } = d;

const FONT = "Calibri";
const MONO = "Cambria Math";

function runs(text, base = {}) {
  // **bold** markup only
  const out = [];
  const parts = String(text).split(/(\*\*[^*]+\*\*)/g).filter(s => s.length);
  for (const p of parts) {
    if (p.startsWith("**") && p.endsWith("**")) out.push(new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: p, font: FONT, ...base }));
  }
  return out;
}
const P = (t, o = {}) => new Paragraph({ children: runs(t, o.run || {}), spacing: { after: 120, line: 276 }, alignment: o.align, ...(o.p || {}) });
const H1 = t => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t, font: FONT })] });
const H2 = t => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t, font: FONT })] });
const H3 = t => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t, font: FONT })] });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60, line: 264 } });
const N = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } });
const EQ = (t, label) => new Paragraph({
  children: [new TextRun({ text: t, font: MONO, size: 21 }), ...(label ? [new TextRun({ text: "\t" + label, font: FONT, size: 21 })] : [])],
  indent: { left: 567 }, spacing: { before: 60, after: 60 },
  tabStops: [{ type: "right", position: 9000 }]
});
const BR = () => new Paragraph({ children: [new PageBreak()] });
const NOTE = t => new Paragraph({ children: runs(t, { italics: true, size: 19, color: "444444" }), spacing: { after: 160 } });

function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "9AA4B2" };
  const borders = { top: border, bottom: border, left: border, right: border };
  const fs = opts.size || 18;
  const mk = (txt, i, head) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: "E7ECF3", color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 80, right: 80 },
    children: String(txt).split("\n").map(line => new Paragraph({ children: runs(line, { size: fs, bold: head || undefined }), spacing: { after: 20 } }))
  });
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => mk(h, i, true)) }),
           ...rows.map(r => new TableRow({ children: r.map((c, i) => mk(c, i, false)) }))]
  });
}
const CAP = t => new Paragraph({ children: runs(t, { size: 19, bold: false }), spacing: { before: 80, after: 200 } });

function image(path, wPx, hPx, widthIn) {
  const fs = require("fs");
  if (!fs.existsSync(path)) return new Paragraph({ children: runs(`[FIGURE MISSING: ${require("path").basename(path)}]`, { color: "C00000", bold: true }) });
  try {   // use the true pixel size of the file for the aspect ratio
    const b = fs.readFileSync(path); if (b.readUInt32BE(12) === 0x49484452) { wPx = b.readUInt32BE(16); hPx = b.readUInt32BE(20); }
  } catch (e) {}
  const w = widthIn * 96, h = w * hPx / wPx;
  return new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: "png", data: fs.readFileSync(path), transformation: { width: w, height: h } })] });
}

const styles = {
  default: { document: { run: { font: FONT, size: 21 } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 30, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 320, after: 140 }, keepNext: true, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 25, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
    { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
      run: { font: FONT, size: 22, bold: true, color: "333333" }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    { id: "Title", name: "Title", basedOn: "Normal", next: "Normal",
      run: { font: FONT, size: 40, bold: true, color: "1F3A5F" }, paragraph: { spacing: { before: 0, after: 200 } } },
  ]
};
const numbering = { config: [
  { reference: "bul", levels: [
    { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } },
    { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 900, hanging: 260 } } } }] },
  ...["num", "num2", "num3", "num4", "num5", "num6"].map(r => ({ reference: r, levels: [
    { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 300 } } } }] }))
] };

module.exports = { d, P, H1, H2, H3, B, N, EQ, BR, NOTE, table, CAP, image, styles, numbering, runs, FONT };
