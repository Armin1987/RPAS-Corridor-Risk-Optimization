#!/usr/bin/env bash
# Assemble the final research package.  Usage: bash assemble_package.sh <results_dir> <processed_dir> <out_dir>
set -euo pipefail
RES=$(realpath "$1"); PROC=$(realpath "$2"); OUT=$(realpath -m "$3")
CODE=/home/claude/rpascap; FIN=/home/claude/final
SK=/root/.claude/skills/synced/de35b8d1-8af1-4e19-bc05-511a01e26e2a_ff41b461-97bf-49a6-a21b-e1a7f68f7ad3/docx
mkdir -p "$OUT"; cd "$FIN"
node build_manuscript.js "$RES" "$OUT/01_Final_Manuscript.docx"
node build_supp.js "$RES" "$OUT/02_Supplementary_Materials.docx"
( cd "$OUT" && python3 $SK/scripts/office/soffice.py --headless --convert-to pdf 01_Final_Manuscript.docx >/dev/null 2>&1 || true )
export MAIN_PAGES=$(pdfinfo "$OUT/01_Final_Manuscript.pdf" 2>/dev/null | awk '/^Pages/{print $2}')
node build_aux.js "$RES" "$OUT"
for f in "$OUT"/0[1-8]*.docx; do zip -d "$f" '*/' >/dev/null 2>&1 || true; done
( cd "$OUT" && python3 $SK/scripts/office/soffice.py --headless --convert-to pdf 06_Reproducibility_Guide.docx >/dev/null 2>&1 \
  && python3 $SK/scripts/office/soffice.py --headless --convert-to pdf 01_Final_Manuscript.docx >/dev/null 2>&1 || true )
rm -f "$OUT/06_Reproducibility_Guide.docx.bak" "$OUT/_refcount.json"
# 03 code
TMP=$(mktemp -d); cp -r "$CODE" "$TMP/rpascap"; find "$TMP" -name __pycache__ -prune -exec rm -rf {} +; rm -rf "$TMP/rpascap/.pytest_cache"
mkdir -p "$TMP/rpascap/document_builders"; cp "$FIN"/*.js "$FIN"/assemble_package.sh "$TMP/rpascap/document_builders/"
( cd "$TMP" && rm -f "$OUT/03_Research_Code.zip" && zip -qr "$OUT/03_Research_Code.zip" rpascap )
# 04 figures (+ scripts)
TMP2=$(mktemp -d); mkdir -p "$TMP2/figures"; cp "$RES"/figures/* "$TMP2/figures/"; cp "$CODE/scripts/make_figures.py" "$TMP2/figures/"
( cd "$TMP2" && rm -f "$OUT/04_Figures.zip" && zip -qr "$OUT/04_Figures.zip" figures )
# 05 results
TMP3=$(mktemp -d); mkdir -p "$TMP3/results"; cp -r "$RES"/raw "$RES"/tables "$RES"/processed "$TMP3/results/"; cp "$RES"/doc_data.json "$TMP3/results/"
[ -f "$RES/test_log.txt" ] && cp "$RES/test_log.txt" "$TMP3/results/"
mkdir -p "$TMP3/results/processed_inputs_manifest"; cp "$PROC"/MANIFEST.json "$PROC"/acquisition_log.jsonl "$PROC"/pneuma_file_inventory.csv "$TMP3/results/processed_inputs_manifest/" 2>/dev/null || true
( cd "$TMP3" && rm -f "$OUT/05_Results_and_Tables.zip" && zip -qr "$OUT/05_Results_and_Tables.zip" results )
cp "$RES/Supplementary_Research_Dataset.xlsx" "$OUT/"
# 09 everything
( cd "$OUT" && rm -f 09_Complete_Research_Package.zip && zip -q 09_Complete_Research_Package.zip 01_Final_Manuscript.docx 02_Supplementary_Materials.docx \
   03_Research_Code.zip 04_Figures.zip 05_Results_and_Tables.zip 06_Reproducibility_Guide.pdf 07_Final_Scientific_Audit.docx \
   08_Submission_Readiness_Checklist.docx Supplementary_Research_Dataset.xlsx )
ls -la "$OUT"
