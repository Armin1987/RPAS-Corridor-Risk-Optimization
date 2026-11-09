#!/usr/bin/env bash
# Reproduce the complete study from the processed official inputs.
# Usage: bash run_all.sh <processed_inputs_dir> [results_dir]
# Step 0 (once, needs internet): run notebooks/01_acquire_official_data_colab.ipynb in Colab
#         -> processed_inputs.zip ; unzip it and pass the folder here.
set -euo pipefail
PROC=${1:?processed inputs folder}
RES=${2:-results}
python -m pytest -q -p no:cacheprovider | tee "$RES.test_log.txt"
python -m scripts.describe_data --processed "$PROC" --out "$RES"
python -m scripts.run_experiments --processed "$PROC" --out "$RES" --sub-stride 2 --soc-stride 25
# strict day-ahead check: context = forecast predicted 48 h before valid time
python -m scripts.run_experiments --processed "$PROC" --out "$RES" --only main --context-lead 2 --tag _lead2
python -m scripts.analyze --results "$RES"
python -m scripts.make_figures --results "$RES"
python -m scripts.export_for_docs --results "$RES" --out "$RES/doc_data.json"
python -m scripts.make_workbook --processed "$PROC" --results "$RES" --out "$RES/Supplementary_Research_Dataset.xlsx"
echo "done: $RES"
