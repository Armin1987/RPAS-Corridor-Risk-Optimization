# Runs the experiments of run_all.sh in stages (prep, main, navA, navB, soc, misc, sensA, sensB, post)
# for environments that limit job length; independent stages can run in parallel processes.
# Usage: bash run_staged.sh <processed_dir> <results_dir> <stage>
set -e
cd "$(dirname "$0")"
P="$1"; R="$2"; ST="$3"; J="--jobs 1"   # one worker per process keeps the shared capacity cache
mkdir -p "$R"
case "$ST" in
  prep)  python -m pytest -q -p no:cacheprovider > "$R/test_log.txt" 2>&1 || true
         python -m scripts.describe_data --processed "$P" --out "$R" ;;
  main)  python -m scripts.run_experiments --processed "$P" --out "$R" $J --only main
         python -m scripts.run_experiments --processed "$P" --out "$R" $J --only main --context-lead 2 --tag _lead2 ;;
  navA)  python -m scripts.run_experiments --processed "$P" --out "$R" $J --only nav --sub-stride 2 --nav-models E1-precise,E2-measured --tag _navA ;;
  navB)  python -m scripts.run_experiments --processed "$P" --out "$R" $J --only nav --sub-stride 2 --nav-models E3-degraded,E4-KimBae --tag _navB ;;
  soc)   python -m scripts.run_experiments --processed "$P" --out "$R" $J --only soc --soc-stride 25 ;;
  misc)  python -m scripts.run_experiments --processed "$P" --out "$R" $J --only dyn,rob,comp --sub-stride 2 ;;
  sensA) python -m scripts.run_experiments --processed "$P" --out "$R" $J --only sens --sub-stride 2 --sens-keys base,lambda_f,theta_G,theta_A,ped_density --tag _sensA ;;
  sensB) python -m scripts.run_experiments --processed "$P" --out "$R" $J --only sens --sub-stride 2 --sens-keys offpeak_scale,demand_scale,cancel_penalty,gamma,mitigation --tag _sensB ;;
  post)  python -m scripts.analyze --results "$R"
         python -m scripts.make_figures --results "$R"
         python -m scripts.export_for_docs --results "$R" --out "$R/doc_data.json"
         python -m scripts.make_workbook --processed "$P" --results "$R" --out "$R/Supplementary_Research_Dataset.xlsx" ;;
esac
echo "STAGE_DONE $ST"
