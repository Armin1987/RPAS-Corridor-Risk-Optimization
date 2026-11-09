# RPAS Corridor Risk Optimization — research reproducibility package

This repository contains the supplied Python implementation, configurations, scripts, figures, processed research data, and computational results associated with a manuscript prepared for submission to *OR Spectrum*.

**Status:** Public-release preparation package. The included files were consolidated from the author-provided archives. A complete independent scientific reproduction and full automated test pass have **not** yet been established. Do not describe this repository as independently verified.

## Repository contents

- `rpascap/`: Python modules for risk, scenario generation and optimization.
- `scripts/`: data acquisition, analysis, experiment execution, table and figure generation.
- `configs/`: study parameters and selected pNEUMA inventory.
- `tests/`: software tests.
- `figures/`: manuscript figures (PDF and PNG).
- `results/raw/`: computational outputs.
- `results/tables/`: summary tables and diagnostics.
- `results/processed/`: derived data products.
- `results/processed_inputs_manifest/`: acquisition and provenance metadata.
- `data/Supplementary_Research_Dataset.xlsx`: research data workbook.
- `document_builders/`: manuscript/supplement assembly scripts retained for provenance; not required to view results.

## Setup and testing

From the repository root, in a Python environment:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

See `configs/study.yaml` and scripts in `scripts/` for the study settings and pipeline entry points. **Check paths and input prerequisites before executing the full pipeline.**

## Source data and licensing

The original pNEUMA archive is not included. Cite the official dataset DOI: https://doi.org/10.5281/zenodo.10491409. Weather inputs originate from Open-Meteo and building footprints from OpenStreetMap. Consult `results/processed_inputs_manifest/` and the workbook provenance sheets. Derived data may carry source-specific attribution and reuse obligations. **No software reuse license is granted by this README; confirm ownership and select a license before broad reuse.**

## Reproducibility cautions

The supplied code's original README (preserved as `README_stage2_original.md`) refers to an earlier Stage 2 prototype and is not a current release statement. Model assumptions, navigation-error distributions, and provenance require scrutiny. The figures and results are author-provided; their numerical correspondence to the submitted manuscript has not been independently certified. Do not claim measured RPAS navigation errors or observed drone demand unless supported by original measurements.

## Citation

Once the repository is reviewed and an archived release has a DOI, cite that specific release DOI here and in the manuscript's Code Availability statement.
