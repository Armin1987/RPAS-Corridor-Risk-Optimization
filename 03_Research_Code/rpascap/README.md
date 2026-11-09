# rpascap — risk-constrained RPAS corridor configuration and admission (Stage 2 prototype)

**Status:** prototype written in Stage 2 to verify that the proposed formulation is
implementable and correct on small instances. **It produces no manuscript results.**
Full experiments are Stage 5–6 and need explicit approval.

## What is implemented and tested (44 tests, `python -m pytest -q`)

| Module | Content | Independent checks |
|---|---|---|
| `risk/reich.py` | Laplace-mixture errors; Reich overlap (closed form) and **exact** overlap; per-aircraft collision coefficients, linear in neighbour flow | quadrature; Monte Carlo; Stage 1 audit values; exact vs approximation |
| `risk/ballistic.py` | drag + wind point-mass descent | vacuum closed form; terminal velocity; wind direction |
| `risk/ground.py` | impact kernel, lethal area, fatality probability, per-crash consequence | normalisation; linearity in density; zero limits; monotonicity |
| `risk/normalization.py` | per-flight-hour vs lane-hour identities (Little's law) | identities; implied-budget demonstration |
| `data/pneuma.py` | reader for the official pNEUMA layout; layout detection; time-weighted vehicle-seconds; road-occupant exposure; drone-overlap handling | synthetic files in the official layout (25 and 23.97 fps; swapped columns; non-Athens rejection) |
| `data/download.py` | Zenodo download with md5 `4e842b2200bea2f80d41c6b785a9de25`; Open-Meteo day-ahead and realised wind | not executed in the authoring workspace (no outbound network) |
| `scenarios/contextual.py` | SAA, seasonal SAA, kNN and random-forest weights | weights sum to one; kNN with k=n equals SAA |
| `model/two_stage.py` | extensive-form MILP (SciPy `milp` -> HiGHS); exhaustive-enumeration checker; out-of-sample evaluator | MILP = enumeration on 5 instances; safety recomputed outside the model; monotonicity; switching limits |

## Data sources (official)
* pNEUMA — Zenodo 10.5281/zenodo.10491409 (v1, 2024-01-11, CC BY 4.0) and open-traffic.epfl.ch.
  Cite Barmpounakis & Geroliminis (2020) doi:10.1016/j.trc.2019.11.023, the dataset DOI, and
  "Data source: pNEUMA – open-traffic.epfl.ch".
* Wind — Open-Meteo Previous Runs API (`_previous_day1`, day-ahead) and Historical Forecast API.
* Platform — DJI Matrice 300 RTK official specifications.

## Known limits of the prototype
* Wind→navigation-error slope, harm-model parameters, vehicle occupancy and sheltering are
  tagged ASSUMED in `configs/base.yaml`.
* Air-risk results are extremely sensitive to the navigation-error tail (see Stage 2 report §15).
