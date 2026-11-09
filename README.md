
# RPAS Corridor Risk Optimization

## Research Code, Computational Results, and Reproducibility Materials

This repository provides the Python implementation, model configurations, computational experiments, figures, and supporting datasets associated with a research manuscript prepared for *OR Spectrum*.

The study develops a risk-constrained optimization framework for Remotely Piloted Aircraft Systems (RPAS) corridor configuration and admission decisions. It examines the relationship between corridor capacity, operational uncertainty, and safety risk using mathematical optimization and contextual scenario analysis.

The repository is intended to support research transparency, computational reproducibility, and independent examination of the proposed methodology.

## 1. Research Framework

The computational framework integrates:

- Risk-constrained RPAS corridor configuration and admission optimization.
- Air collision risk assessment using Reich-based models and navigation-error distributions.
- Ground risk assessment using aircraft descent, impact exposure, and consequence models.
- Contextual stochastic scenario generation and uncertainty analysis.
- Two-stage mixed-integer linear programming (MILP).
- Out-of-sample evaluation, sensitivity analysis, and robustness assessment.

The optimization implementation uses SciPy's MILP interface with the HiGHS solver.

## 2. Repository Structure

The repository is organized as follows:

| Directory or File | Description |
|---|---|
| `03_Research_Code/rpascap/` | Main Python research implementation |
| `03_Research_Code/rpascap/risk/` | Air and ground risk calculations |
| `03_Research_Code/rpascap/model/` | Optimization models |
| `03_Research_Code/rpascap/scenarios/` | Contextual scenario generation |
| `03_Research_Code/rpascap/data/` | Data loading and processing modules |
| `03_Research_Code/rpascap/configs/` | Model parameters and experimental configurations |
| `03_Research_Code/rpascap/scripts/` | Data preparation, experiments, analysis, and output generation |
| `03_Research_Code/rpascap/tests/` | Software tests and validation checks |
| `04_Figures/figures/` | Manuscript figures in PDF and PNG formats |
| `05_Results_and_Tables/results/` | Computational results and supporting research outputs |
| `05_Results_and_Tables/results/tables/` | Numerical tables, diagnostics, and sensitivity results |
| `05_Results_and_Tables/results/processed/` | Processed research data |
| `05_Results_and_Tables/results/processed_inputs_manifest/` | Data acquisition and provenance records |
| `Supplementary_Research_Dataset.xlsx` | Supplementary research workbook |

## 3. Data Sources

### 3.1 pNEUMA Traffic Dataset

The study uses the publicly available pNEUMA traffic trajectory dataset for the Athens urban road network.

Official dataset:

https://doi.org/10.5281/zenodo.10491409

Original project:

https://open-traffic.epfl.ch/

Relevant publication:

Barmpounakis, E., and Geroliminis, N. (2020).

https://doi.org/10.1016/j.trc.2019.11.023

The original pNEUMA archive is not redistributed through this repository. The dataset should be obtained from its official source in accordance with the applicable terms.

The observed traffic trajectories provide information about road traffic and exposure conditions. They should not be interpreted as observations of actual RPAS delivery demand.

### 3.2 Meteorological Data

Meteorological inputs are derived from Open-Meteo services, including historical reanalysis and forecast information.

https://open-meteo.com/

The research considers wind conditions and forecast uncertainty in the computational analysis.

### 3.3 Geographic Data

Geographic information, including building footprints, is derived from OpenStreetMap.

https://www.openstreetmap.org/

OpenStreetMap data are subject to their applicable attribution and licensing requirements.

### 3.4 Aircraft Parameters

Selected aircraft characteristics are based on published DJI Matrice 300 RTK specifications.

Parameters not established through direct measurement are treated as modelling assumptions.

## 4. Mathematical and Computational Methods

### 4.1 Air Risk Assessment

The air risk component includes Reich-based collision modelling, navigation-error distributions, and overlap calculations.

The implementation includes numerical verification procedures and comparisons between analytical and numerical calculations.

### 4.2 Ground Risk Assessment

The ground risk component considers aircraft descent, wind effects, impact location, population exposure, and potential consequences.

Risk calculations distinguish modelling assumptions from directly observed data.

### 4.3 Stochastic Optimization

The optimization framework incorporates uncertainty through scenario-based methods.

The implementation includes sample average approximation, contextual scenario weighting, and a two-stage MILP formulation.

### 4.4 Computational Evaluation

The research materials include scripts and outputs for evaluating corridor configurations, operational policies, sensitivity to assumptions, and robustness under different scenarios.

## 5. Installation and Software Requirements

The implementation is written primarily in Python.

To prepare the computational environment, clone the repository:

```bash
git clone https://github.com/Armin1987/RPAS-Corridor-Risk-Optimization.git
cd RPAS-Corridor-Risk-Optimization/03_Research_Code/rpascap
```

Install the required Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The project requirements file specifies the Python packages used by the implementation.

Some computational experiments may require additional source datasets and substantial processing time.

## 6. Software Testing

From the research code directory, execute:

```bash
python -m pytest -q
```

The repository contains tests covering risk calculations, data processing, scenario generation, optimization behaviour, and numerical consistency.

The presence of tests and recorded computational outputs does not, by itself, constitute independent verification of every manuscript result.

## 7. Reproducing Computational Results

The principal research scripts are located in:

`03_Research_Code/rpascap/scripts/`

Relevant scripts include:

- `acquire_and_process.py`: Data acquisition and preparation.
- `run_experiments.py`: Computational experiment execution.
- `analyze.py`: Analysis of computational results.
- `make_figures.py`: Figure generation.
- `make_workbook.py`: Supplementary workbook generation.
- `export_for_docs.py`: Export of research outputs for documentation.

Model and study settings are located in:

`03_Research_Code/rpascap/configs/`

The code package also contains workflow scripts:

- `run_all.sh`
- `run_staged.sh`

Full execution may require obtaining the original datasets, preparing the expected input directories, and reviewing configuration paths before running the workflow.

The repository includes computational results and derived data products to support examination of the reported analyses.

## 8. Figures and Numerical Results

The manuscript figures are available in:

`04_Figures/figures/`

Computational results and numerical tables are available in:

`05_Results_and_Tables/results/`

The supplementary workbook is available as:

`Supplementary_Research_Dataset.xlsx`

These materials provide supporting outputs for the computational study, including scenario comparisons, corridor characteristics, risk analysis, and sensitivity assessments.

## 9. Assumptions and Interpretation

The computational framework contains parameters based on published information, empirical inputs, and explicitly specified modelling assumptions.

In particular, navigation-error behaviour, selected consequence-model parameters, occupancy assumptions, and sheltering effects require careful interpretation.

The sensitivity of collision risk to navigation-error distribution tails is an important consideration.

The reported results represent computational evaluations under the specified assumptions and scenarios. They should not be interpreted as direct evidence of operational RPAS safety certification or regulatory approval.

## 10. Research Reproducibility and Data Availability

This repository provides the research implementation and supporting computational materials.

The original third-party datasets should be obtained from their official sources. Processed data and provenance information are provided where available.

Independent reproduction of the complete computational workflow may require additional data preparation and environment configuration.

## 11. Citation

If using this repository, please cite the corresponding *OR Spectrum* manuscript once its final bibliographic information is available.

A version-specific archived software citation and DOI may be added following the creation of an official repository release.

## 12. Licensing and Attribution

Third-party datasets remain subject to their respective licenses and attribution requirements.

No general software reuse license is granted by this README. A separate repository license may be added after confirming the applicable ownership and distribution permissions.

## 13. Repository Information

Repository:

https://github.com/Armin1987/RPAS-Corridor-Risk-Optimization

Research area: Risk-aware RPAS corridor optimization and computational analysis.

Associated journal: *OR Spectrum*.
