# STAR-2: Iterative machine-learning framework for early prediction of chronic post-surgical pain in adolescents

Code accompanying the manuscript **"Predicting chronic post-surgical pain as early as possible: a patient-specific, machine-learning framework"** (STAR-2).

## Overview

This repository contains the full analysis pipeline used to predict chronic post-surgical pain (CPSP) at six months in adolescents undergoing posterior spinal fusion for idiopathic scoliosis. The framework integrates 14 preoperative clinical and psychosocial variables with a 30-day daily postoperative pain diary, re-estimates each patient's risk daily, and commits to a confident classification at the earliest day the model reaches a fold-optimized confidence threshold.

Primary result: Random Forest reached a repeated nested cross-validated test AUROC of **0.741** (95% CI 0.722–0.760) on a cohort of N = 144 adolescents, with 51% of patients receiving a confident early decision within seven postoperative days.

## Repository structure

```
star2-cpsp-ml/
├── pipeline/          Main modelling pipelines
│   ├── nested_cv_balanced.py         Baseline + Diary nested CV (primary result)
│   ├── nested_cv_diary_only.py       Diary-Only nested CV (BD vs DO comparison)
│   ├── RF_fit.py                     Full-data RF refit + SHAP (BD)
│   └── RF_fit_diary_only.py          Full-data RF refit + SHAP (DO)
│
├── analysis/          Downstream statistical analyses
│   ├── analysis_paired_comparison.py       Wilcoxon signed-rank BD vs DO
│   ├── analysis_auroc_vs_day.py            Day-by-day AUROC (BD)
│   ├── analysis_auroc_vs_day_diary_only.py Day-by-day AUROC (DO)
│   ├── analysis_pvalue.py                  Model-vs-model p-values (BD)
│   ├── analysis_pvalue_diary_only.py       Model-vs-model p-values (DO)
│   ├── analysis_shap_waterfall.py          Per-patient SHAP waterfalls (BD)
│   └── analysis_shap_waterfall_diary_only.py
│
└── figures/           Figure generation
    ├── make_study_timeline.py         Study timeline
    ├── make_pipeline_schematic.py     Nested CV pipeline schematic
    ├── make_figure2_combined.py       ROC / PR / Calibration
    ├── make_figure5_combined.py       Day-by-day AUROC
    ├── make_figure6_combined.py       SHAP beeswarm + bar
    ├── regenerate_shap_labels.py      SHAP figures with human-readable labels
    ├── regenerate_figures.py          General figure regeneration
    └── replot_auroc_vs_day_ci.py      Day-by-day AUROC with CI bands
```

## Requirements

Python 3.10+ with the packages listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Data availability

The STAR cohort data are not included in this repository. Deidentified individual participant data are available upon reasonable request to the corresponding author of the manuscript, subject to appropriate data-use agreements. The pipeline expects two CSV files:

- `STAR_T1_T3_T4_measures_QST_numeric.csv` — preoperative and outcome measures
- `T2_deid_diary_data_numeric.csv` — 30-day daily diary items

Paths are configured at the top of each pipeline script.

## Reproducing the main analysis

```bash
# 1. Nested cross-validation (Baseline + Diary and Diary-Only)
python pipeline/nested_cv_balanced.py
python pipeline/nested_cv_diary_only.py

# 2. Full-data refit + SHAP
python pipeline/RF_fit.py
python pipeline/RF_fit_diary_only.py

# 3. Downstream analyses
python analysis/analysis_paired_comparison.py
python analysis/analysis_auroc_vs_day.py
python analysis/analysis_shap_waterfall.py

# 4. Figures
python figures/make_figure2_combined.py
python figures/make_figure5_combined.py
python figures/make_figure6_combined.py
python figures/regenerate_shap_labels.py
```

## Method highlights

- **Repeated nested cross-validation** (Repeated Stratified 5-fold × 10 repeats = 50 outer folds) with recursive feature elimination embedded inside the inner loop, yielding an unbiased internal generalization estimate.
- **Item-level daily diary aggregation** into six statistics per item (last, mean, std, min, max, OLS slope), recomputed at each decision day d ∈ {1, ..., 30}.
- **Patient-specific early-decision rule** with jointly optimized positive/negative confidence thresholds (41 × 41 grid) minimizing a weighted misclassification cost on a held-out development set within each outer fold.
- **SHAP-based interpretation** on the final Random Forest refitted on the full cohort.
