# Credit Risk Modeling Pipeline

A production-style, modular machine-learning pipeline that estimates the **probability
that a loan applicant will default**. Built for clarity, reproducibility, and
interpretability — engineering quality first, accuracy second.

> **Business framing.** Lenders lose money approving applicants who default and lose
> revenue rejecting good ones. This pipeline estimates probability of default so approval
> decisions can be optimized to minimize losses.

---

## Status

Built in reviewable stages (see `CLAUDE.md` §5). Current progress:

- [x] **Stage A** — project scaffold, config, data loader, Stage-1 data-quality report
- [x] **Stage B** — EDA notebook + findings (`notebooks/01_eda.ipynb`, `reports/eda_findings.md`)
- [x] **Stage C** — reusable preprocessing pipeline (leakage-safe): cleaning + impute + missing-indicators + one-hot + stratified split
- [x] **Stage D** — feature engineering: 6 business ratios (credit/income, annuity/income, credit term, employment/age, income-per-person, goods/credit)
- [x] **Stage E** — model development: LogReg / RandomForest / XGBoost, bounded tuning, **XGBoost selected** (val ROC-AUC 0.763); artifact saved
- [x] **Stage F** — evaluation on untouched test (ROC-AUC **0.764**, PR-AUC 0.251), curves, confusion matrix, FN/FP business-cost threshold analysis
- [x] **Stage G** — interpretation: TreeSHAP importance/beeswarm/dependence + plain-language drivers (`reports/interpretation.md`)
- [x] **Stage H** — reproducibility: inference script (`inference/predict.py`) + end-to-end runner (`run_pipeline.py`)
- [x] **Polish** — probability **calibration** (isotonic, OOF; `models/calibrate.py`) + a **pytest** suite (`tests/`)

**Phase 1 complete.** Phase 2 (deferred): the 8 relational side-tables and a deeper
protected-attribute fairness review.

## Dataset

**Home Credit Default Risk.** Phase 1 uses the single main application table
(`data/raw/application_train.csv`, 307,511 × 122). The 8 relational side-tables are
deferred to Phase 2.

- **Target:** `TARGET` — `1 = default`, `0 = repaid` (already in this convention; verified,
  not re-mapped). Default rate ≈ 8.1% (imbalance ≈ 11.4 : 1).
- See [`reports/data_quality.md`](reports/data_quality.md) for the full Stage-1 report.

The raw CSVs are **not** committed (large). Obtain them from the Home Credit Default Risk
dataset and place them under `data/raw/`.

## Project structure

```
config/         central YAML config + loader (paths, seed, split ratios, model params)
preprocessing/  data loading, cleaning, reusable leakage-safe preprocessing pipeline
features/       business-relevant feature engineering
models/         training + tuning, probability calibration, scoring helper, artifacts/
evaluation/     metrics, plots, business-cost analysis, SHAP interpretation
inference/      load saved pipeline + model, score new rows (never re-fits)
tests/          pytest suite (leakage-safety, pipeline shape, calibration, inference)
notebooks/      EDA only (exploration, not the source of truth)
reports/        generated reports, business insights, figures
run_pipeline.py single end-to-end entry point
```

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Unix: source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Running

```bash
# Full pipeline: data overview -> train+select -> evaluate -> SHAP -> sample inference
python run_pipeline.py

# Reuse the saved model, just re-run evaluation + interpretation + demo (fast)
python run_pipeline.py --skip-train

# Score new applicants (raw columns like application_train.csv, TARGET optional)
python -m inference.predict --input new_apps.csv --output scored.csv
python -m inference.predict          # demo on a few rows from the raw data
```

Individual stages are also runnable: `python -m models.train`,
`python -m models.calibrate`, `python -m evaluation.evaluate`,
`python -m evaluation.interpret`. EDA lives in `notebooks/01_eda.ipynb`.

Tests: `python -m pytest` (from the repo root).

## Results (Phase 1)

Test set = 15% held out, scored once. Final model **XGBoost** (selected over Logistic
Regression and Random Forest on validation ROC-AUC).

| Metric (untouched test) | Value |
|---|---|
| ROC-AUC | **0.763** |
| PR-AUC | 0.244 (vs 0.081 prevalence baseline) |
| Brier (after calibration) | **0.067** (from 0.196 raw) |
| Recall @ cost-optimal threshold (FN:FP=10:1) | 0.65 |

**Top default drivers (SHAP):** external credit scores `EXT_SOURCE_1/2/3` dominate;
two **engineered** ratios (`GOODS_CREDIT_RATIO`, `CREDIT_TERM`) rank #3-#4, above every
other raw feature — evidence the feature engineering added real signal. Full write-ups in
[`reports/`](reports/): `model_selection.md`, `evaluation.md`, `interpretation.md`,
`calibration.md`.

> **Scores are calibrated.** An isotonic map (fit on out-of-fold train+val scores) corrects
> the `scale_pos_weight` inflation, so `probability_default` reads as an actual probability
> (mean ≈ base rate; Brier 0.196 → 0.067) with ranking unchanged. The operating threshold
> is a business decision (see `reports/evaluation.md`).

## Design principles

- **No data leakage.** Every learned transform (imputation, scaling, encoding) is fit on
  the training split only, inside a single scikit-learn `Pipeline` / `ColumnTransformer`,
  and reused unchanged at validation/test/inference time.
- **Reproducible.** One centralized random seed and all tunable parameters in
  `config/config.yaml`; artifacts saved with `joblib`.
- **Config-driven.** No parameters hardcoded across modules.
