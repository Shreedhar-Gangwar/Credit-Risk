# CLAUDE.md — Credit Risk Modeling Pipeline

> This file gives you (Claude Code) persistent context for this project. Read it fully at the start of every session before doing anything.

---

## 1. What this project is

Build a **production-style credit risk modeling pipeline** that predicts the probability that a loan applicant will **default**.

This is **not** a Kaggle-style notebook. The whole point is to demonstrate clean engineering: modular code, reproducibility, a reusable preprocessing pipeline, business-oriented analysis, and interpretable results. **Model accuracy is secondary to engineering quality and clarity.** The end goal is a portfolio-quality project suitable for Data Scientist / ML Engineer interviews, where I will be asked to *defend every design decision*.

### Business framing
Lenders lose money approving applicants who default, and lose revenue rejecting good applicants. The pipeline should estimate probability of default so approval decisions can be optimized. It should help answer:
- Which applicants are high risk?
- Which customer characteristics drive default?
- How can approval decisions be optimized to minimize losses?

---

## 2. Dataset — **I will fill this in; confirm it before building**

> **FILL IN before your first build task.** I am placing my own dataset in `data/raw/`.

- **File(s) / location:** `data/raw/<FILENAME>`  ← _(update this)_
- **Single table or multiple tables?** `<single | multiple>` _(if multiple, list them and the join keys)_
- **Target column:** `<COLUMN NAME>`
- **Target meaning:** `<which value = default, which = repaid>`
- **Data dictionary:** `<paste here, or "none — infer from column names">`

### IMPORTANT — your very first task
Before writing any pipeline code, **inspect the actual dataset on disk** and report back:
1. Shape, schema, dtypes.
2. Missing values per column.
3. Class balance of the target.
4. A short data-quality summary.
5. **Confirm the target encoding and re-map it to the spec convention: `1 = default`, `0 = repaid`.** Tell me what mapping you applied.

Do not assume the dataset's contents from this file — read it and confirm with me.

---

## 3. How I want you to work (read this carefully)

This is the most important section. I am using this project to learn, and I have to be able to explain it in interviews.

- **Plan before building.** Use plan mode. Propose your approach and let me approve it before you write code.
- **Work in stages, not all at once.** Build one pipeline stage at a time (see Section 5). After each stage, stop so I can review the diff before moving on. Do **not** generate the whole project in one shot.
- **Explain the rationale** for non-obvious choices in a sentence or two as you go — especially feature engineering, leakage prevention, and metric selection. Each engineered feature must have a clear stated reason.
- **Show me, don't just tell.** I review each file change as a diff and approve it.
- **Ask if the dataset is ambiguous** rather than guessing schema or semantics.
- Keep me honest: if I ask for something that would hurt the project (e.g. introduce data leakage, optimize the wrong metric), push back and explain.

---

## 4. Required project structure

Build the project as a **modular Python application**, not a single notebook:

```
credit-risk-pipeline/
├── data/
│   ├── raw/            # original dataset (input, never modified)
│   └── processed/      # generated; gitignored
├── config/             # config files (paths, params, random seeds)
├── preprocessing/      # reusable preprocessing pipeline (must work at inference)
├── features/           # feature engineering
├── models/             # training, tuning, saved model artifacts
├── evaluation/         # metrics, plots, reports
├── inference/          # load saved pipeline + model, predict on new data
├── notebooks/          # EDA only (exploration, not the source of truth)
├── reports/            # generated reports, business insights, figures
├── README.md
├── requirements.txt
└── run_pipeline.py     # runs the whole thing end to end
```

The entire pipeline must be **runnable start-to-finish with minimal manual steps** (a single entry point like `run_pipeline.py` or a clear `make`/CLI command).

---

## 5. Pipeline stages (build in this order, one at a time)

1. **Data understanding** — load, schema, dtypes, missing values, class imbalance, data-quality report.
2. **EDA** (in `notebooks/`) — distributions, relationships with default, correlations, outliers, default rates across customer groups. Visuals must support *business understanding*, not just exist.
3. **Preprocessing** — a **reusable** pipeline: missing-value treatment, categorical encoding, scaling where appropriate, outlier handling, train/validation/test split. **Prevent data leakage** (fit transforms on training data only; the same fitted pipeline must be reusable at inference).
4. **Feature engineering** — business-relevant features *where the dataset supports them*, e.g. debt-to-income ratio, credit utilization, payment-behavior indicators, credit-history length, income ratios, loan-burden measures. Each feature needs a clear rationale. Only build features the columns actually support — confirm with me if unsure.
5. **Model development** — baselines + advanced: Logistic Regression, Random Forest, XGBoost. Cross-validation for comparison. Hyperparameter tuning for the tree-based models. Final model chosen on **both** performance and interpretability.
6. **Model evaluation** — metrics for **imbalanced** binary classification: ROC-AUC, precision, recall, F1, confusion matrix, precision-recall curve, ROC curve. Discuss the **business cost of false positives vs false negatives**.
7. **Model interpretation** — feature importance and **SHAP** (preferred). Translate the important drivers into plain-language business interpretation a non-technical stakeholder could follow.
8. **Reproducibility** — fixed random seeds everywhere, config-driven parameters, saved model + preprocessing artifacts, and the end-to-end runner.

---

## 6. Deliverables

- Clean, modular source code
- Reusable preprocessing pipeline (works at inference)
- Trained model artifacts (saved to disk)
- Evaluation reports (metrics + plots)
- Business insights write-up
- Visualizations
- `README.md` (how to set up and run)
- `requirements.txt`
- Example **inference script** that loads the saved pipeline + model and predicts on new/unseen rows

---

## 7. Engineering conventions

- **Python**, with `pandas`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`/`seaborn`.
- Prefer `scikit-learn` `Pipeline` / `ColumnTransformer` so preprocessing is fit once and reused at inference — this is the core anti-leakage mechanism.
- **Reproducibility:** set and centralize a random seed; keep tunable parameters in `config/`, not hardcoded across files.
- Save artifacts (fitted preprocessing pipeline + model) with `joblib`; the inference script loads these — it must never re-fit on new data.
- Keep `data/processed/` and model artifacts out of git (`.gitignore`); keep `data/raw/` reproducible (document the source in the README).
- Write a `requirements.txt` and a clear `README.md` so the project runs on a clean machine.
- Notebooks are for exploration only — production logic lives in the modules, not the notebooks.

---

## 8. Success criteria

The finished project should demonstrate: strong software-engineering practices, a proper ML workflow, reproducibility, business understanding, clear documentation, and production-style organization — **not** maximal accuracy. Optimize for a project I can confidently walk an interviewer through end to end.
