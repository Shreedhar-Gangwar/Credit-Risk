# Stage B — EDA Findings

**Source:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) (executed).
**Figures:** [`reports/figures/`](figures/).
Exploration only — conclusions here feed the preprocessing/feature/model stages; production
logic lives in the modules, not the notebook.

---

## 1. Target balance — `01_target_balance.png`
Default rate **8.07%** (24,825 default vs 282,686 repaid), imbalance **11.4 : 1**.
→ Headline metric is **ROC-AUC + PR-AUC**, not accuracy; use `class_weight` /
`scale_pos_weight`. A naive "approve everyone" model is 91.9% accurate yet useless.

## 2. Numeric distributions — `02_numeric_distributions.png`, `03_class_separation.png`
- `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY` are **right-skewed with long tails** →
  favors **median** imputation and tree models / scaling for the linear baseline.
- **`EXT_SOURCE_1/2/3` and age show visible class separation**: defaulters skew toward
  **lower external scores** and **younger** ages. These external scores are the clearest
  single signals in the data.

## 3. Default rate across customer groups — `04_default_rate_by_group.png`
Segments with default rate **above** the 8.07% baseline (business risk segments):
- **Education:** *Lower secondary*, *Secondary*, *Incomplete higher* > baseline; *Higher
  education* and *Academic degree* well below (academic ~2%).
- **Income type:** *Working* above baseline; *Commercial associate*, *State servant*,
  *Pensioner* below. (Tiny groups like *Unemployed*/*Maternity leave*, n<100, are filtered
  from the plot but have very high rates in the raw data.)
- **Family status:** *Civil marriage*, *Single / not married*, *Separated* above; *Married*,
  *Widow* below.
- **Contract type:** *Cash loans* > *Revolving loans*.
- **Occupation:** *Low-skill Laborers*, *Drivers*, *Waiters/barmen* above; *Accountants*,
  *Managers*, *High-skill tech* below.
- **Gender:** M modestly above F.
→ These directly motivate keeping education, income type, occupation, contract type as features.

## 4. Correlation with default — `05_target_correlation.png`
Point-biserial correlations (|value|, direction matters):

| Feature | corr | reading |
|---|---|---|
| EXT_SOURCE_3 | **-0.179** | higher external score → less default |
| EXT_SOURCE_2 | **-0.160** | " |
| EXT_SOURCE_1 | **-0.155** | " |
| AGE (yrs) | -0.078 | older → less default |
| DAYS_EMPLOYED / employment | ~0.075 | longer employment → less default |
| REGION_RATING_CLIENT(_W_CITY) | +0.06 | worse-rated region → more default |

Correlations are individually modest (typical for credit risk) — signal is **spread across
many weak features**, which is why ensemble models help.

## 5. Missingness — `06_missingness.png`, `07_missingness_vs_default.png`
- 67/122 columns have gaps; housing block, `OWN_CAR_AGE` (66%), `EXT_SOURCE_1` (56%) worst.
- **Missingness is informative** — default rate present vs missing:
  - `EXT_SOURCE_1` missing → **8.5%** vs present **7.5%**
  - `OWN_CAR_AGE` missing → **8.5%** vs present **7.2%**
  - `EXT_SOURCE_3` missing → **9.3%** vs present **7.8%**
  - `OCCUPATION_TYPE` missing → **6.5%** vs present **8.8%** (inverted — likely pensioners)
→ **Add missing-indicator features in Stage C**; don't impute silently.

## 6. Outliers / anomalies — `08_outliers.png`
- `AMT_INCOME_TOTAL` max **117,000,000** vs p99 **472,500** — extreme tail (a few records).
  → robust imputation + tolerate/cap; trees are unaffected.
- Confirmed Stage-1 sentinels: `DAYS_EMPLOYED == 365243` (18.0%), `CODE_GENDER=='XNA'` (4),
  `NAME_FAMILY_STATUS=='Unknown'` (2) → set to NaN in Stage C cleaning.

## Decisions carried into later stages
| Finding | Action |
|---|---|
| 8% imbalance | ROC-AUC/PR-AUC; class weighting; stratified split & CV |
| EXT_SOURCE strongest but missing | median impute **+ missing-indicators** |
| Missingness carries signal | missing-indicators for high-missing columns |
| Skewed amounts, extreme income | median impute; scale for linear model; trees for tails |
| Risk segments (education/income/occupation) | keep as encoded categoricals |
| Weak-but-many signals | ensemble models (RF, XGBoost) alongside logistic baseline |
| Amount relationships | Stage D ratios: credit/income, annuity/income, credit term, etc. |
