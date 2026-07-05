# Stage G — Model Interpretation (SHAP)

Final model: **xgboost**. Global explanations via **TreeSHAP** (computed natively by XGBoost) on a 5,000-row test sample. mean|SHAP| = importance; the direction column is the sign of the (feature value → SHAP value) correlation.

## Top 15 default-risk drivers

| # | Feature | mean\|SHAP\| | Direction | Plain-language meaning |
|---|---|---|---|---|
| 1 | `EXT_SOURCE_3` | 0.3581 | ↓ risk | External credit score #3 — higher = safer. |
| 2 | `EXT_SOURCE_2` | 0.3347 | ↓ risk | External credit score #2 — a bureau-style risk score; higher = safer. |
| 3 | `GOODS_CREDIT_RATIO` | 0.1286 | ↓ risk | Goods price ÷ credit — collateral coverage / down-payment proxy. |
| 4 | `CREDIT_TERM` | 0.1282 | ↓ risk | Annuity ÷ credit (repayment intensity; proxy for shorter term). |
| 5 | `EXT_SOURCE_1` | 0.1224 | ↓ risk | External credit score #1 — higher = safer. |
| 6 | `DAYS_EMPLOYED` | 0.0874 | ↑ risk | Employment length (negative days) — longer tenure = safer. |
| 7 | `AMT_ANNUITY` | 0.0745 | ↑ risk | Loan instalment amount. |
| 8 | `NAME_EDUCATION_TYPE_Higher education` | 0.0711 | ↓ risk | Has higher education — lowers risk. |
| 9 | `DAYS_BIRTH` | 0.0590 | ↑ risk | Age (stored as negative days) — younger applicants default more. |
| 10 | `CODE_GENDER_F` | 0.0568 | ↓ risk | Gender = Female (see fairness caveat). |
| 11 | `CODE_GENDER_M` | 0.0560 | ↑ risk | Gender = Male (see fairness caveat). |
| 12 | `DAYS_ID_PUBLISH` | 0.0426 | ↑ risk | Days since ID document updated — behavioural/stability signal. |
| 13 | `OWN_CAR_AGE` | 0.0424 | ↑ risk | Age of the applicant's car (missing when no car). |
| 14 | `AMT_GOODS_PRICE` | 0.0408 | ↓ risk | Price of the goods the loan finances. |
| 15 | `NAME_FAMILY_STATUS_Married` | 0.0393 | ↓ risk | Married — associated with lower risk. |

## What drives default (plain language)

1. **External credit scores dominate.** `EXT_SOURCE_3/2/1` are the top signals by a wide margin (mean|SHAP| ~0.12-0.36 vs <0.09 for everything else) — higher scores → lower predicted default. They summarise external bureau information the application form can't.
2. **Engineered loan-structure ratios pay off.** `GOODS_CREDIT_RATIO` (#3) and `CREDIT_TERM` (#4) — both built in Stage D — rank above every raw feature except the external scores. Higher goods-to-credit (more of the loan backed by the purchased good, i.e. a larger down-payment) and higher annuity-to-credit (shorter effective term) both **lower** risk. This directly validates the feature-engineering step.
3. **Age & employment stability.** Shorter employment (`DAYS_EMPLOYED`, #6) and younger applicants (`DAYS_BIRTH`, #9) default more; larger instalments (`AMT_ANNUITY`, #7) raise risk — life/job stability and affordable payments lower it.
4. **Demographics & behaviour.** Higher education (#8) and being married (#15) lower risk; stability signals like `DAYS_ID_PUBLISH` (#12) and `OWN_CAR_AGE` (#13) contribute. `CODE_GENDER` appears (#10-11) — flagged under fairness below.
5. **Missingness carries signal.** `OWN_CAR_AGE` is missing when the applicant has no car, and such flags surface among the drivers — validating the Stage-C decision to flag missing values rather than silently impute.

## Figures

- `figures/14_shap_importance.png` — global importance (mean |SHAP|)
- `figures/15_shap_beeswarm.png` — per-feature impact **and** direction
- `figures/16_shap_dependence.png` — how risk varies across the top drivers

## Caveats

- SHAP shows the **model's** associations, **not causation**.
- Engineered ratios share information with their source amounts, so importance can spread across related features — read them as a group, not in isolation.
- Fairness: `CODE_GENDER` and similar attributes are present; a production system should review protected-attribute usage and disparate impact (Phase-2 item).
