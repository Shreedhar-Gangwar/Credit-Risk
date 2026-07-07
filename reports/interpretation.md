# Stage G — Model Interpretation (SHAP)

Final model: **xgboost**. Global explanations via **TreeSHAP** (computed natively by XGBoost) on a 5,000-row test sample. mean|SHAP| = importance; the direction column is the sign of the (feature value → SHAP value) correlation.

## Top 15 default-risk drivers

| # | Feature | mean\|SHAP\| | Direction | Plain-language meaning |
|---|---|---|---|---|
| 1 | `EXT_SOURCE_2` | 0.3196 | ↓ risk | External credit score #2 — a bureau-style risk score; higher = safer. |
| 2 | `EXT_SOURCE_3` | 0.2861 | ↓ risk | External credit score #3 — higher = safer. |
| 3 | `EXT_SOURCE_1` | 0.1143 | ↓ risk | External credit score #1 — higher = safer. |
| 4 | `GOODS_CREDIT_RATIO` | 0.1074 | ↓ risk | Goods price ÷ credit — collateral coverage / down-payment proxy. |
| 5 | `DAYS_EMPLOYED` | 0.0900 | ↑ risk | Employment length (negative days) — longer tenure = safer. |
| 6 | `CREDIT_TERM` | 0.0846 | ↓ risk | Annuity ÷ credit (repayment intensity; proxy for shorter term). |
| 7 | `PREV_DAYS_LAST_DUE_1ST_VERSION_MAX` | 0.0802 | ↑ risk | Timing of prior loans' scheduled last payment. |
| 8 | `AMT_ANNUITY` | 0.0711 | ↑ risk | Loan instalment amount. |
| 9 | `NAME_EDUCATION_TYPE_Higher education` | 0.0606 | ↓ risk | Has higher education — lowers risk. |
| 10 | `CODE_GENDER_M` | 0.0585 | ↑ risk | Gender = Male (see fairness caveat). |
| 11 | `DAYS_BIRTH` | 0.0533 | ↑ risk | Age (stored as negative days) — younger applicants default more. |
| 12 | `INS_LATE_MEAN` | 0.0533 | ↑ risk | Share of past installments paid late — direct repayment discipline. |
| 13 | `CODE_GENDER_F` | 0.0519 | ↓ risk | Gender = Female (see fairness caveat). |
| 14 | `INS_AMT_PAYMENT_SUM` | 0.0484 | ↓ risk | Total amount repaid across past installments. |
| 15 | `BURO_AMT_CREDIT_SUM_DEBT_STD` | 0.0455 | ↑ risk | Variability of outstanding debt across bureau credits. |

## What drives default (plain language)

1. **External credit scores still dominate.** `EXT_SOURCE_2/3/1` are the top signals by a wide margin — higher scores → lower predicted default. They summarise external bureau information the application form can't.
2. **Engineered loan-structure ratios pay off.** `GOODS_CREDIT_RATIO` and `CREDIT_TERM` (built in Phase 1) rank among the very top features — higher goods-to-credit (larger down-payment) and shorter effective term both **lower** risk.
3. **Repayment history now matters (Phase 2).** Side-table features surface among the drivers — most tellingly **`INS_LATE_MEAN`**, the share of past installments paid late (**more past lateness → higher default**), plus prior-application timing (`PREV_*`) and bureau debt variability (`BURO_*`). This is exactly the behavioural signal Phase 2 set out to capture, and it directly explains the +0.016 test-AUC lift.
4. **Age & employment stability.** Shorter employment (`DAYS_EMPLOYED`) and younger applicants (`DAYS_BIRTH`) default more; larger instalments (`AMT_ANNUITY`) raise risk — life/job stability and affordable payments lower it.
5. **Demographics.** Higher education lowers risk; `CODE_GENDER` appears among the drivers — flagged under fairness below.

## Figures

- `figures/14_shap_importance.png` — global importance (mean |SHAP|)
- `figures/15_shap_beeswarm.png` — per-feature impact **and** direction
- `figures/16_shap_dependence.png` — how risk varies across the top drivers

## Caveats

- SHAP shows the **model's** associations, **not causation**.
- Engineered ratios share information with their source amounts, so importance can spread across related features — read them as a group, not in isolation.
- Fairness: `CODE_GENDER` and similar attributes are present; a production system should review protected-attribute usage and disparate impact (Phase-2 item).
