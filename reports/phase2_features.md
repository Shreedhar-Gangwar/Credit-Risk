# Phase 2 — Relational Side-Table Features

Phase 1 used only the application form. Phase 2 folds in the applicant's **prior credit
history** from the relational side-tables, each aggregated to one row per `SK_ID_CURR`
(leakage-safe: per-applicant, no target, no cross-applicant statistics) and left-joined
onto the application table. Applicants with no history get NaN (handled by median-impute +
missing-indicator; "no history" is itself informative).

## Feature groups (final model)
| Table | Prefix | Features | Train coverage | What it captures |
|---|---|---|---|---|
| bureau (+ bureau_balance) | `BURO_` / `BB_` | 140 | 85.7% | credits at other lenders, active/closed, monthly DPD status |
| previous_application | `PREV_` | 180 | 94.6% | prior Home Credit apps: approve/refuse mix, applied-vs-granted, terms |
| installments_payments | `INS_` | 56 | 94.8% | **repayment behaviour**: paid late/early, under/overpaid |
| application (Phase 1) | — | 120 (+7 engineered) | 100% | the application form |

**Dropped after testing:** `pos_cash` (no lift, −0.0009 — redundant with installments) and
`credit_card` (marginal +0.0014, only 28% coverage). High-cardinality categoricals in
`previous_application` were capped to their 6 most common levels (244 → 180 features) to
control feature count and memory. Final matrix: **~503 features**.

## Lift measured incrementally (regularized XGBoost, validation ROC-AUC)
| Added | Val ROC-AUC | Δ |
|---|---|---|
| Baseline (Phase 1) | 0.7653 | — |
| + bureau | 0.7688 | +0.0035 |
| + previous_application | 0.7783 | +0.0095 |
| + installments | 0.7831 | +0.0048 |
| ~~+ pos_cash~~ (dropped) | 0.7822 | −0.0009 |
| ~~+ credit_card~~ (dropped) | 0.7836 | +0.0014 |

`previous_application` and `installments` carried the most signal; POS/credit-card did not
earn their place.

## Final model — Phase 1 vs Phase 2 (untouched test set)
| Metric (test, n=46,127) | Phase 1 | Phase 2 | Δ |
|---|---|---|---|
| ROC-AUC | 0.7636 | **0.7800** | **+0.0164** |
| PR-AUC | 0.2505 | **0.2687** | +0.0182 |
| Recall @ cost-optimal (FN:FP=10:1) | 0.648 | **0.683** | +0.035 |
| Brier (calibrated) | 0.067 | 0.066 | — |

Same final model family (XGBoost, `max_depth=3, lr=0.05`), re-tuned and re-calibrated
(isotonic) on the wide matrix. Ranking preserved by calibration; probabilities remain
readable (mean ≈ base rate).

## Interpretation (SHAP) — history features earn their place
External scores still dominate, but **`INS_LATE_MEAN`** (share of past installments paid
late) enters the top 15 (↑ risk), alongside prior-application timing (`PREV_*`) and bureau
debt variability (`BURO_*`). This is exactly the behavioural signal Phase 2 set out to
capture and explains the test-AUC lift. See `reports/interpretation.md`.

## Reproducibility & memory notes
- Side-table aggregation is config-driven (`features.side_tables`); emptying the list
  reproduces the Phase-1 model. Aggregates are cached to `data/processed/agg_*.parquet`.
- The wide matrix is memory-heavy; aggregates are stored as float32 and the training/
  calibration steps free the full frames after splitting to stay within RAM.
