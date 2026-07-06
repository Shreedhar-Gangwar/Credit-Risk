# Stage F — Model Evaluation (untouched test set)

Final model: **xgboost** (refit on train+val, calibrated). Test set = 15% held out, scored once. Prevalence (default rate) = 0.0807.

## Threshold-independent metrics (headline)

| Metric | Value | Reading |
|---|---|---|
| ROC-AUC | **0.7630** | ranking quality across all thresholds |
| PR-AUC | **0.2436** | precision/recall on the rare positive class (baseline = 0.081) |
| Brier | 0.0674 | probability calibration (scores are calibrated (see `calibration.md`) — readable as actual probabilities) |

ROC-AUC and PR-AUC are the fair summary because they don't depend on an operating point. PR-AUC well above the prevalence baseline shows real signal on the minority (default) class.

## Operating point & the business trade-off

Two errors cost differently: a **false negative** (approve a defaulter) loses principal; a **false positive** (reject a good applicant) loses interest margin. Using illustrative relative costs **FN:FP = 10:1**, the expected-cost-minimising threshold — chosen on the validation set, then applied to test — is **0.090**.

| Operating point | Threshold | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| Cost-optimal (FN:FP=10:1) | 0.090 | 0.180 | 0.645 | 0.281 | 2403 | 10969 | 1321 | 31434 |
| Base-rate reference | 0.081 | 0.164 | 0.712 | 0.266 | 2650 | 13545 | 1074 | 28858 |

At the cost-optimal threshold the model catches **65% of defaulters** (recall) at **18% precision** — i.e. it prevents 2,403 bad loans while wrongly declining 10,969 good applicants. Because a false negative is judged 10× costlier than a false positive, the optimum leans toward recall: catching defaulters matters more than avoiding the occasional wrongful decline. Raising the FN:FP ratio pushes the threshold lower and recall higher (see `figures/12_threshold_analysis.png`); the right value is a business policy decision, not a modelling one.

## Figures

- `figures/09_roc_curve.png` — ROC curve
- `figures/10_pr_curve.png` — precision–recall curve
- `figures/11_confusion_matrix.png` — confusion matrix at the cost-optimal threshold
- `figures/12_threshold_analysis.png` — precision/recall/F1 and expected cost vs threshold
- `figures/13_score_distribution.png` — score distribution by class (shows separation + inflation)

## Caveats

- Costs are **illustrative**; real values come from the lender's loss-given-default and margin economics.
- Probabilities are **uncalibrated** (`scale_pos_weight`); if literal PDs are needed, add probability calibration (e.g. isotonic/Platt on a held-out fold) — a Phase-2 item.
