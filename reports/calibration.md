# Probability Calibration

The base XGBoost uses `scale_pos_weight`, which inflates output probabilities (test mean 0.406 vs 0.081 prevalence). A **isotonic** map, fit on out-of-fold train+val scores, corrects this without changing ranking.

| Metric (test) | Before | After |
|---|---|---|
| Brier score (lower better) | 0.1887 | **0.0663** |
| Mean predicted PD | 0.406 | 0.081 (prevalence 0.081) |
| ROC-AUC | 0.7798 | 0.7798 (unchanged) |

Calibration was selected by out-of-fold Brier (isotonic 0.0665 vs sigmoid 0.0667); the test set was not used to choose it. Because isotonic/sigmoid are monotonic, ROC-AUC and PR-AUC are unchanged — only probability quality improves, so `probability_default` can now be read as an actual probability.

See `figures/17_calibration.png` for the reliability diagram (raw vs calibrated).
