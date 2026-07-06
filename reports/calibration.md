# Probability Calibration

The base XGBoost uses `scale_pos_weight`, which inflates output probabilities (test mean 0.416 vs 0.081 prevalence). A **isotonic** map, fit on out-of-fold train+val scores, corrects this without changing ranking.

| Metric (test) | Before | After |
|---|---|---|
| Brier score (lower better) | 0.1964 | **0.0674** |
| Mean predicted PD | 0.416 | 0.081 (prevalence 0.081) |
| ROC-AUC | 0.7630 | 0.7630 (unchanged) |

Calibration was selected by out-of-fold Brier (isotonic 0.0677 vs sigmoid 0.0679); the test set was not used to choose it. Because isotonic/sigmoid are monotonic, ROC-AUC and PR-AUC are unchanged — only probability quality improves, so `probability_default` can now be read as an actual probability.

See `figures/17_calibration.png` for the reliability diagram (raw vs calibrated).
