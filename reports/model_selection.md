# Stage E — Model Selection

_Trained 2026-07-05T10:29:38+00:00 · seed 42 · imbalance handled via class_weight / scale_pos_weight=11.39._

Three models on the same leakage-safe pipeline. Hyperparameters tuned via bounded `RandomizedSearchCV` (StratifiedKFold) on a 50,000-row stratified subsample of train; each winner then refit on FULL train for the validation comparison. Comparison metric = subsample CV ROC-AUC; final selection = validation ROC-AUC.

| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Fit (s) |
|---|---|---|---|---|
| logistic_regression | 0.7436 | 0.7528 | 0.2360 | 37.6 |
| random_forest | 0.7387 | 0.7468 | 0.2289 | 193.7 |
| xgboost **(selected)** | 0.7514 | 0.7633 | 0.2533 | 97.1 |

**Selected:** `xgboost` (refit on train+val; 268 model features).

**Tuned params:** `{'subsample': 0.9, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.7}`

_PR-AUC is emphasised alongside ROC-AUC because of the ~8% positive rate; the untouched test set is scored in Stage F._
