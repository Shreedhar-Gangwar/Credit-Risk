# Stage E — Model Selection

_Trained 2026-07-07T06:35:45+00:00 · seed 42 · imbalance handled via class_weight / scale_pos_weight=11.39._

Three models on the same leakage-safe pipeline. Hyperparameters tuned via bounded `RandomizedSearchCV` (StratifiedKFold) on a 50,000-row stratified subsample of train; each winner then refit on FULL train for the validation comparison. Comparison metric = subsample CV ROC-AUC; final selection = validation ROC-AUC.

| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Fit (s) |
|---|---|---|---|---|
| logistic_regression | 0.7617 | 0.7758 | 0.2640 | 319.8 |
| random_forest | 0.7506 | 0.7584 | 0.2417 | 4506.7 |
| xgboost **(selected)** | 0.7698 | 0.7799 | 0.2742 | 932.1 |

**Selected:** `xgboost` (refit on train+val; 1020 model features).

**Tuned params:** `{'subsample': 0.9, 'max_depth': 3, 'learning_rate': 0.05, 'colsample_bytree': 0.7}`

**Interpretability note:** Logistic Regression is within 0.01 val ROC-AUC of the winner — the simpler, more interpretable model is a defensible alternative if transparency is prioritized.

_PR-AUC is emphasised alongside ROC-AUC because of the ~8% positive rate; the untouched test set is scored in Stage F._
