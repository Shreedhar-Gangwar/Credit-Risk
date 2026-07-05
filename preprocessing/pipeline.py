"""Stage 3 (part 2) — reusable, leakage-safe preprocessing.

Provides:
  * `make_split`      — stratified train/validation/test split (70/15/15 by default).
  * `infer_feature_types` — split columns into numeric vs categorical after cleaning.
  * `build_preprocessor`  — a `ColumnTransformer` (impute + missing-indicators +
                            optional scaling + one-hot) fit on TRAIN ONLY.

Anti-leakage design (CLAUDE.md §5.3): the split happens *before* any transform is
fit; every stateful step (imputer statistics, scaler mean/var, one-hot categories)
is learned inside the `ColumnTransformer` from the training split alone, then reused
unchanged on validation/test/inference. The `CreditDataCleaner` in front is stateless.

The preprocessor is a factory with a `scale` switch so the linear model (Logistic
Regression) gets standardized features while the tree models (RF, XGBoost) — which
are invariant to monotonic scaling — skip it. Both share identical impute/encode
logic, so there is one consistent feature space.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from config import load_config
from preprocessing.cleaning import CreditDataCleaner
from features.engineer import CreditFeatureEngineer

# One-hot rare-category collapse: categories rarer than this fraction of the
# training rows are merged into an "infrequent" bucket. This tames the high
# cardinality of ORGANIZATION_TYPE (58) / OCCUPATION_TYPE (18) without a manual map,
# and is learned on train only (leakage-safe). Unknown categories at inference are
# routed to the same infrequent bucket.
ONEHOT_MIN_FREQUENCY = 0.01


def make_split(X: pd.DataFrame, y: pd.Series, cfg: dict | None = None):
    """Stratified train/validation/test split.

    ``test_size`` and ``val_size`` in config are fractions of the WHOLE dataset.
    Test is carved out first, then validation from the remainder, so the resulting
    proportions match the config (default 0.70 / 0.15 / 0.15).

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``.
    """
    cfg = cfg or load_config()
    seed = cfg["seed"]
    test_size = cfg["split"]["test_size"]
    val_size = cfg["split"]["val_size"]
    stratify = cfg["split"].get("stratify", True)

    strat = y if stratify else None
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat
    )
    # validation as a fraction of the whole -> relative fraction of the remainder
    val_relative = val_size / (1.0 - test_size)
    strat_tmp = y_tmp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_relative, random_state=seed, stratify=strat_tmp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def infer_feature_types(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_features, categorical_features) by dtype.

    Call on a CLEANED + ENGINEERED frame so derived columns (``DAYS_EMPLOYED_ANOM``
    and the engineered ratios) are typed correctly. Binary ``FLAG_*`` columns are ints
    and fall in the numeric group — they pass through the median imputer untouched
    (they have no missing values) and are only standardized in the scaled variant.
    """
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numeric, categorical


def prepare_feature_types(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Infer feature types on a cleaned + engineered view of the TRAINING features.

    Single source of truth for the column lists that `build_preprocessor` needs, so
    Stage E and `run_pipeline.py` don't each re-derive them. Uses stateless transforms,
    so applying them here for typing introduces no leakage.
    """
    preview = CreditFeatureEngineer().fit_transform(
        CreditDataCleaner().fit_transform(X_train)
    )
    return infer_feature_types(preview)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale: bool = True,
) -> ColumnTransformer:
    """Build the fit-on-train `ColumnTransformer`.

    Numeric  : median impute (robust to the skew/outliers seen in EDA) + a
               missing-indicator per feature that had gaps (missingness is
               informative — EDA §5b) + optional standardization.
    Categorical: constant impute ('Missing') + one-hot with rare-category collapse.
    """
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=ONEHOT_MIN_FREQUENCY,
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scale: bool = True,
    memory=None,
) -> Pipeline:
    """Full leakage-safe preprocessing head: cleaner -> feature engineer -> ColumnTransformer.

    This is what Stage E prepends to each model. Kept here so preprocessing owns the
    whole transform chain and it is reused identically at inference. The engineered
    ratio columns must be present in ``numeric_features`` (use `prepare_feature_types`).

    ``memory`` (a path or joblib ``Memory``) caches the fitted transformer steps so that,
    during hyperparameter search, preprocessing is computed once per CV fold and reused
    across candidates that differ only in the model — a big speedup, still leakage-safe
    (the cache key includes the fold's training data).
    """
    return Pipeline(
        [
            ("clean", CreditDataCleaner()),
            ("engineer", CreditFeatureEngineer()),
            ("prep", build_preprocessor(numeric_features, categorical_features, scale)),
        ],
        memory=memory,
    )
