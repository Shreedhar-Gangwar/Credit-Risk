"""Stage 4 — business-relevant feature engineering.

`CreditFeatureEngineer` derives ratio features that encode lending-domain intuition the
raw columns only imply. Each ratio has a clear business rationale (CLAUDE.md §5.4) and is
built only from columns the dataset actually provides.

It is **stateless** (pure arithmetic, learns nothing), so it is leakage-safe and sits in
the model pipeline between the cleaner and the `ColumnTransformer`. It takes a DataFrame in
and returns a DataFrame out (original columns + new ratios) so the downstream
`ColumnTransformer` can still select by name and the new features flow into imputation.

Divide-by-zero / undefined results are converted to NaN and left for the downstream
median imputer + missing-indicator to handle consistently (train-fitted, no leakage).

Engineered features and why they matter for default risk
--------------------------------------------------------
CREDIT_INCOME_RATIO   = AMT_CREDIT / AMT_INCOME_TOTAL
    Loan size relative to income — core measure of loan burden / over-leverage.
ANNUITY_INCOME_RATIO  = AMT_ANNUITY / AMT_INCOME_TOTAL
    Debt-to-income proxy: what share of income each instalment consumes. High = strain.
CREDIT_TERM           = AMT_ANNUITY / AMT_CREDIT
    Repayment intensity (annuity per unit credit); its inverse approximates loan term.
EMPLOYED_TO_AGE       = DAYS_EMPLOYED / DAYS_BIRTH
    Employment length as a fraction of age — job stability / career maturity. (Both are
    negative day-counts, so the ratio is positive; NaN for the pensioner cohort, which is
    already flagged by DAYS_EMPLOYED_ANOM.)
INCOME_PER_PERSON     = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
    Per-capita income — the same salary supports fewer dependants at higher family size.
GOODS_CREDIT_RATIO    = AMT_GOODS_PRICE / AMT_CREDIT
    Share of the loan backed by the purchased good; a low ratio implies more borrowed
    beyond the good (a down-payment / collateral-coverage proxy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# (new_column, numerator, denominator)
RATIO_SPECS: list[tuple[str, str, str]] = [
    ("CREDIT_INCOME_RATIO", "AMT_CREDIT", "AMT_INCOME_TOTAL"),
    ("ANNUITY_INCOME_RATIO", "AMT_ANNUITY", "AMT_INCOME_TOTAL"),
    ("CREDIT_TERM", "AMT_ANNUITY", "AMT_CREDIT"),
    ("EMPLOYED_TO_AGE", "DAYS_EMPLOYED", "DAYS_BIRTH"),
    ("INCOME_PER_PERSON", "AMT_INCOME_TOTAL", "CNT_FAM_MEMBERS"),
    ("GOODS_CREDIT_RATIO", "AMT_GOODS_PRICE", "AMT_CREDIT"),
]

ENGINEERED_FEATURES: list[str] = [name for name, _, _ in RATIO_SPECS]


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add business ratio features. Stateless and leakage-safe.

    A ratio is only created if both of its source columns are present, so the
    transformer degrades gracefully on partial inputs at inference.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        added = [name for name, num, den in RATIO_SPECS
                 if num in X.columns and den in X.columns]
        self.added_features_ = added
        self.feature_names_out_ = np.asarray(list(X.columns) + added)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditFeatureEngineer expects a pandas DataFrame.")
        X = X.copy()
        for name, num, den in RATIO_SPECS:
            if num in X.columns and den in X.columns:
                # divide, then neutralise divide-by-zero (inf) -> NaN for the imputer
                ratio = X[num] / X[den]
                X[name] = ratio.replace([np.inf, -np.inf], np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
