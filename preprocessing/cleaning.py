"""Stage 3 (part 1) — data cleaning as a leakage-safe sklearn transformer.

`CreditDataCleaner` fixes the anomalies identified in Stage 1 / confirmed in EDA:
  * `DAYS_EMPLOYED == 365243` sentinel (18% of rows, = "not employed / pensioner")
    -> NaN, and records the fact in a new binary column `DAYS_EMPLOYED_ANOM`
    (the *fact* of being a pensioner is itself predictive — EDA §5b).
  * `CODE_GENDER == 'XNA'` (4 rows) and `NAME_FAMILY_STATUS == 'Unknown'` (2 rows)
    -> NaN, so they are imputed consistently downstream.

It is **stateless** (uses fixed constants, learns nothing from data), so it is
leakage-safe by construction and can sit at the front of the model pipeline and be
reused unchanged at inference. It takes a DataFrame in and returns a DataFrame out,
so the downstream `ColumnTransformer` can still select columns by name.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Sentinel / anomalous codes to neutralise (constants, not learned).
DAYS_EMPLOYED_SENTINEL = 365243
ANOMALOUS_CATEGORIES = {
    "CODE_GENDER": ["XNA"],
    "NAME_FAMILY_STATUS": ["Unknown"],
}


class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """Replace known sentinels/anomalies with NaN and flag the DAYS_EMPLOYED sentinel.

    Parameters
    ----------
    add_employed_anom_flag : bool, default True
        Add a ``DAYS_EMPLOYED_ANOM`` (0/1) column marking the pensioner sentinel.
    """

    def __init__(self, add_employed_anom_flag: bool = True):
        self.add_employed_anom_flag = add_employed_anom_flag

    def fit(self, X: pd.DataFrame, y=None):
        # Nothing is learned; record input columns for get_feature_names_out.
        self.feature_names_in_ = np.asarray(X.columns)
        out = list(X.columns)
        if self.add_employed_anom_flag and "DAYS_EMPLOYED" in X.columns:
            out.append("DAYS_EMPLOYED_ANOM")
        self.feature_names_out_ = np.asarray(out)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditDataCleaner expects a pandas DataFrame.")
        X = X.copy()

        if "DAYS_EMPLOYED" in X.columns:
            is_anom = (X["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL)
            if self.add_employed_anom_flag:
                X["DAYS_EMPLOYED_ANOM"] = is_anom.astype("int64")
            X.loc[is_anom, "DAYS_EMPLOYED"] = np.nan

        for col, bad_values in ANOMALOUS_CATEGORIES.items():
            if col in X.columns:
                X.loc[X[col].isin(bad_values), col] = np.nan

        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
