"""Raw data loading for the credit-risk pipeline (Stage 1: data understanding).

This module only *reads* the raw application table and performs light integrity
checks. It does NOT clean, impute, encode, or split — those live in later stages
so that all learned transforms can be fit on the training split alone (leakage
prevention, CLAUDE.md §5.3).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_config, resolve_path


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw ``application_train.csv`` table.

    Parameters
    ----------
    path : optional
        Override the CSV location; defaults to ``data.raw_train`` in config.

    Returns
    -------
    pandas.DataFrame
        The raw table, unmodified except for being read into memory.
    """
    cfg = load_config()
    csv_path = resolve_path(path) if path is not None else resolve_path(cfg["data"]["raw_train"])
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. "
            "See README for how to obtain the Home Credit dataset."
        )
    return pd.read_csv(csv_path)


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from target, dropping the ID column.

    The ID (``SK_ID_CURR``) is an identifier, not a predictor, so it is removed
    from the feature matrix. Returns ``(X, y)``.
    """
    cfg = load_config()
    target = cfg["data"]["target"]
    id_col = cfg["data"]["id_column"]

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not present in dataframe.")

    y = df[target]
    drop_cols = [c for c in (target, id_col) if c in df.columns]
    X = df.drop(columns=drop_cols)
    return X, y


def quick_overview(df: pd.DataFrame) -> dict:
    """Return a compact data-understanding summary (shape, target balance, missing).

    Used by the Stage-1 report and as a sanity check in ``run_pipeline.py``.
    """
    cfg = load_config()
    target = cfg["data"]["target"]

    n_rows, n_cols = df.shape
    target_counts = df[target].value_counts().sort_index().to_dict() if target in df else {}
    default_rate = float(df[target].mean()) if target in df else None
    missing = df.isna().sum()
    cols_with_missing = int((missing > 0).sum())

    return {
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "target_counts": {int(k): int(v) for k, v in target_counts.items()},
        "default_rate": default_rate,
        "n_cols_with_missing": cols_with_missing,
        "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()},
    }


if __name__ == "__main__":
    # Manual smoke test: load and print a one-line overview.
    frame = load_raw()
    print(quick_overview(frame))
