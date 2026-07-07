"""Phase 2 — assemble the wide modelling matrix (application + side-table aggregates).

`build_feature_matrix` left-joins each configured side-table's per-applicant aggregates
onto the application table, producing one row per SK_ID_CURR. Aggregates are cached to
`data/processed/agg_<table>.parquet` (deterministic, expensive to recompute) and reused.

The wide matrix flows unchanged into the existing pipeline (`split_X_y` -> `make_split` ->
`prepare_feature_types` -> `build_preprocessing_pipeline`): the aggregates are numeric, so
they pass through median-impute + missing-indicator; applicants with no history get NaN,
which is both handled and informative.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_config, resolve_path
from preprocessing.load import load_raw
from features.aggregate import TABLE_AGGREGATORS


def _downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 for aggregate columns — halves the wide matrix's memory.

    The imputer upcasts to float64 at fit time, but this keeps the pre-fit frame and
    its copies (merge, cleaner, engineer) small, which is where memory pressure builds.
    """
    f64 = df.select_dtypes("float64").columns
    if len(f64):
        df[f64] = df[f64].astype("float32")
    return df


def build_aggregate(name: str, cfg: dict) -> pd.DataFrame:
    """Return per-applicant aggregates for one side-table, using the parquet cache."""
    if name not in TABLE_AGGREGATORS:
        raise KeyError(f"No aggregator implemented for '{name}'. "
                       f"Available: {sorted(TABLE_AGGREGATORS)}")
    processed = resolve_path(cfg["data"]["processed_dir"]); processed.mkdir(parents=True, exist_ok=True)
    cache = processed / f"agg_{name}.parquet"
    use_cache = cfg["features"].get("cache_aggregates", True)

    if use_cache and cache.exists():
        return _downcast_floats(pd.read_parquet(cache))

    raw_dir = resolve_path(cfg["data"]["raw_dir"])
    agg = TABLE_AGGREGATORS[name](raw_dir)
    if use_cache:
        agg.to_parquet(cache, index=False)
    return _downcast_floats(agg)


def build_feature_matrix(app_df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Left-join all configured side-table aggregates onto the application frame."""
    cfg = cfg or load_config()
    matrix = app_df
    for name in cfg["features"].get("side_tables", []):
        agg = build_aggregate(name, cfg)
        matrix = matrix.merge(agg, how="left", on=cfg["data"]["id_column"])
    return matrix


def load_modeling_frame(cfg: dict | None = None) -> pd.DataFrame:
    """Application table widened with the configured side-table aggregates.

    With an empty ``features.side_tables`` this returns the Phase-1 application table
    unchanged (reproducibility). This is the single entry point training/eval/inference
    use to obtain the modelling matrix.
    """
    cfg = cfg or load_config()
    return build_feature_matrix(load_raw(), cfg)


def clear_aggregate_cache(cfg: dict | None = None) -> None:
    """Delete cached aggregate parquet files (use after changing aggregation logic)."""
    cfg = cfg or load_config()
    processed = resolve_path(cfg["data"]["processed_dir"])
    for p in processed.glob("agg_*.parquet"):
        p.unlink()
