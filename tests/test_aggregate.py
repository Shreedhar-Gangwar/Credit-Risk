"""Tests for the Phase-2 side-table aggregation helpers (fast, synthetic data)."""
import numpy as np
import pandas as pd

from features.aggregate import (
    aggregate_numeric, aggregate_categorical, group_count,
)


def _toy():
    # 3 rows for id=1, 2 rows for id=2
    return pd.DataFrame({
        "ID": [1, 1, 1, 2, 2],
        "AMT": [10.0, 20.0, 30.0, 5.0, 15.0],
        "CAT": ["a", "a", "b", "b", "c"],
    })


def test_aggregate_numeric_shapes_and_names():
    g = aggregate_numeric(_toy(), "ID", "T", num_aggs=["min", "max", "mean"])
    assert list(g.index) == [1, 2]
    assert set(g.columns) == {"T_AMT_MIN", "T_AMT_MAX", "T_AMT_MEAN"}
    assert g.loc[1, "T_AMT_MEAN"] == 20.0          # (10+20+30)/3
    assert g.loc[2, "T_AMT_MAX"] == 15.0


def test_group_count():
    c = group_count(_toy(), "ID", "T_COUNT")
    assert c.loc[1, "T_COUNT"] == 3 and c.loc[2, "T_COUNT"] == 2


def test_aggregate_categorical_rates_sum_to_one():
    g = aggregate_categorical(_toy(), "ID", ["CAT"], "T")
    # id=1: a,a,b -> a=2/3, b=1/3 ; rates across the row sum to 1
    row = g.loc[1].fillna(0)
    assert abs(row.sum() - 1.0) < 1e-9
    assert abs(g.loc[1, "T_CAT_a_RATE"] - 2 / 3) < 1e-9


def test_categorical_top_n_caps_columns():
    # 5 distinct levels but top_n=2 keeps only the 2 most common
    df = pd.DataFrame({"ID": [1] * 5 + [2] * 5,
                       "CAT": list("aabbc") + list("aaabc")})
    g = aggregate_categorical(df, "ID", ["CAT"], "T", top_n=2)
    assert g.shape[1] == 2                          # capped
    # 'a' is the most common overall and must be retained
    assert any("_CAT_a_RATE" in c for c in g.columns)


def test_build_feature_matrix_preserves_rows(cfg):
    """If aggregates are cached, a left-join keeps the application row count."""
    import pytest
    from config import resolve_path
    from features.build import build_feature_matrix
    processed = resolve_path(cfg["data"]["processed_dir"])
    if not any(processed.glob("agg_*.parquet")):
        pytest.skip("no cached aggregates — run the Phase-2 build first")
    app = pd.DataFrame({cfg["data"]["id_column"]: [100002, 100003, 999999999],
                        cfg["data"]["target"]: [1, 0, 0]})
    wide = build_feature_matrix(app, cfg)
    assert len(wide) == 3                           # left-join, no row change
    assert wide.shape[1] > 2                        # aggregate columns added
