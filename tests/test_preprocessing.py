"""Tests for cleaning, feature engineering, splitting, and leakage-safety."""
import numpy as np

from preprocessing.load import split_X_y
from preprocessing.cleaning import CreditDataCleaner, DAYS_EMPLOYED_SENTINEL
from features.engineer import CreditFeatureEngineer, ENGINEERED_FEATURES
from preprocessing.pipeline import (
    make_split, prepare_feature_types, build_preprocessing_pipeline,
)


def test_cleaner_neutralises_sentinels(raw_sample):
    cleaned = CreditDataCleaner().fit_transform(raw_sample)
    # DAYS_EMPLOYED sentinel replaced and flagged
    assert (cleaned["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum() == 0
    assert "DAYS_EMPLOYED_ANOM" in cleaned.columns
    assert set(cleaned["DAYS_EMPLOYED_ANOM"].unique()).issubset({0, 1})
    # anomaly flag aligns with where the sentinel was
    orig_anom = (raw_sample["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL).sum()
    assert cleaned["DAYS_EMPLOYED_ANOM"].sum() == orig_anom
    # string missings neutralised
    assert (cleaned["CODE_GENDER"] == "XNA").sum() == 0


def test_feature_engineer_adds_ratios_without_inf(raw_sample):
    cleaned = CreditDataCleaner().fit_transform(raw_sample)
    eng = CreditFeatureEngineer().fit_transform(cleaned)
    for feat in ENGINEERED_FEATURES:
        assert feat in eng.columns
        assert not np.isinf(eng[feat].to_numpy()).any(), f"{feat} has inf"


def test_split_is_stratified_and_disjoint(raw_sample, cfg):
    X, y = split_X_y(raw_sample)
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y, cfg)
    # sizes sum and no leakage across splits
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    idx = set(X_train.index) | set(X_val.index) | set(X_test.index)
    assert len(idx) == len(X)
    # stratification: prevalence close across splits
    for ys in (y_train, y_val, y_test):
        assert abs(ys.mean() - y.mean()) < 0.02


def test_preprocessor_fit_on_train_only_transforms_cleanly(raw_sample, cfg):
    X, y = split_X_y(raw_sample)
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y, cfg)
    numeric, categorical = prepare_feature_types(X_train)

    pipe = build_preprocessing_pipeline(numeric, categorical, scale=False)
    pipe.fit(X_train)                       # fit on train ONLY
    Zt, Zv, Zte = pipe.transform(X_train), pipe.transform(X_val), pipe.transform(X_test)

    # no NaN leaks through, and column space is stable across splits
    for Z in (Zt, Zv, Zte):
        assert not np.isnan(Z).any()
    assert Zt.shape[1] == Zv.shape[1] == Zte.shape[1]
    # engineered features survive into the model feature space
    names = pipe.named_steps["prep"].get_feature_names_out()
    assert any("CREDIT_INCOME_RATIO" in n for n in names)
