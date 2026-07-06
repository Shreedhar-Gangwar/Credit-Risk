"""Tests for the saved artifact, calibration, and the inference path."""
import numpy as np

from models.scoring import default_proba
from inference.predict import score_frame, DECISION_APPROVE, DECISION_REVIEW


def test_bundle_structure(bundle):
    assert "pipeline" in bundle and "metadata" in bundle
    assert bundle["metadata"]["selected_model"] == "xgboost"


def test_default_proba_in_unit_interval(bundle, raw_sample, cfg):
    features = raw_sample.drop(columns=[cfg["data"]["target"]]).head(50)
    p = default_proba(bundle, features)
    assert p.shape == (50,)
    assert np.all((p >= 0) & (p <= 1))


def test_calibration_present_and_monotonic(bundle):
    cal = bundle.get("calibrator")
    if cal is None:
        import pytest
        pytest.skip("artifact has no calibrator (run `python -m models.calibrate`)")
    grid = np.linspace(0, 1, 50)
    out = cal.predict(grid)
    assert np.all((out >= 0) & (out <= 1))
    # calibration map must be monotonic non-decreasing (preserves ranking)
    assert np.all(np.diff(out) >= -1e-9)


def test_calibrated_scores_near_base_rate(bundle, raw_sample, cfg):
    """If calibrated, mean predicted PD should be far below the inflated raw mean."""
    if bundle.get("calibrator") is None:
        import pytest
        pytest.skip("no calibrator")
    features = raw_sample.drop(columns=[cfg["data"]["target"]])
    base = bundle["pipeline"].predict_proba(features)[:, 1]
    calibrated = default_proba(bundle, features)
    assert calibrated.mean() < base.mean()          # calibration pulled scores down
    assert calibrated.mean() < 0.25                 # near an ~8% base-rate regime


def test_score_frame_outputs(bundle, raw_sample, cfg):
    df = raw_sample.drop(columns=[cfg["data"]["target"]]).head(20)
    out = score_frame(df, cfg)
    assert list(out.columns) == [cfg["data"]["id_column"], "probability_default", "decision"]
    assert len(out) == 20
    assert out["probability_default"].between(0, 1).all()
    assert set(out["decision"]).issubset({DECISION_APPROVE, DECISION_REVIEW})


def test_inference_ignores_target_if_present(bundle, raw_sample, cfg):
    """Scoring must work whether or not TARGET is in the input (it's ignored)."""
    with_target = raw_sample.head(10)                 # still has TARGET
    without = raw_sample.drop(columns=[cfg["data"]["target"]]).head(10)
    a = score_frame(with_target, cfg)["probability_default"].to_numpy()
    b = score_frame(without, cfg)["probability_default"].to_numpy()
    assert np.allclose(a, b)
