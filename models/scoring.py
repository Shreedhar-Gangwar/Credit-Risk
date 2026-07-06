"""Shared scoring helper — the single place that turns a saved bundle into a
probability of default, applying probability calibration if the bundle has it.

Used by evaluation, inference, and the runner so every consumer produces identical
scores. Keeping `ProbabilityCalibrator` in this stable module means joblib can always
resolve the class when loading a calibrated artifact.
"""
from __future__ import annotations

import numpy as np


class ProbabilityCalibrator:
    """1-D calibration mapping a base model score -> a calibrated probability.

    Wraps either an isotonic regressor or a sigmoid (Platt) logistic model behind a
    single ``predict(scores) -> calibrated`` interface.
    """

    def __init__(self, method: str, model):
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"Unknown calibration method: {method}")
        self.method = method
        self.model = model

    def predict(self, scores) -> np.ndarray:
        p = np.asarray(scores, dtype=float).reshape(-1)
        if self.method == "isotonic":
            out = self.model.predict(p)
        else:  # sigmoid / Platt
            out = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        return np.clip(out, 0.0, 1.0)


def default_proba(bundle: dict, X) -> np.ndarray:
    """Probability of default for rows ``X`` from a saved bundle.

    Applies the bundle's calibrator when present; otherwise returns the raw base
    model probability (ranking-equivalent, but uncalibrated).
    """
    base = bundle["pipeline"].predict_proba(X)[:, 1]
    calibrator = bundle.get("calibrator")
    return calibrator.predict(base) if calibrator is not None else base
