"""Stage — probability calibration (Phase-1 polish).

The base XGBoost uses `scale_pos_weight`, which sharpens ranking but inflates the output
probabilities (predicted ~0.4 vs an 8% base rate). This step learns a 1-D calibration map
so `probability_default` can be read as an actual probability, without touching ranking.

Method (leakage-safe):
  * Get **out-of-fold** base predictions on train+val via `cross_val_predict` (each row is
    scored by a model that did not train on it).
  * Fit both **isotonic** and **sigmoid (Platt)** maps on those OOF scores; pick the lower
    OOF Brier (chosen without looking at test).
  * **Augment** the existing artifact with the calibrator — the base pipeline (used for SHAP
    and ranking) is unchanged; only a small calibrator is added.
  * Report Brier + a reliability diagram on the untouched test set.

Isotonic/sigmoid are monotonic, so ROC-AUC / PR-AUC are unchanged — only probability
quality improves. Run:  python -m models.calibrate
"""
from __future__ import annotations

import joblib
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import ROOT, load_config, resolve_path
from preprocessing.load import split_X_y
from preprocessing.pipeline import make_split
from features.build import load_modeling_frame
from models.train import build_model_pipeline
from models.scoring import ProbabilityCalibrator

BLUE, RED = "#4C78A8", "#E4572E"


def calibrate(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    art_path = resolve_path(cfg["artifacts"]["pipeline_file"])
    bundle = joblib.load(art_path)
    base_pipe, meta = bundle["pipeline"], bundle["metadata"]
    numeric, categorical = meta["numeric_features"], meta["categorical_features"]
    spw = meta["scale_pos_weight"]

    df = load_modeling_frame(cfg)
    X, y = split_X_y(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y, cfg)
    X_fit = pd.concat([X_train, X_val]); y_fit = pd.concat([y_train, y_val])
    # Free the wide intermediates (~4 GB on the Phase-2 matrix); keep only what's needed:
    # X_fit/y_fit for OOF calibration, X_test/y_test for the test-set effect.
    del df, X, y, X_train, X_val, y_train, y_val
    gc.collect()

    # Out-of-fold base scores on train+val (rebuild the tuned base pipeline).
    oof_pipe = build_model_pipeline("xgboost", numeric, categorical, cfg, spw, seed, n_jobs_model=-1)
    oof_pipe.set_params(**{f"model__{k}": v for k, v in meta["best_params"].items()})
    skf = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=seed)
    print(f"computing out-of-fold scores on {len(X_fit):,} rows ({cfg['cv']['n_splits']}-fold)...")
    oof = cross_val_predict(oof_pipe, X_fit, y_fit, cv=skf,
                            method="predict_proba", n_jobs=1)[:, 1]

    # Fit both calibrators on OOF; choose by OOF Brier (never on test).
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_fit)
    sig = LogisticRegression(max_iter=1000).fit(oof.reshape(-1, 1), y_fit)
    cand = {
        "isotonic": ProbabilityCalibrator("isotonic", iso),
        "sigmoid": ProbabilityCalibrator("sigmoid", sig),
    }
    oof_brier = {name: brier_score_loss(y_fit, c.predict(oof)) for name, c in cand.items()}
    best = min(oof_brier, key=oof_brier.get)
    calibrator = cand[best]
    print(f"OOF Brier — isotonic={oof_brier['isotonic']:.4f}  sigmoid={oof_brier['sigmoid']:.4f}"
          f"  -> selected: {best}")

    # Test-set effect (base already fit on train+val, in the bundle).
    base_test = base_pipe.predict_proba(X_test)[:, 1]
    cal_test = calibrator.predict(base_test)
    brier_before = brier_score_loss(y_test, base_test)
    brier_after = brier_score_loss(y_test, cal_test)
    auc_before = roc_auc_score(y_test, base_test)
    auc_after = roc_auc_score(y_test, cal_test)
    print(f"TEST Brier: {brier_before:.4f} -> {brier_after:.4f}   "
          f"ROC-AUC: {auc_before:.4f} -> {auc_after:.4f} (unchanged; calibration is monotonic)")
    print(f"TEST mean pred: base={base_test.mean():.4f}  calibrated={cal_test.mean():.4f}  "
          f"(prevalence {y_test.mean():.4f})")

    # Persist: augment the artifact with the calibrator + metadata.
    meta["calibration"] = {
        "method": best,
        "oof_brier_isotonic": float(oof_brier["isotonic"]),
        "oof_brier_sigmoid": float(oof_brier["sigmoid"]),
        "test_brier_before": float(brier_before),
        "test_brier_after": float(brier_after),
        "test_roc_auc": float(auc_after),
    }
    bundle["calibrator"] = calibrator
    bundle["metadata"] = meta
    joblib.dump(bundle, art_path)
    print(f"updated artifact (added calibrator) -> {art_path.relative_to(ROOT)}")

    _plot_reliability(y_test, base_test, cal_test, best, cfg)
    _write_report(meta["calibration"], y_test, base_test, cal_test, cfg)
    return meta["calibration"]


def _plot_reliability(y, before, after, method, cfg):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for scores, label, color in [(before, "before (raw)", RED), (after, f"after ({method})", BLUE)]:
        frac_pos, mean_pred = calibration_curve(y, scores, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, "o-", color=color, label=label)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfectly calibrated")
    ax.set(xlabel="mean predicted probability", ylabel="observed default rate",
           title="Reliability diagram — test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    d = resolve_path(cfg["reports"]["figures_dir"]); d.mkdir(parents=True, exist_ok=True)
    out = d / "17_calibration.png"
    fig.savefig(out, bbox_inches="tight", dpi=130); plt.close(fig)
    print(f"saved -> {out.relative_to(ROOT)}")


def _write_report(cal: dict, y, before, after, cfg):
    lines = [
        "# Probability Calibration\n",
        f"The base XGBoost uses `scale_pos_weight`, which inflates output probabilities "
        f"(test mean {before.mean():.3f} vs {y.mean():.3f} prevalence). A **{cal['method']}** "
        "map, fit on out-of-fold train+val scores, corrects this without changing ranking.\n",
        "| Metric (test) | Before | After |",
        "|---|---|---|",
        f"| Brier score (lower better) | {cal['test_brier_before']:.4f} | "
        f"**{cal['test_brier_after']:.4f}** |",
        f"| Mean predicted PD | {before.mean():.3f} | {after.mean():.3f} (prevalence {y.mean():.3f}) |",
        f"| ROC-AUC | {cal['test_roc_auc']:.4f} | {cal['test_roc_auc']:.4f} (unchanged) |",
        "\nCalibration was selected by out-of-fold Brier "
        f"(isotonic {cal['oof_brier_isotonic']:.4f} vs sigmoid {cal['oof_brier_sigmoid']:.4f}); "
        "the test set was not used to choose it. Because isotonic/sigmoid are monotonic, "
        "ROC-AUC and PR-AUC are unchanged — only probability quality improves, so "
        "`probability_default` can now be read as an actual probability.\n",
        "See `figures/17_calibration.png` for the reliability diagram (raw vs calibrated).\n",
    ]
    out = resolve_path(cfg["reports"]["dir"]) / "calibration.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    calibrate()
