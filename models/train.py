"""Stage 5 — model development, tuning, selection, and artifact saving.

Trains three models on the same leakage-safe pipeline and picks a final one:
  * Logistic Regression  — interpretable linear baseline (scaled features).
  * Random Forest        — non-linear bagged trees (unscaled).
  * XGBoost              — gradient-boosted trees (unscaled).

Workflow (all leakage-safe — every transform is fit inside CV on training folds only):
  1. Stratified 70/15/15 split (test held out, untouched here).
  2. For each model: bounded `RandomizedSearchCV` (StratifiedKFold) on TRAIN for
     hyperparameters; the CV ROC-AUC is the cross-model comparison metric.
  3. Evaluate each tuned model on the VALIDATION set (ROC-AUC + PR-AUC).
  4. Select the final model on validation ROC-AUC (with an interpretability note).
  5. Refit the winning pipeline on TRAIN+VAL and save one joblib artifact
     (preprocessing + features + model together, so inference can never re-fit).

Imbalance (~8% default, 11.4:1) is handled by `class_weight` (LR, RF) and
`scale_pos_weight = n_neg/n_pos` (XGBoost).

Run:  python -m models.train
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from config import ROOT, load_config, resolve_path
from preprocessing.load import load_raw, split_X_y
from preprocessing.pipeline import (
    build_preprocessing_pipeline,
    make_split,
    prepare_feature_types,
)

MODEL_NAMES = ["logistic_regression", "random_forest", "xgboost"]
# Logistic Regression is the only model that needs standardized features.
NEEDS_SCALING = {"logistic_regression": True, "random_forest": False, "xgboost": False}


# --------------------------------------------------------------------------- #
# Model + pipeline construction
# --------------------------------------------------------------------------- #
def _build_estimator(name: str, cfg: dict, scale_pos_weight: float, seed: int, n_jobs_model: int):
    """Instantiate a bare classifier with config defaults and imbalance handling."""
    m = cfg["models"][name]
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=m["max_iter"], class_weight=m["class_weight"],
            solver=m["solver"], random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=m["n_estimators"], max_depth=m["max_depth"],
            min_samples_leaf=m["min_samples_leaf"], class_weight=m["class_weight"],
            random_state=seed, n_jobs=n_jobs_model,
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=m["n_estimators"], tree_method=m["tree_method"],
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=seed, n_jobs=n_jobs_model,
        )
    raise ValueError(f"Unknown model: {name}")


def build_model_pipeline(name, numeric, categorical, cfg, scale_pos_weight, seed,
                         n_jobs_model=1, memory=None):
    """Full pipeline: clean -> engineer -> preprocess(scale?) -> classifier.

    ``memory`` caches the preprocessing steps across hyperparameter-search candidates
    (only the final ``model`` step is never cached).
    """
    pipe = build_preprocessing_pipeline(numeric, categorical,
                                        scale=NEEDS_SCALING[name], memory=memory)
    estimator = _build_estimator(name, cfg, scale_pos_weight, seed, n_jobs_model)
    pipe.steps.append(("model", estimator))
    return pipe


def _search_space(name: str, cfg: dict) -> dict:
    """Config param grid mapped onto the pipeline's `model__<param>` namespace."""
    grid = cfg["models"][name]["search"]["param_grid"]
    return {f"model__{k}": v for k, v in grid.items()}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _proba(estimator, X) -> np.ndarray:
    return estimator.predict_proba(X)[:, 1]


def evaluate_scores(estimator, X, y) -> dict:
    """Threshold-independent metrics for imbalanced classification."""
    p = _proba(estimator, X)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
    }


# --------------------------------------------------------------------------- #
# Training / selection
# --------------------------------------------------------------------------- #
def train_and_select(cfg: dict | None = None, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    primary = cfg["models"]["primary_metric"]

    df = load_raw()
    X, y = split_X_y(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y, cfg)
    numeric, categorical = prepare_feature_types(X_train)

    n_neg, n_pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = n_neg / n_pos
    if verbose:
        print(f"train={len(X_train):,} val={len(X_val):,} test={len(X_test):,} (test held out)")
        print(f"features: {len(numeric)} numeric + {len(categorical)} categorical")
        print(f"scale_pos_weight (n_neg/n_pos) = {scale_pos_weight:.3f}\n")

    # Hyperparameter search runs on a stratified subsample of TRAIN — param rankings
    # are stable at this size and it keeps the search fast. Each winner is then refit on
    # the FULL train (fair validation comparison) and finally on train+val (artifact).
    tune_n = cfg["cv"].get("tune_sample_size")
    if tune_n and tune_n < len(X_train):
        X_tune, _, y_tune, _ = train_test_split(
            X_train, y_train, train_size=tune_n, random_state=seed, stratify=y_train)
    else:
        X_tune, y_tune = X_train, y_train
    if verbose:
        print(f"tuning subsample: {len(X_tune):,} rows (default_rate={y_tune.mean():.4f})\n")

    skf = StratifiedKFold(n_splits=cfg["cv"]["tune_n_splits"], shuffle=True, random_state=seed)
    results = {}

    for name in MODEL_NAMES:
        t0 = time.time()
        # Windows-safe parallelism: search runs sequentially (n_jobs=1 — no per-candidate
        # data copies to loky workers, which exhausted resources with WinError 1450) while
        # each estimator uses all cores internally (RF threads, XGBoost threads).
        search_pipe = build_model_pipeline(name, numeric, categorical, cfg,
                                           scale_pos_weight, seed, n_jobs_model=-1)
        search = RandomizedSearchCV(
            search_pipe, param_distributions=_search_space(name, cfg),
            n_iter=cfg["models"][name]["search"]["n_iter"],
            scoring=primary, cv=skf, random_state=seed, n_jobs=1, refit=False,
            error_score="raise",
        )
        search.fit(X_tune, y_tune)                       # tune on subsample (CV on folds)
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        # Refit tuned config on FULL train for a fair validation comparison.
        tuned_full = build_model_pipeline(name, numeric, categorical, cfg,
                                          scale_pos_weight, seed, n_jobs_model=-1)
        tuned_full.set_params(**{f"model__{k}": v for k, v in best_params.items()})
        tuned_full.fit(X_train, y_train)
        val = evaluate_scores(tuned_full, X_val, y_val)
        results[name] = {
            "cv_roc_auc": float(search.best_score_),      # subsample CV (comparison)
            "val_roc_auc": val["roc_auc"],                # full-train fit, held-out val
            "val_pr_auc": val["pr_auc"],
            "best_params": best_params,
            "fit_seconds": round(time.time() - t0, 1),
        }
        if verbose:
            r = results[name]
            print(f"[{name:20s}] cv_roc_auc={r['cv_roc_auc']:.4f}  "
                  f"val_roc_auc={r['val_roc_auc']:.4f}  val_pr_auc={r['val_pr_auc']:.4f}  "
                  f"({r['fit_seconds']}s)  params={r['best_params']}")

    # --- Selection: best validation ROC-AUC, with interpretability note --------
    ranked = sorted(MODEL_NAMES, key=lambda n: results[n]["val_roc_auc"], reverse=True)
    selected = ranked[0]
    lr = results["logistic_regression"]["val_roc_auc"]
    note = ""
    if selected != "logistic_regression" and (results[selected]["val_roc_auc"] - lr) < 0.01:
        note = ("Logistic Regression is within 0.01 val ROC-AUC of the winner — the simpler, "
                "more interpretable model is a defensible alternative if transparency is prioritized.")
    if verbose:
        print(f"\nSelected: {selected} (val_roc_auc={results[selected]['val_roc_auc']:.4f})")
        if note:
            print("Note:", note)

    # --- Refit winner on TRAIN+VAL with tuned params, then save ---------------
    X_fit = pd.concat([X_train, X_val]); y_fit = pd.concat([y_train, y_val])
    final_pipe = build_model_pipeline(selected, numeric, categorical, cfg,
                                      scale_pos_weight, seed, n_jobs_model=-1)
    final_pipe.set_params(**{f"model__{k}": v for k, v in results[selected]["best_params"].items()})
    final_pipe.fit(X_fit, y_fit)

    metadata = {
        "selected_model": selected,
        "selection_metric": "val_roc_auc",
        "selection_note": note,
        "best_params": results[selected]["best_params"],
        "comparison": {n: {k: results[n][k] for k in
                           ("cv_roc_auc", "val_roc_auc", "val_pr_auc", "best_params", "fit_seconds")}
                       for n in MODEL_NAMES},
        "refit_on": "train+val",
        "tune_sample_size": int(len(X_tune)),
        "n_features_out": int(final_pipe.named_steps["prep"].get_feature_names_out().shape[0]),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "scale_pos_weight": scale_pos_weight,
        "seed": seed,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _save_artifact(final_pipe, metadata, cfg)
    _write_report(metadata, cfg)
    return {"selected": selected, "results": results, "metadata": metadata}


def _save_artifact(pipeline, metadata: dict, cfg: dict) -> Path:
    model_dir = resolve_path(cfg["artifacts"]["model_dir"]); model_dir.mkdir(parents=True, exist_ok=True)
    path = resolve_path(cfg["artifacts"]["pipeline_file"])
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, path)
    with (model_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    print(f"\nsaved artifact -> {path.relative_to(ROOT)}")
    return path


def _write_report(metadata: dict, cfg: dict) -> None:
    comp = metadata["comparison"]
    lines = [
        "# Stage E — Model Selection\n",
        f"_Trained {metadata['trained_at_utc']} · seed {metadata['seed']} · "
        f"imbalance handled via class_weight / scale_pos_weight={metadata['scale_pos_weight']:.2f}._\n",
        "Three models on the same leakage-safe pipeline. Hyperparameters tuned via bounded "
        f"`RandomizedSearchCV` (StratifiedKFold) on a {metadata['tune_sample_size']:,}-row "
        "stratified subsample of train; each winner then refit on FULL train for the "
        "validation comparison. Comparison metric = subsample CV ROC-AUC; final selection "
        "= validation ROC-AUC.\n",
        "| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Fit (s) |",
        "|---|---|---|---|---|",
    ]
    for n in MODEL_NAMES:
        c = comp[n]
        star = " **(selected)**" if n == metadata["selected_model"] else ""
        lines.append(f"| {n}{star} | {c['cv_roc_auc']:.4f} | {c['val_roc_auc']:.4f} "
                     f"| {c['val_pr_auc']:.4f} | {c['fit_seconds']} |")
    lines += [
        f"\n**Selected:** `{metadata['selected_model']}` "
        f"(refit on {metadata['refit_on']}; {metadata['n_features_out']} model features).",
        f"\n**Tuned params:** `{metadata['best_params']}`",
    ]
    if metadata["selection_note"]:
        lines.append(f"\n**Interpretability note:** {metadata['selection_note']}")
    lines += [
        "\n_PR-AUC is emphasised alongside ROC-AUC because of the ~8% positive rate; "
        "the untouched test set is scored in Stage F._\n",
    ]
    out = resolve_path(cfg["reports"]["dir"]) / "model_selection.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    train_and_select()
