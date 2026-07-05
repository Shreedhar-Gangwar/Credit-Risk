"""Stage 6 — evaluation of the final model on the UNTOUCHED test set.

Loads the saved artifact (preprocessing + features + XGBoost, refit on train+val) and
scores the held-out test split exactly once. Produces metrics appropriate for imbalanced
binary classification and a business-cost discussion of the two error types.

Methodology (honest by design):
  * Headline metrics are THRESHOLD-INDEPENDENT (ROC-AUC, PR-AUC) — they don't depend on
    an operating point and are the fair summary of ranking quality.
  * The operating threshold is chosen on the VALIDATION set (minimising expected business
    cost), then applied to test — we never tune the threshold on the test set.
  * We also report a naive 0.5 threshold for reference and a Brier score to flag that
    `scale_pos_weight` leaves the probabilities uncalibrated (good for ranking, not for
    reading as literal PDs).

Run:  python -m evaluation.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")                      # headless: save figures, never open a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

from config import ROOT, load_config, resolve_path
from preprocessing.load import load_raw, split_X_y
from preprocessing.pipeline import make_split

BLUE, RED = "#4C78A8", "#E4572E"


# --------------------------------------------------------------------------- #
# Data / model loading
# --------------------------------------------------------------------------- #
def load_artifact(cfg: dict):
    bundle = joblib.load(resolve_path(cfg["artifacts"]["pipeline_file"]))
    return bundle["pipeline"], bundle["metadata"]


def recreate_splits(cfg: dict):
    """Deterministically recreate the same split used in training (seeded)."""
    df = load_raw()
    X, y = split_X_y(df)
    X_train, X_val, X_test, y_train, y_val, y_test = make_split(X, y, cfg)
    return X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def threshold_independent(y, proba) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "prevalence": float(np.mean(y)),
    }


def metrics_at(y, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cost_optimal_threshold(y, proba, cost_fn: float, cost_fp: float):
    """Threshold minimising expected business cost = cost_fn*FN + cost_fp*FP.

    Swept over candidate thresholds; returns (best_threshold, thresholds, costs).
    """
    y = np.asarray(y)
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        costs.append(cost_fn * fn + cost_fp * fp)
    costs = np.array(costs)
    best_t = float(thresholds[int(np.argmin(costs))])
    return best_t, thresholds, costs


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _fig_dir(cfg) -> Path:
    d = resolve_path(cfg["reports"]["figures_dir"]); d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"saved -> {path.relative_to(ROOT)}")


def plot_roc(y, proba, auc, out: Path):
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC curve — test set")
    ax.legend(loc="lower right")
    _save(fig, out)


def plot_pr(y, proba, ap, prevalence, out: Path):
    prec, rec, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(rec, prec, color=BLUE, lw=2, label=f"XGBoost (AP = {ap:.3f})")
    ax.axhline(prevalence, ls="--", color="gray", lw=1,
               label=f"baseline = prevalence ({prevalence:.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall curve — test set")
    ax.legend(loc="upper right")
    _save(fig, out)


def plot_confusion(m: dict, title: str, out: Path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["Repaid (0)", "Default (1)"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="Actual",
           title=f"{title}\n(threshold = {m['threshold']:.2f})")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{cm[i, j]/total:.1%}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _save(fig, out)


def plot_threshold_analysis(y, proba, cost_t, thresholds, costs, out: Path):
    prec, rec, thr = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(thr, prec[:-1], label="precision", color=BLUE)
    ax1.plot(thr, rec[:-1], label="recall", color=RED)
    ax1.plot(thr, f1[:-1], label="F1", color="#6A994E")
    ax1.axvline(cost_t, ls="--", color="black", lw=1, label=f"cost-opt = {cost_t:.2f}")
    ax1.set(xlabel="threshold", ylabel="score", title="Precision / Recall / F1 vs threshold")
    ax1.legend()
    ax2.plot(thresholds, costs / costs.min(), color="#8C6BAE")
    ax2.axvline(cost_t, ls="--", color="black", lw=1, label=f"min-cost threshold = {cost_t:.2f}")
    ax2.set(xlabel="threshold", ylabel="relative expected cost (1.0 = min)",
            title="Business cost vs threshold")
    ax2.legend()
    fig.tight_layout()
    _save(fig, out)


def plot_score_distribution(y, proba, out: Path):
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(proba[y == 0], bins=50, alpha=0.6, color=BLUE, label="Repaid (0)", density=True)
    ax.hist(proba[y == 1], bins=50, alpha=0.6, color=RED, label="Default (1)", density=True)
    ax.set(xlabel="predicted probability of default", ylabel="density",
           title="Predicted-score distribution by true class (test)\n"
                 "note: shifted high — scale_pos_weight leaves scores uncalibrated")
    ax.legend()
    _save(fig, out)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def evaluate(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    pipe, meta = load_artifact(cfg)
    X_val, y_val, X_test, y_test = recreate_splits(cfg)

    cost_fn = cfg["evaluation"]["cost_false_negative"]
    cost_fp = cfg["evaluation"]["cost_false_positive"]
    default_t = cfg["evaluation"]["default_threshold"]

    p_val = pipe.predict_proba(X_val)[:, 1]
    p_test = pipe.predict_proba(X_test)[:, 1]

    # Choose the operating threshold on VALIDATION (never on test).
    op_t, thresholds, val_costs = cost_optimal_threshold(y_val, p_val, cost_fn, cost_fp)

    ti = threshold_independent(y_test, p_test)
    at_op = metrics_at(y_test, p_test, op_t)
    at_half = metrics_at(y_test, p_test, default_t)
    _, _, test_costs = cost_optimal_threshold(y_test, p_test, cost_fn, cost_fp)

    print(f"TEST  n={len(y_test):,}  prevalence={ti['prevalence']:.4f}")
    print(f"  ROC-AUC={ti['roc_auc']:.4f}  PR-AUC={ti['pr_auc']:.4f}  Brier={ti['brier']:.4f}")
    print(f"  operating threshold (from val, cost {cost_fn}:{cost_fp}) = {op_t:.2f}")
    for tag, m in [("@0.50", at_half), ("@op", at_op)]:
        print(f"  {tag}: precision={m['precision']:.3f} recall={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")

    # Figures
    fd = _fig_dir(cfg)
    plot_roc(y_test, p_test, ti["roc_auc"], fd / "09_roc_curve.png")
    plot_pr(y_test, p_test, ti["pr_auc"], ti["prevalence"], fd / "10_pr_curve.png")
    plot_confusion(at_op, "Confusion matrix — cost-optimal", fd / "11_confusion_matrix.png")
    plot_threshold_analysis(y_test, p_test, op_t, thresholds, test_costs,
                            fd / "12_threshold_analysis.png")
    plot_score_distribution(y_test, p_test, fd / "13_score_distribution.png")

    results = {"threshold_independent": ti, "operating_threshold": op_t,
               "at_operating": at_op, "at_half": at_half, "costs": (cost_fn, cost_fp)}

    # Persist the chosen operating point so inference applies the same business
    # decision threshold (derived on validation) without recomputing it.
    op_path = resolve_path(cfg["artifacts"]["model_dir"]) / "operating_point.json"
    with op_path.open("w", encoding="utf-8") as fh:
        json.dump({"operating_threshold": op_t, "cost_false_negative": cost_fn,
                   "cost_false_positive": cost_fp, "chosen_on": "validation",
                   "test_roc_auc": ti["roc_auc"], "test_pr_auc": ti["pr_auc"]}, fh, indent=2)
    print(f"saved operating point -> {op_path.relative_to(ROOT)}")

    _write_report(results, meta, cfg)
    return results


def _write_report(r: dict, meta: dict, cfg: dict) -> None:
    ti, op, half = r["threshold_independent"], r["at_operating"], r["at_half"]
    cfn, cfp = r["costs"]
    lines = [
        "# Stage F — Model Evaluation (untouched test set)\n",
        f"Final model: **{meta['selected_model']}** (refit on {meta['refit_on']}). "
        f"Test set = 15% held out, scored once. Prevalence (default rate) = "
        f"{ti['prevalence']:.4f}.\n",
        "## Threshold-independent metrics (headline)\n",
        "| Metric | Value | Reading |",
        "|---|---|---|",
        f"| ROC-AUC | **{ti['roc_auc']:.4f}** | ranking quality across all thresholds |",
        f"| PR-AUC | **{ti['pr_auc']:.4f}** | precision/recall on the rare positive "
        f"class (baseline = {ti['prevalence']:.3f}) |",
        f"| Brier | {ti['brier']:.4f} | probability calibration (high → scores inflated "
        "by `scale_pos_weight`; fine for ranking, not literal PDs) |",
        "\nROC-AUC and PR-AUC are the fair summary because they don't depend on an "
        "operating point. PR-AUC well above the prevalence baseline shows real signal on "
        "the minority (default) class.\n",
        "## Operating point & the business trade-off\n",
        f"Two errors cost differently: a **false negative** (approve a defaulter) loses "
        f"principal; a **false positive** (reject a good applicant) loses interest margin. "
        f"Using illustrative relative costs **FN:FP = {cfn}:{cfp}**, the expected-cost-"
        f"minimising threshold — chosen on the validation set, then applied to test — is "
        f"**{op['threshold']:.2f}**.\n",
        "| Operating point | Threshold | Precision | Recall | F1 | TP | FP | FN | TN |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| Cost-optimal (FN:FP={cfn}:{cfp}) | {op['threshold']:.2f} | {op['precision']:.3f} "
        f"| {op['recall']:.3f} | {op['f1']:.3f} | {op['tp']} | {op['fp']} | {op['fn']} | {op['tn']} |",
        f"| Naive reference | {half['threshold']:.2f} | {half['precision']:.3f} | "
        f"{half['recall']:.3f} | {half['f1']:.3f} | {half['tp']} | {half['fp']} | {half['fn']} | {half['tn']} |",
        f"\nAt the cost-optimal threshold the model catches **{op['recall']:.0%} of "
        f"defaulters** (recall) at **{op['precision']:.0%} precision** — i.e. it prevents "
        f"{op['tp']:,} bad loans while wrongly declining {op['fp']:,} good applicants. "
        f"Because a false negative is judged {cfn}× costlier than a false positive, the "
        f"optimum leans toward recall: catching defaulters matters more than avoiding the "
        f"occasional wrongful decline. Raising the FN:FP ratio pushes the threshold lower "
        f"and recall higher (see `figures/12_threshold_analysis.png`); the right value is a "
        f"business policy decision, not a modelling one.\n",
        "## Figures\n",
        "- `figures/09_roc_curve.png` — ROC curve\n"
        "- `figures/10_pr_curve.png` — precision–recall curve\n"
        "- `figures/11_confusion_matrix.png` — confusion matrix at the cost-optimal threshold\n"
        "- `figures/12_threshold_analysis.png` — precision/recall/F1 and expected cost vs threshold\n"
        "- `figures/13_score_distribution.png` — score distribution by class (shows separation + inflation)\n",
        "## Caveats\n",
        "- Costs are **illustrative**; real values come from the lender's loss-given-default "
        "and margin economics.\n"
        "- Probabilities are **uncalibrated** (`scale_pos_weight`); if literal PDs are needed, "
        "add probability calibration (e.g. isotonic/Platt on a held-out fold) — a Phase-2 item.\n",
    ]
    out = resolve_path(cfg["reports"]["dir"]) / "evaluation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    evaluate()
