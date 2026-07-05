"""Stage 7 — model interpretation (feature importance + SHAP).

Explains the final XGBoost model both globally (which features drive default risk) and
directionally (does a higher value raise or lower risk), then translates the top drivers
into plain-language business insight for a non-technical stakeholder.

Approach:
  * Transform the untouched test set through the saved preprocessing (clean -> engineer ->
    encode) to get the exact 268-column matrix the model sees, with readable names.
  * SHAP `TreeExplainer` on a sample: mean|SHAP| = global importance; the sign of the
    (feature value, SHAP value) correlation = direction (↑ or ↓ risk).
  * Cross-check with XGBoost's own gain importance.

Run:  python -m evaluation.interpret
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from config import ROOT, load_config, resolve_path
from evaluation.evaluate import recreate_splits

SAMPLE_SIZE = 5000            # rows for SHAP (TreeExplainer is exact & fast at this size)


def clean_name(name: str) -> str:
    """Make ColumnTransformer output names human-readable."""
    n = name.replace("num__", "").replace("cat__", "")
    n = n.replace("missingindicator_", "MISSING:")
    return n


def load_pipeline(cfg):
    bundle = joblib.load(resolve_path(cfg["artifacts"]["pipeline_file"]))
    return bundle["pipeline"], bundle["metadata"]


def _fig_dir(cfg) -> Path:
    d = resolve_path(cfg["reports"]["figures_dir"]); d.mkdir(parents=True, exist_ok=True)
    return d


def interpret(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    seed = cfg["seed"]
    pipe, meta = load_pipeline(cfg)
    _, _, X_test, y_test = recreate_splits(cfg)

    # Transform to the model's input space; keep readable feature names.
    pre = pipe[:-1]                                   # clean -> engineer -> prep
    model = pipe.named_steps["model"]
    feat_names = [clean_name(n) for n in pipe.named_steps["prep"].get_feature_names_out()]
    X_model = pre.transform(X_test)

    # Sample for SHAP
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_model), size=min(SAMPLE_SIZE, len(X_model)), replace=False)
    Xs = X_model[idx]

    # Exact TreeSHAP via XGBoost natively (avoids a shap<->xgboost version-parsing
    # incompatibility with TreeExplainer). Returns per-feature contributions in margin
    # (log-odds) space; the trailing column is the base value, which we drop.
    contribs = model.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)
    shap_values = contribs[:, :-1]                    # (n_sample, n_features)

    mean_abs = np.abs(shap_values).mean(axis=0)
    # Direction: correlation between feature value and its SHAP value across the sample.
    direction = np.array([
        np.corrcoef(Xs[:, j], shap_values[:, j])[0, 1] if np.std(Xs[:, j]) > 0 else 0.0
        for j in range(Xs.shape[1])
    ])
    imp = (pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs,
                         "dir_corr": direction})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp["direction"] = np.where(imp["dir_corr"] >= 0, "up", "down")   # ascii-safe for console

    # XGBoost gain importance (cross-check), aligned to the same feature order
    gain = model.feature_importances_
    gain_rank = (pd.DataFrame({"feature": feat_names, "gain": gain})
                 .sort_values("gain", ascending=False).reset_index(drop=True))

    print("Top 15 drivers by mean|SHAP|:")
    print(imp.head(15).to_string(index=False))

    # --- Figures --------------------------------------------------------------
    fd = _fig_dir(cfg)

    shap.summary_plot(shap_values, Xs, feature_names=feat_names, plot_type="bar",
                      max_display=20, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 8)
    plt.title("Global feature importance (mean |SHAP|)")
    fig.savefig(fd / "14_shap_importance.png", bbox_inches="tight", dpi=130); plt.close(fig)
    print(f"saved -> {(fd / '14_shap_importance.png').relative_to(ROOT)}")

    shap.summary_plot(shap_values, Xs, feature_names=feat_names, max_display=20, show=False)
    fig = plt.gcf(); fig.set_size_inches(9, 8)
    plt.title("SHAP summary — impact & direction per feature")
    fig.savefig(fd / "15_shap_beeswarm.png", bbox_inches="tight", dpi=130); plt.close(fig)
    print(f"saved -> {(fd / '15_shap_beeswarm.png').relative_to(ROOT)}")

    # Dependence plots for the top continuous drivers (skip MISSING indicators)
    top_cont = [f for f in imp["feature"].tolist() if not f.startswith("MISSING:")][:4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, feat in zip(axes.ravel(), top_cont):
        j = feat_names.index(feat)
        shap.dependence_plot(j, shap_values, Xs, feature_names=feat_names,
                             interaction_index=None, ax=ax, show=False)
        ax.set_title(feat)
    fig.suptitle("SHAP dependence — how risk varies with the top drivers", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fd / "16_shap_dependence.png", bbox_inches="tight", dpi=130); plt.close(fig)
    print(f"saved -> {(fd / '16_shap_dependence.png').relative_to(ROOT)}")

    _write_report(imp, gain_rank, meta, len(idx), cfg)
    return {"importance": imp, "gain": gain_rank}


# Plain-language meaning for the features we expect near the top (business translation).
FEATURE_MEANING = {
    "EXT_SOURCE_2": "External credit score #2 — a bureau-style risk score; higher = safer.",
    "EXT_SOURCE_3": "External credit score #3 — higher = safer.",
    "EXT_SOURCE_1": "External credit score #1 — higher = safer.",
    "CREDIT_TERM": "Annuity ÷ credit (repayment intensity; proxy for shorter term).",
    "CREDIT_INCOME_RATIO": "Loan size ÷ income — loan burden / over-leverage.",
    "ANNUITY_INCOME_RATIO": "Instalment ÷ income — debt-to-income (payment strain).",
    "DAYS_BIRTH": "Age (stored as negative days) — younger applicants default more.",
    "DAYS_EMPLOYED": "Employment length (negative days) — longer tenure = safer.",
    "EMPLOYED_TO_AGE": "Share of life employed — job/career stability.",
    "DAYS_ID_PUBLISH": "Days since ID document updated — behavioural/stability signal.",
    "DAYS_REGISTRATION": "Days since registration change — stability signal.",
    "AMT_ANNUITY": "Loan instalment amount.",
    "AMT_CREDIT": "Loan amount.",
    "GOODS_CREDIT_RATIO": "Goods price ÷ credit — collateral coverage / down-payment proxy.",
    "INCOME_PER_PERSON": "Per-capita household income.",
    "DAYS_LAST_PHONE_CHANGE": "Days since phone number changed — stability signal.",
    "AMT_GOODS_PRICE": "Price of the goods the loan finances.",
    "OWN_CAR_AGE": "Age of the applicant's car (missing when no car).",
    "NAME_EDUCATION_TYPE_Higher education": "Has higher education — lowers risk.",
    "NAME_FAMILY_STATUS_Married": "Married — associated with lower risk.",
    "CODE_GENDER_F": "Gender = Female (see fairness caveat).",
    "CODE_GENDER_M": "Gender = Male (see fairness caveat).",
}


def _write_report(imp: pd.DataFrame, gain: pd.DataFrame, meta: dict, n_sample: int, cfg: dict):
    top = imp.head(15)
    lines = [
        "# Stage G — Model Interpretation (SHAP)\n",
        f"Final model: **{meta['selected_model']}**. Global explanations via **TreeSHAP** "
        f"(computed natively by XGBoost) on a {n_sample:,}-row test sample. mean|SHAP| = "
        "importance; the direction column is the sign of the (feature value → SHAP value) "
        "correlation.\n",
        "## Top 15 default-risk drivers\n",
        "| # | Feature | mean\\|SHAP\\| | Direction | Plain-language meaning |",
        "|---|---|---|---|---|",
    ]
    for i, row in top.iterrows():
        base = row["feature"].replace("MISSING:", "")
        meaning = FEATURE_MEANING.get(base, "—")
        if row["feature"].startswith("MISSING:"):
            meaning = f"Whether **{base}** was missing (missingness itself carries signal)."
        arrow = "↑ risk" if row["direction"] == "up" else "↓ risk"
        lines.append(f"| {i+1} | `{row['feature']}` | {row['mean_abs_shap']:.4f} | "
                     f"{arrow} | {meaning} |")
    lines += [
        "\n## What drives default (plain language)\n",
        "1. **External credit scores dominate.** `EXT_SOURCE_3/2/1` are the top signals by a "
        "wide margin (mean|SHAP| ~0.12-0.36 vs <0.09 for everything else) — higher scores → "
        "lower predicted default. They summarise external bureau information the application "
        "form can't.\n"
        "2. **Engineered loan-structure ratios pay off.** `GOODS_CREDIT_RATIO` (#3) and "
        "`CREDIT_TERM` (#4) — both built in Stage D — rank above every raw feature except the "
        "external scores. Higher goods-to-credit (more of the loan backed by the purchased "
        "good, i.e. a larger down-payment) and higher annuity-to-credit (shorter effective "
        "term) both **lower** risk. This directly validates the feature-engineering step.\n"
        "3. **Age & employment stability.** Shorter employment (`DAYS_EMPLOYED`, #6) and "
        "younger applicants (`DAYS_BIRTH`, #9) default more; larger instalments "
        "(`AMT_ANNUITY`, #7) raise risk — life/job stability and affordable payments lower it.\n"
        "4. **Demographics & behaviour.** Higher education (#8) and being married (#15) lower "
        "risk; stability signals like `DAYS_ID_PUBLISH` (#12) and `OWN_CAR_AGE` (#13) "
        "contribute. `CODE_GENDER` appears (#10-11) — flagged under fairness below.\n"
        "5. **Missingness carries signal.** `OWN_CAR_AGE` is missing when the applicant has no "
        "car, and such flags surface among the drivers — validating the Stage-C decision to "
        "flag missing values rather than silently impute.\n",
        "## Figures\n",
        "- `figures/14_shap_importance.png` — global importance (mean |SHAP|)\n"
        "- `figures/15_shap_beeswarm.png` — per-feature impact **and** direction\n"
        "- `figures/16_shap_dependence.png` — how risk varies across the top drivers\n",
        "## Caveats\n",
        "- SHAP shows the **model's** associations, **not causation**.\n"
        "- Engineered ratios share information with their source amounts, so importance can "
        "spread across related features — read them as a group, not in isolation.\n"
        "- Fairness: `CODE_GENDER` and similar attributes are present; a production system "
        "should review protected-attribute usage and disparate impact (Phase-2 item).\n",
    ]
    out = resolve_path(cfg["reports"]["dir"]) / "interpretation.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote report -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    interpret()
