"""Stage 8 — inference: score new/unseen applicants with the saved pipeline.

Loads the single joblib artifact (preprocessing + features + model, fit during training)
and scores raw application rows. It **never re-fits** — the same fitted transforms learned
on the training split are reused, which is the core anti-leakage guarantee (CLAUDE.md §7).

Input: a CSV/DataFrame with the same raw columns as `application_train.csv` (the `TARGET`
column is optional and ignored; `SK_ID_CURR`, if present, is carried through as an id).

Output: per-row probability of default and an approve/decline decision at the business
operating threshold chosen in Stage F (falls back to config `default_threshold`).

CLI:
    python -m inference.predict --input new_apps.csv --output scored.csv
    python -m inference.predict           # demo on a few held-out rows from the raw data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from config import ROOT, load_config, resolve_path
from models.scoring import default_proba
from features.build import build_feature_matrix

DECISION_REVIEW = "review/decline"
DECISION_APPROVE = "approve"


def load_scorer(cfg: dict | None = None):
    """Load the saved bundle (pipeline + optional calibrator) and operating threshold."""
    cfg = cfg or load_config()
    bundle = joblib.load(resolve_path(cfg["artifacts"]["pipeline_file"]))

    # Operating threshold from Stage F if available, else config default.
    op_path = resolve_path(cfg["artifacts"]["model_dir"]) / "operating_point.json"
    if op_path.exists():
        threshold = json.loads(op_path.read_text(encoding="utf-8"))["operating_threshold"]
    else:
        threshold = cfg["evaluation"]["default_threshold"]
    return bundle, float(threshold)


def score_frame(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Score raw application rows. Returns id (if any) + probability + decision.

    The input is passed through the saved pipeline unchanged (calibration applied if
    present); identifier/target columns are set aside so they don't interfere, then
    re-attached for output.
    """
    cfg = cfg or load_config()
    bundle, threshold = load_scorer(cfg)

    id_col, target = cfg["data"]["id_column"], cfg["data"]["target"]
    ids = df[id_col] if id_col in df.columns else pd.Series(range(len(df)), name=id_col)
    # Build the same wide matrix used in training (application + side-table aggregates,
    # joined by SK_ID_CURR). Applicants absent from the side tables get NaN = no history.
    wide = build_feature_matrix(df, cfg)
    features = wide.drop(columns=[c for c in (id_col, target) if c in wide.columns])

    proba = default_proba(bundle, features)
    out = pd.DataFrame({
        id_col: ids.values,
        "probability_default": proba,
        "decision": [DECISION_REVIEW if p >= threshold else DECISION_APPROVE for p in proba],
    })
    out.attrs["threshold"] = threshold
    return out


def demo(n: int = 5, cfg: dict | None = None) -> pd.DataFrame:
    """Score a few rows from the raw data (TARGET dropped) to show inference works."""
    from preprocessing.load import load_raw
    cfg = cfg or load_config()
    df = load_raw().drop(columns=[cfg["data"]["target"]]).head(n)
    scored = score_frame(df, cfg)
    print(f"Demo - scored {len(scored)} rows (operating threshold "
          f"{scored.attrs['threshold']:.2f}):")
    print(scored.to_string(index=False))
    return scored


def main():
    ap = argparse.ArgumentParser(description="Score applicants for probability of default.")
    ap.add_argument("--input", type=str, help="CSV of raw application rows to score")
    ap.add_argument("--output", type=str, help="where to write scored CSV")
    args = ap.parse_args()
    cfg = load_config()

    if not args.input:
        demo(cfg=cfg)
        return

    df = pd.read_csv(resolve_path(args.input))
    scored = score_frame(df, cfg)
    if args.output:
        out_path = resolve_path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(out_path, index=False)
        print(f"wrote {len(scored)} predictions -> {out_path}")
    else:
        print(scored.to_string(index=False))


if __name__ == "__main__":
    main()
