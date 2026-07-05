"""End-to-end runner for the credit-risk pipeline.

Runs the whole thing from the raw CSV to a saved, evaluated, interpreted, and
demonstrably deployable model — all driven by `config/config.yaml` and a single seed.

Stages:
    1. Data understanding  — shape / target balance / missingness overview.
    2. Train + select      — LR / RF / XGBoost, bounded tuning, save artifact (Stage E).
    3. Evaluate            — untouched test metrics, curves, business-cost threshold (F).
    4. Interpret           — SHAP importance + drivers (Stage G).
    5. Sample inference    — score a few rows via the saved artifact, no re-fit (Stage H).

Usage:
    python run_pipeline.py                # full run (training takes a few minutes)
    python run_pipeline.py --skip-train   # reuse the existing artifact, just re-eval/interpret
"""
from __future__ import annotations

import argparse

from config import load_config, resolve_path
from preprocessing.load import load_raw, split_X_y, quick_overview


def _banner(text: str):
    print("\n" + "=" * 74 + f"\n{text}\n" + "=" * 74)


def main():
    ap = argparse.ArgumentParser(description="Run the credit-risk pipeline end to end.")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse the existing saved artifact instead of retraining")
    args = ap.parse_args()
    cfg = load_config()

    _banner("STAGE 1 - Data understanding")
    df = load_raw()
    ov = quick_overview(df)
    print(f"shape           : {ov['n_rows']:,} x {ov['n_cols']}")
    print(f"target balance  : {ov['target_counts']}  (default rate {ov['default_rate']:.4f})")
    print(f"cols w/ missing : {ov['n_cols_with_missing']} / {ov['n_cols']}")

    _banner("STAGE 2 - Train & select model")
    artifact = resolve_path(cfg["artifacts"]["pipeline_file"])
    if args.skip_train and artifact.exists():
        print(f"--skip-train: reusing existing artifact at {artifact.name}")
    else:
        from models.train import train_and_select
        train_and_select(cfg)

    _banner("STAGE 3 - Evaluate on untouched test set")
    from evaluation.evaluate import evaluate
    evaluate(cfg)

    _banner("STAGE 4 - Interpret (SHAP)")
    from evaluation.interpret import interpret
    interpret(cfg)

    _banner("STAGE 5 - Sample inference from saved artifact")
    from inference.predict import demo
    demo(n=5, cfg=cfg)

    _banner("DONE - artifacts in models/artifacts/, figures & reports in reports/")


if __name__ == "__main__":
    main()
