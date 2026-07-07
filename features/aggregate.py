"""Phase 2 — side-table aggregation to one row per SK_ID_CURR.

Each side-table describes an applicant's prior credit history. We aggregate it to the
applicant grain (leakage-safe: per-SK_ID_CURR, no target, no cross-applicant stats) and
left-join onto the application table. Two-hop tables (e.g. bureau_balance) are aggregated
to their parent id first, then to the applicant.

Helpers keep the recipe consistent across tables; per-table functions live below and are
exposed via ``TABLE_AGGREGATORS``. Categorical aggregation uses grouped
``value_counts(normalize=True)`` (memory-light — avoids wide one-hot frames on the
tens-of-millions-of-rows balance tables).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUM_AGGS = ["min", "max", "mean", "sum", "std"]

# side-table name -> raw filename
TABLE_FILES = {
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def aggregate_numeric(df, group_col, prefix, num_aggs=NUM_AGGS, exclude=()):
    """Group numeric columns by ``group_col`` with a fixed set of agg funcs.

    Returns a frame indexed by ``group_col`` with columns ``PREFIX_COL_STAT``.
    """
    exclude = set(exclude) | {group_col}
    num_cols = [c for c in df.select_dtypes("number").columns if c not in exclude]
    g = df.groupby(group_col)[num_cols].agg(num_aggs)
    g.columns = [f"{prefix}_{col}_{stat.upper()}" for col, stat in g.columns]
    return g


def aggregate_categorical(df, group_col, cat_cols, prefix, top_n=None):
    """Per-group proportion of each category level (memory-light via value_counts).

    ``top_n`` caps each column to its most common levels (globally), lumping rare ones out
    — controls feature-count blow-up from high-cardinality columns (e.g. goods category).
    """
    parts = []
    for c in cat_cols:
        p = df.groupby(group_col)[c].value_counts(normalize=True).unstack(fill_value=0)
        if top_n is not None and p.shape[1] > top_n:
            keep = p.sum(axis=0).sort_values(ascending=False).index[:top_n]   # most common levels
            p = p[keep]
        p.columns = [f"{prefix}_{c}_{str(v).strip().replace(' ', '_')}_RATE" for v in p.columns]
        parts.append(p)
    return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df[group_col].unique())


def group_count(df, group_col, name):
    return df.groupby(group_col).size().rename(name).to_frame()


# --------------------------------------------------------------------------- #
# bureau (+ bureau_balance)
# --------------------------------------------------------------------------- #
def aggregate_bureau(raw_dir: Path) -> pd.DataFrame:
    """Aggregate bureau + bureau_balance to one row per SK_ID_CURR (prefix BURO/BB)."""
    bureau = pd.read_csv(raw_dir / TABLE_FILES["bureau"])
    bb = pd.read_csv(raw_dir / TABLE_FILES["bureau_balance"])

    # --- bureau_balance -> per SK_ID_BUREAU ---
    bb_months = aggregate_numeric(bb, "SK_ID_BUREAU", "BB", num_aggs=["min", "max"])
    bb_status = aggregate_categorical(bb, "SK_ID_BUREAU", ["STATUS"], "BB")
    bb_size = group_count(bb, "SK_ID_BUREAU", "BB_MONTHS_COUNT")
    bb_agg = pd.concat([bb_size, bb_months, bb_status], axis=1)
    del bb

    # attach per-credit balance features onto each bureau row
    bureau = bureau.merge(bb_agg, how="left", on="SK_ID_BUREAU")
    del bb_agg

    # active-credit count (a strong, interpretable signal) before dropping the column
    active = (bureau.assign(_active=(bureau["CREDIT_ACTIVE"] == "Active").astype("int8"))
              .groupby("SK_ID_CURR")["_active"].sum().rename("BURO_ACTIVE_COUNT").to_frame())

    # --- bureau (+ bb features) -> per SK_ID_CURR ---
    b = bureau.drop(columns=["SK_ID_BUREAU"])
    cnt = group_count(b, "SK_ID_CURR", "BURO_COUNT")
    num = aggregate_numeric(b, "SK_ID_CURR", "BURO")
    cat = aggregate_categorical(
        b, "SK_ID_CURR", ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"], "BURO")

    agg = pd.concat([cnt, active, num, cat], axis=1)
    agg.index.name = "SK_ID_CURR"
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# previous_application
# --------------------------------------------------------------------------- #
# DAYS_* columns in previous_application use 365243 as a "not applicable" sentinel
# (same quirk as the main table); neutralise before aggregating.
PREV_DAYS_SENTINEL_COLS = [
    "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE", "DAYS_TERMINATION",
]


def aggregate_previous_application(raw_dir: Path) -> pd.DataFrame:
    """Aggregate prior Home Credit applications to one row per SK_ID_CURR (prefix PREV)."""
    prev = pd.read_csv(raw_dir / TABLE_FILES["previous_application"])

    for c in PREV_DAYS_SENTINEL_COLS:
        if c in prev.columns:
            prev[c] = prev[c].replace(365243, np.nan)

    # how much was granted vs requested — a strong prior-behaviour signal
    prev["APP_CREDIT_RATIO"] = (prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]) \
        .replace([np.inf, -np.inf], np.nan)

    cat_cols = prev.select_dtypes("object").columns.tolist()
    p = prev.drop(columns=["SK_ID_PREV"])
    cnt = group_count(p, "SK_ID_CURR", "PREV_COUNT")
    num = aggregate_numeric(p, "SK_ID_CURR", "PREV")
    # cap high-cardinality categoricals (goods category, loan purpose, product combo, ...)
    # to their 6 most common levels to keep the feature count and memory in check
    cat = aggregate_categorical(p, "SK_ID_CURR", cat_cols, "PREV", top_n=6)

    agg = pd.concat([cnt, num, cat], axis=1)
    agg.index.name = "SK_ID_CURR"
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# installments_payments  (behavioural repayment history)
# --------------------------------------------------------------------------- #
def aggregate_installments(raw_dir: Path) -> pd.DataFrame:
    """Aggregate installment payment history to one row per SK_ID_CURR (prefix INS).

    Derives per-installment behaviour before aggregating: under/overpayment and how
    early/late each payment landed — the strongest repayment-discipline signals.
    """
    ins = pd.read_csv(raw_dir / TABLE_FILES["installments"])
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]       # +ve = underpaid
    ins["PAYMENT_RATIO"] = (ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]) \
        .replace([np.inf, -np.inf], np.nan)
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)  # days late
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)  # days early
    ins["LATE"] = (ins["DPD"] > 0).astype("int8")

    p = ins.drop(columns=["SK_ID_PREV"])
    cnt = group_count(p, "SK_ID_CURR", "INS_COUNT")
    num = aggregate_numeric(p, "SK_ID_CURR", "INS")
    agg = pd.concat([cnt, num], axis=1)
    agg.index.name = "SK_ID_CURR"
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# POS_CASH_balance  (monthly POS / cash-loan snapshots)
# --------------------------------------------------------------------------- #
def aggregate_pos_cash(raw_dir: Path) -> pd.DataFrame:
    """Aggregate POS/cash monthly balances to one row per SK_ID_CURR (prefix POS)."""
    pos = pd.read_csv(raw_dir / TABLE_FILES["pos_cash"])
    p = pos.drop(columns=["SK_ID_PREV"])
    cnt = group_count(p, "SK_ID_CURR", "POS_COUNT")
    num = aggregate_numeric(p, "SK_ID_CURR", "POS")               # incl. SK_DPD (days past due)
    cat = aggregate_categorical(p, "SK_ID_CURR", ["NAME_CONTRACT_STATUS"], "POS")
    agg = pd.concat([cnt, num, cat], axis=1)
    agg.index.name = "SK_ID_CURR"
    return agg.reset_index()


# --------------------------------------------------------------------------- #
# credit_card_balance  (monthly credit-card snapshots; only ~28% of applicants)
# --------------------------------------------------------------------------- #
def aggregate_credit_card(raw_dir: Path) -> pd.DataFrame:
    """Aggregate credit-card monthly balances to one row per SK_ID_CURR (prefix CC).

    Uses 4 agg funcs (not 5) since it has 23 columns — keeps the feature count in check.
    """
    cc = pd.read_csv(raw_dir / TABLE_FILES["credit_card"])
    p = cc.drop(columns=["SK_ID_PREV"])
    cnt = group_count(p, "SK_ID_CURR", "CC_COUNT")
    num = aggregate_numeric(p, "SK_ID_CURR", "CC", num_aggs=["min", "max", "mean", "sum"])
    cat = aggregate_categorical(p, "SK_ID_CURR", ["NAME_CONTRACT_STATUS"], "CC")
    agg = pd.concat([cnt, num, cat], axis=1)
    agg.index.name = "SK_ID_CURR"
    return agg.reset_index()


# Registry of implemented aggregators (grows over Phase-2 stages).
TABLE_AGGREGATORS = {
    "bureau": aggregate_bureau,
    "previous_application": aggregate_previous_application,
    "installments": aggregate_installments,
    "pos_cash": aggregate_pos_cash,
    "credit_card": aggregate_credit_card,
}
