# Stage 1 — Data Quality Report

**Dataset:** Home Credit Default Risk — `data/raw/application_train.csv`
**Scope:** Phase 1 uses this single main table only. The 8 relational side-tables
(bureau, previous_application, POS_CASH, installments, credit_card, bureau_balance,
application_test) are deferred to Phase 2.
**Generated from:** one-off inspection of the raw CSV (read-only).

---

## 1. Shape & structural integrity

| Property | Value |
|---|---|
| Rows | 307,511 |
| Columns | 122 |
| Primary key | `SK_ID_CURR` — **unique** (307,511 distinct) |
| Duplicate rows | 0 |
| Constant columns | 0 |
| In-memory size | ~563 MB |

One row per loan application. Clean at the row level.

## 2. Schema / dtypes

| dtype | count | nature |
|---|---|---|
| `float64` | 65 | continuous amounts, `EXT_SOURCE_*` scores, housing block (`*_AVG/_MODE/_MEDI`) |
| `int64` | 41 | IDs, counts, `DAYS_*`, and many binary `FLAG_*` / region-mismatch indicators |
| `object` | 16 | categoricals (contract type, gender, education, occupation, org type, …) |

**Schema notes for later stages**
- ~40 `FLAG_DOCUMENT_*` and region-mismatch columns are **binary integers** — treat as
  categorical/binary, **do not scale**.
- The housing block comes in three correlated flavors (`_AVG`/`_MODE`/`_MEDI`) — highly
  redundant and heavily missing; candidate for reduction.

## 3. Target — class balance ✅

| TARGET | meaning | count | share |
|---|---|---|---|
| 0 | repaid | 282,686 | 91.93% |
| 1 | default | 24,825 | 8.07% |

- **Default rate ≈ 8.07%**, imbalance ratio **11.4 : 1**. Values strictly `{0, 1}`, none missing.
- **Target encoding confirmed and unchanged:** Home Credit's convention (`1` = client with
  payment difficulties = default; `0` = all other = repaid) already matches the project
  spec (`1 = default, 0 = repaid`). **No re-mapping was applied — only verified.**
- Consequences: stratified splits, `class_weight` / `scale_pos_weight`, and **ROC-AUC +
  PR-AUC / recall** as headline metrics rather than accuracy.

## 4. Missing values

- **67 / 122** columns have ≥1 missing value; **55** are complete.
- Missingness is concentrated, not uniform:

| Column(s) | % missing |
|---|---|
| COMMONAREA_{AVG,MODE,MEDI} | 69.9% |
| NONLIVINGAPARTMENTS_* | 69.4% |
| FONDKAPREMONT_MODE | 68.4% |
| LIVINGAPARTMENTS_* | 68.4% |
| FLOORSMIN_* | 67.9% |
| YEARS_BUILD_* | 66.5% |
| OWN_CAR_AGE | 66.0% |
| **EXT_SOURCE_1** | **56.4%** |
| EXT_SOURCE_3 | 19.8% |
| OCCUPATION_TYPE | 31.3% |

Buckets: 17 cols at 60–80% missing, 32 cols at 40–60%. The `EXT_SOURCE_*` external
credit scores are among the strongest known predictors, so their imputation + missing
indicators is a deliberate modeling decision, not an afterthought.

## 5. Data-quality anomalies (flagged; fixed in Stage 3 cleaning)

- **`DAYS_EMPLOYED == 365243`** (≈1000 years) in **55,374 rows (18.0%)** — a sentinel for
  "not employed / pensioner". Convert to `NaN` and add a boolean flag.
- **`CODE_GENDER == 'XNA'`** — 4 rows (missing-as-string) → treat as missing.
- **`NAME_FAMILY_STATUS == 'Unknown'`** — 2 rows → treat as missing.
- All other `DAYS_*` columns are correctly negative (measured backward from application date).
- Some categoricals encode missing as their own string (`FONDKAPREMONT_MODE`,
  `HOUSETYPE_MODE`, etc., ~50–68% NaN).

## 6. Categorical cardinality

| Column | # levels | note |
|---|---|---|
| ORGANIZATION_TYPE | 58 | high cardinality — group rare / frequency-encode |
| OCCUPATION_TYPE | 18 | plus 31% missing |
| NAME_INCOME_TYPE | 8 | |
| NAME_TYPE_SUITE | 7 | 1,292 missing |
| WALLSMATERIAL_MODE | 7 | 50.8% missing |
| others | 2–6 | one-hot friendly |

## 7. Summary

Structurally clean table (unique key, no duplicate rows, target already in the spec
convention `1 = default`). The real Phase-1 work is:
1. the ~8% **class imbalance**,
2. a large, redundant, heavily-missing **housing block**,
3. the **`DAYS_EMPLOYED = 365243`** and **`XNA` / `Unknown`** sentinels,
4. **high-cardinality categoricals** (`ORGANIZATION_TYPE`, `OCCUPATION_TYPE`).
