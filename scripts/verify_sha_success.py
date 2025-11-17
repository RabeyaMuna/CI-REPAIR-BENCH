#!/usr/bin/env python3
"""
verify_sha_success.py

Goal:
- Find all rows where `sha_success` is *not actually a successful commit*.
- Print those rows (`id`, repo, sha_fail, bad sha_success, status).
- For each, find the *actual successful* commit for that (repo_owner, repo_name, sha_fail)
  from within the same dataset and print it.

Dataset path (hardcoded):
    /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet
"""

import logging
from pathlib import Path
from typing import Optional, Any
from datetime import datetime, date

import pandas as pd

# ========= HARD-CODED DATASET PATH =========
DATASET_PATH = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet"
)

# ========= CONFIG / HEURISTICS =========

# Columns that might store the CI conclusion for the *success run*.
# Adjust these to match your lca_dataset schema if needed.
SUCCESS_CONCLUSION_CANDIDATES = [
    "conclusion_success",
    "success_conclusion",
    "run_conclusion_success",
    "workflow_conclusion_success",
    "ci_conclusion_success",
    "success_status",
    "status_success",
    # Often datasets just use a generic conclusion column for the success commit:
    "conclusion",
]

# Values that count as "successful"
SUCCESS_VALUES = {"success", "succeeded", "passed", "pass", "ok", "green"}

# Values that clearly mean "not successful"
FAILURE_KEYWORDS = [
    "failure",
    "failed",
    "cancelled",
    "canceled",
    "timed_out",
    "timeout",
    "error",
]

# Columns used to define ordering (so we can find the "next" success)
ORDER_COLUMNS = [
    "success_run_started_at",
    "success_run_created_at",
    "run_started_at_success",
    "run_number_success",
    "id",  # fallback; your dataset usually has this
]


# ========= HELPERS =========

def infer_success_conclusion(row: pd.Series) -> Optional[bool]:
    """
    Try to infer whether this row's sha_success is actually successful,
    based on any conclusion-like column we can find.

    Returns:
        True  -> clearly successful
        False -> clearly not successful
        None  -> unknown / cannot determine
    """
    for col in SUCCESS_CONCLUSION_CANDIDATES:
        if col in row and pd.notna(row[col]):
            val = str(row[col]).strip().lower()
            if not val:
                continue

            if val in SUCCESS_VALUES:
                return True

            if any(k in val for k in FAILURE_KEYWORDS):
                return False

    # Nothing useful found
    return None


def get_order_value(row: pd.Series) -> Any:
    """
    Return a sortable value for 'time/sequence' ordering of rows.
    Tries several ORDER_COLUMNS; falls back to the row index.

    Uses explicit try/except around datetime parsing to avoid
    the deprecated errors="ignore" behavior.
    """
    for col in ORDER_COLUMNS:
        if col in row and pd.notna(row[col]):
            val = row[col]

            # If it's already a nice type, just use it
            if isinstance(val, (pd.Timestamp, datetime, date, int, float)):
                return val

            # Otherwise try to parse as datetime
            try:
                parsed = pd.to_datetime(val)  # no errors="ignore"
                return parsed
            except Exception:
                # If parsing fails, just use the raw value
                return val

    # Fallback if none of the ORDER_COLUMNS exist or are populated
    return row.name


def find_next_success_for_fail(
    df: pd.DataFrame,
    base_row: pd.Series,
) -> Optional[str]:
    """
    For a given row (with sha_fail, sha_success), find the *next* successful commit
    for the same (repo_owner, repo_name, sha_fail).

    We assume df already has:
        - df["_is_sha_success_successful"]
        - df["_order_value"]

    Returns:
        sha_success string of the next successful row, or None if not found.
    """
    repo_owner = base_row["repo_owner"]
    repo_name = base_row["repo_name"]
    sha_fail = base_row["sha_fail"]
    current_order = base_row["_order_value"]

    mask_same = (
        (df["repo_owner"] == repo_owner)
        & (df["repo_name"] == repo_name)
        & (df["sha_fail"] == sha_fail)
    )

    # Candidates: same failing commit, with clearly successful sha_success,
    # and strictly later in order than the current row.
    candidates = df.loc[
        mask_same
        & (df["_is_sha_success_successful"] == True)  # noqa: E712
        & (df["_order_value"] > current_order)
    ]

    if candidates.empty:
        return None

    candidates_sorted = candidates.sort_values("_order_value", ascending=True)
    best_row = candidates_sorted.iloc[0]
    return best_row["sha_success"]


# ========= MAIN =========

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")

    logging.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_parquet(DATASET_PATH)

    required_cols = {"id", "repo_owner", "repo_name", "sha_fail", "sha_success"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    logging.info("Computing success flags and order values for all rows...")
    df["_order_value"] = df.apply(get_order_value, axis=1)
    df["_is_sha_success_successful"] = df.apply(infer_success_conclusion, axis=1)

    # Filter rows where sha_success is NOT clearly successful (False or None)
    problematic_mask = df["_is_sha_success_successful"] != True  # noqa: E712
    problematic = df.loc[problematic_mask].copy()

    logging.info(
        "Found %d rows where sha_success is not clearly successful.",
        len(problematic),
    )

    if problematic.empty:
        print("No rows found where sha_success is non-successful or unknown.")
        return

    print(
        "id, repo, sha_fail, original_sha_success, status(is_sha_success_successful), actual_success_sha"
    )

    # For each problematic row, try to find the actual successful commit
    for idx, row in problematic.iterrows():
        actual_success_sha = find_next_success_for_fail(df, row)
        status = row["_is_sha_success_successful"]
        if actual_success_sha is None:
            actual_success_sha = "NOT FOUND"

        print(
            f"{row['id']}, "
            f"{row['repo_owner']}/{row['repo_name']}, "
            f"{row['sha_fail']}, "
            f"{row['sha_success']}, "
            f"{status}, "
            f"{actual_success_sha}"
        )

    logging.info("Done printing problematic sha_success entries.")


if __name__ == "__main__":
    main()
