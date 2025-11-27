#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ========= EDIT THESE =========
DATASET_PATH = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet"

RECORD_ID =  304
NEW_SHA_SUCCESS = "2ec9f6bfd750182ce643fad233e0388effff4d15"
# ==============================


def main():
    df = pd.read_parquet(DATASET_PATH)

    if "id" not in df.columns:
        raise SystemExit("[ERROR] Dataset has no 'id' column.")

    mask = df["id"] == RECORD_ID
    if not mask.any():
        raise SystemExit(f"[ERROR] No row with id={RECORD_ID}")

    # Ensure the correct column exists
    if "sha_success" not in df.columns:
        df["sha_success"] = pd.NA

    old_value = df.loc[mask, "sha_success"].iloc[0]

    df.loc[mask, "sha_success"] = NEW_SHA_SUCCESS

    # Atomic save
    tmp = DATASET_PATH + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, DATASET_PATH)

    print(f"[INFO] Updated row id={RECORD_ID}")
    print(f"  - old sha_success: {old_value}")
    print(f"  - new sha_success: {NEW_SHA_SUCCESS}")


if __name__ == "__main__":
    main()
