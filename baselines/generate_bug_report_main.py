#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate / update fault localization bug reports from:
- main parquet dataset (lca_dataset.parquet)
- log_details.json (per-commit error context)
- FaultLocalization (LLM-based fault localization)
- changed_files_info (diff + workflow context)
"""

import os
import json
from typing import Any, Dict, List

import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv

from utilities.llm_provider import get_llm
from utilities.ensure_repo import ensure_repo_at_commit
from utilities.fetch_failed_commit_changed_files import (
    collect_changed_files_for_fail_and_parent,
)
from ci_repair.fault_localization import FaultLocalization


load_dotenv()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(config: OmegaConf) -> List[Dict[str, Any]]:
    """
    Load the main parquet dataset as a list of dicts.
    Assumes it is under <project_root>/dataset/lca_dataset.parquet
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    df = pd.read_parquet(dataset_path)
    return df.to_dict(orient="records")


def load_log_details(result_dir: str) -> List[Dict[str, Any]]:
    """
    Load log_details.json from the given result_dir.
    Expects a list of dicts, each with at least "sha_fail".
    """
    log_details_path = os.path.join(result_dir, "log_details.json")
    if not os.path.exists(log_details_path):
        raise FileNotFoundError(f"log_details.json not found at: {log_details_path}")

    with open(log_details_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("log_details.json is expected to be a list of entries")

    return data


def save_fault_localization(
    result_dir: str,
    new_report: Dict[str, Any],
    sha_fail: str,
) -> str:
    """
    Append/update the bug report (fault localization result) in fault_localization.json.

    Behavior:
    - Always uses the *canonical* sha_fail passed as argument.
    - If the file exists, remove any old entry with the same sha_fail and append the new one.
    - If not, create it with a single-element list.
    """
    fault_path = os.path.join(result_dir, "fault_localization.json")

    existing: List[Dict[str, Any]] = []
    if os.path.exists(fault_path):
        with open(fault_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
                if not isinstance(existing, list):
                    existing = []
            except json.JSONDecodeError:
                existing = []

    # Remove any existing entry for this sha_fail
    filtered = [e for e in existing if e.get("sha_fail") != sha_fail]

    # ALWAYS override sha_fail with the canonical value from the dataset / loop
    new_report["sha_fail"] = sha_fail

    filtered.append(new_report)

    with open(fault_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4)

    return fault_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ========= CONFIG & PATHS =========
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config: OmegaConf = OmegaConf.load(config_path)

    model_key = "gpt-5-mini"       # same as your other script
    log_analyzer_type = "bm25"     # folder name suffix (matches log_details.json location)

    # Results directory:
    # e.g. /Users/.../CI-REPAIR-BENCH/baselines/results/gpt-5-mini_bm25
    result_dir = os.path.join(
        config.project_result_dir,
        f"{model_key}_{log_analyzer_type}",
    )
    os.makedirs(result_dir, exist_ok=True)

    # Where to store changed files JSONs (per sha_fail), e.g. baselines/changed_files/
    changed_files_dir = config.changed_files_folder
    os.makedirs(changed_files_dir, exist_ok=True)

    # ========= SELECT WHICH IDs TO PROCESS =========
    # You can change this set like in your main.py
    # Example: process only specific IDs
    target_ids = {319}  # e.g. {241, 243, 281, 323}
    # If you want to process everything, set target_ids = None and adjust below.

    # ========= LOAD DATASET & SUBSET =========
    dataset = load_dataset(config)

    if target_ids is None:
        subset = dataset
    else:
        subset = [dp for dp in dataset if dp.get("id") in target_ids]

    if not subset:
        print(f"[MAIN] No datapoints found for ids: {target_ids}")
        raise SystemExit(0)

    # ========= LOAD LOG DETAILS ONCE =========
    log_details = load_log_details(result_dir)

    # Index by sha_fail for quick lookup
    log_by_sha: Dict[str, Dict[str, Any]] = {
        entry.get("sha_fail"): entry for entry in log_details if entry.get("sha_fail")
    }

    # ========= PREPARE LLM =========
    llm = get_llm(model_key)
    benchmark_owner: str = config.benchmark_owner

    # ========= PROCESS EACH DATAPOINT =========
    fault_localization_results: List[Dict[str, Any]] = []

    for dp in subset:
        task_id = dp["id"]
        sha_fail = dp["sha_fail"]
        repo_name = dp["repo_name"]
        repo_owner = dp["repo_owner"]  # kept for completeness (if needed later)
        workflow = dp["workflow"]
        workflow_path = dp["workflow_path"]

        print(f"\n[MAIN] Processing id={task_id}, sha_fail={sha_fail}")

        # 1) Check if we have log_details for this sha_fail
        log_entry = log_by_sha.get(sha_fail)
        if not log_entry:
            print(f"[MAIN] No log_details entry found for sha_fail={sha_fail}, skipping.")
            continue

        print(f"[MAIN] Found log_details entry for sha_fail={sha_fail}")

        # Optionally keep the *original* error_context from log_details
        original_error_context = log_entry.get("error_context")

        # 2) Ensure repo is cloned and at the failed commit
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)

        print(f"[MAIN] Ensuring repo {repo_name} is cloned and at {sha_fail}")
        ensure_repo_at_commit(repo_url, repo_path, sha_fail)

        # 3) Collect changed_files_info for sha_fail (+ parent)
        try:
            changed_files_info = collect_changed_files_for_fail_and_parent(
                owner=benchmark_owner,
                repo=repo_name,
                repo_path=repo_path,
                sha_fail=sha_fail,
                workflow_rel_path=workflow_path,
                workflow_yaml_from_dataset=workflow,
            )

            # Save per-commit changed files JSON
            changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
            with open(changed_files_path, "w", encoding="utf-8") as f:
                json.dump(changed_files_info, f, indent=4)

            print(f"[MAIN] Saved changed files info to {changed_files_path}")

        except Exception as e:
            print(f"[ERROR] Failed to collect/save changed files for {sha_fail}: {e}")
            continue

        # 4) FAULT LOCALIZATION (bug report), with error_logs + changed_files_info
        try:
            print(f"[MAIN] Running FaultLocalization for sha_fail={sha_fail}")
            fault_report = FaultLocalization(
                sha_fail=sha_fail,
                repo_path=repo_path,
                error_logs=log_entry,            # from log_details.json
                workflow=workflow,
                llm=llm,
                model_name=model_key,
                changed_files_info=changed_files_info,
            ).run()

            fault_localization_results.append(fault_report)

            # Save/merge into fault_localization.json
            out_path = save_fault_localization(result_dir, fault_report, sha_fail)
            print(f"[MAIN] Bug report (fault localization) saved/updated in: {out_path}")

            # If no suspicious files were found, you may skip patch generation later
            if not fault_report.get("fault_localization_data"):
                print(f"[MAIN] No suspicious files found for {sha_fail}, skipping patch.")
                continue

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during fault localization: {e}")
            continue

    print("\n[MAIN] Done.")
