#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import json
import subprocess
from omegaconf import OmegaConf
from dotenv import load_dotenv
from datasets import load_dataset  # (unused right now but kept)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from utilities.fetch_failed_commit_changed_files import (
    collect_changed_files_for_fail_and_parent,
)
from utilities.llm_provider import get_llm
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()

# ----------------------------------------------------------------------
# Helper functions: load + ordered save (by dataset sha_fail order)
# ----------------------------------------------------------------------
def _load_results_index(path: str, key_field: str = "sha_fail") -> dict:
    """
    Load an existing JSON file that contains a list of dicts
    and build an index { key_field: record }.

    If file does not exist or is invalid, returns an empty dict.
    """
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # If something is wrong with the file, don't crash the whole run
        return {}

    if not isinstance(data, list):
        return {}

    index = {}
    for item in data:
        if isinstance(item, dict) and key_field in item:
            index[item[key_field]] = item
    return index


def _save_results_index_ordered(
    path: str,
    index: dict,
    dataset: list,
    key_field: str = "sha_fail",
) -> None:
    """
    Save an index { key_field: record } to a JSON file as a list,
    ordered according to the dataset's order of `key_field` (sha_fail).

    - dataset is the full dataset (or same ordering you want to follow)
    - index holds { sha_fail: record }
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Order keys based on dataset order
    sha_order = []
    seen = set()
    for dp in dataset:
        sha = dp.get(key_field)
        if sha and sha not in seen:
            sha_order.append(sha)
            seen.add(sha)

    ordered = []
    # First, in dataset order
    for sha in sha_order:
        if sha in index:
            ordered.append(index[sha])

    # If there are any extra keys not in the dataset (unlikely, but safe):
    for sha, rec in index.items():
        if sha not in seen:
            ordered.append(rec)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=4)


def process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm"):
    """
    Process the dataset incrementally.

    - Before saving, we always:
        1) Load existing data from JSON
        2) Keep everything keyed by sha_fail
        3) When saving, order by dataset's sha_fail order
        4) If a sha_fail already exists, replace it with the new one
    """

    # Folder to store one JSON per sha_fail with changed file info
    changed_files_dir = config.changed_files_folder
    os.makedirs(changed_files_dir, exist_ok=True)

    # Per-model result directory
    result_dir = os.path.join(config.project_result_dir, f"{model_key}_{log_analyzer_type}")
    os.makedirs(result_dir, exist_ok=True)

    # Paths for incremental result files
    log_details_path = os.path.join(result_dir, "log_details.json")
    fault_loc_path = os.path.join(result_dir, "fault_localization.json")
    patches_path = os.path.join(result_dir, "generated_patches.json")

    # Load existing results as {sha_fail: record}
    log_details_index = _load_results_index(log_details_path, key_field="sha_fail")
    fault_loc_index = _load_results_index(fault_loc_path, key_field="sha_fail")
    patches_index = _load_results_index(patches_path, key_field="sha_fail")

    # Use the full dataset order as the canonical order
    # If you want a subset for processing, change here:
    subset = dataset[144:]  # or dataset[start:end], etc.

    for datapoint in subset:
        task_id = datapoint["id"]
        repo_name = datapoint["repo_name"]
        repo_owner = datapoint["repo_owner"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)
        head_branch = datapoint["head_branch"]
        sha_fail = datapoint["sha_fail"]
        benchmark_owner = config.benchmark_owner
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]
        sha_success = datapoint["sha_success"]

        print(f"\n[MAIN] Proceeding with failed commit: {sha_fail}")

        # Make sure repo is cloned and at sha_fail
        ensure_repo_at_commit(repo_url, repo_path, sha_fail)

        # ------------------------------------------------------------------
        # 1) CI LOG ANALYSIS
        # ------------------------------------------------------------------
        try:
            if log_analyzer_type == "llm":
                log_analysis_result = CILogAnalyzerLLM(
                    repo_path,
                    logs,
                    sha_fail,
                    workflow,
                    workflow_path,
                    llm=llm,
                    model_name=model_key,
                ).run()
            else:
                log_analysis_result = CILogAnalyzerBM25(
                    repo_path,
                    logs,
                    sha_fail,
                    workflow,
                    workflow_path,
                    llm=llm,
                    model_name=model_key,
                ).run()

            if isinstance(log_analysis_result, dict):
                log_analysis_result.setdefault("sha_fail", sha_fail)
                # Replace old entry (if any) with new one
                log_details_index[sha_fail] = log_analysis_result
                # Save in dataset order
                _save_results_index_ordered(
                    log_details_path, log_details_index, dataset, key_field="sha_fail"
                )

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during error extraction: {e}")
            continue

        # ------------------------------------------------------------------
        # 2) CHANGED FILES COLLECTION
        # ------------------------------------------------------------------
        try:
            changed_files_info = collect_changed_files_for_fail_and_parent(
                owner=benchmark_owner,
                repo=repo_name,
                repo_path=repo_path,
                sha_fail=sha_fail,
                workflow_rel_path=workflow_path,
                workflow_yaml_from_dataset=workflow,
            )

            # Save to per-commit JSON file in baselines/changed_files/{sha_fail}.json
            changed_files_path = os.path.join(changed_files_dir, f"{sha_fail}.json")
            with open(changed_files_path, "w", encoding="utf-8") as f:
                json.dump(changed_files_info, f, indent=4)

            print(f"[MAIN] Saved changed files info to {changed_files_path}")

        except Exception as e:
            print(f"[ERROR] Failed to collect/save changed files for {sha_fail}: {e}")
            continue

        # ------------------------------------------------------------------
        # 3) FAULT LOCALIZATION
        # ------------------------------------------------------------------
        try:
            fault_localizer = FaultLocalization(
                sha_fail=sha_fail,
                repo_path=repo_path,
                error_logs=log_analysis_result,
                workflow=workflow,
                llm=llm,
                model_name=model_key,
                changed_files_info=changed_files_info,
            ).run()

            if isinstance(fault_localizer, dict):
                fault_localizer.setdefault("sha_fail", sha_fail)
                fault_loc_index[sha_fail] = fault_localizer
                _save_results_index_ordered(
                    fault_loc_path, fault_loc_index, dataset, key_field="sha_fail"
                )

            if not fault_localizer.get("fault_localization_data"):
                print(f"[MAIN] No suspicious files found for {sha_fail}, skipping patch generation...")
                continue

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during fault localization: {e}")
            continue

        # ------------------------------------------------------------------
        # 4) PATCH GENERATION
        # ------------------------------------------------------------------
        try:
            patch_generator = PatchGeneration(
                bug_report=fault_localizer,
                repo_path=repo_path,
                task_id=task_id,
                error_details=log_analysis_result,
                workflow_path=workflow_path,
                workflow=workflow,
                llm=llm,
                model_name=model_key,
            ).run()

            if not isinstance(patch_generator, dict):
                print(f"[MAIN] Patch generator returned non-dict for {sha_fail}, skipping...")
                continue

            if not patch_generator.get("diff"):
                print(f"[MAIN] No patch generated for {sha_fail}")
                # Keep behavior consistent with your original code:
                # don't store entries without a diff in generated_patches.json
                continue

            patch_generator.setdefault("sha_fail", sha_fail)
            patches_index[sha_fail] = patch_generator
            _save_results_index_ordered(
                patches_path, patches_index, dataset, key_field="sha_fail"
            )

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during patch generation: {e}")
            continue

    # Final in-memory list of generated patches (deduped, ordered)
    # NOTE: they are still ordered if we rebuild using dataset
    ordered_patches = []
    sha_order = [dp.get("sha_fail") for dp in dataset if dp.get("sha_fail")]
    seen = set()
    for sha in sha_order:
        if sha in patches_index and sha not in seen:
            ordered_patches.append(patches_index[sha])
            seen.add(sha)
    for sha, rec in patches_index.items():
        if sha not in seen:
            ordered_patches.append(rec)

    return ordered_patches


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")

    model_key = "gpt-5-mini"   # or "gpt4o", "deepseek-chat", etc.
    llm = get_llm(model_key)

    # Load dataset (this is the canonical order)
    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    # Run processing (incrementally updates per-model JSON files)
    results = process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm")

    # Optional: also maintain a global generated_patches.json
    # in config.project_result_dir, with same ordered-by-dataset behavior
    global_output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    global_index = _load_results_index(global_output_file, key_field="sha_fail")
    for item in results:
        if isinstance(item, dict) and "sha_fail" in item:
            global_index[item["sha_fail"]] = item
    _save_results_index_ordered(
        global_output_file, global_index, dataset, key_field="sha_fail"
    )

    print(f"[MAIN] Global results saved in {global_output_file}")
