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


def process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm"):
    error_details = []
    fault_localization = []
    generated_patches = []

    # Folder to store one JSON per sha_fail with changed file info
    changed_files_dir = config.changed_files_folder

    # If your helper needs a token:

    # TEMP: only one datapoint
    subset = dataset[0:120]
    # target_ids = {241, 243, 281, 323}
    # subset = [dp for dp in dataset if dp.get("id") in target_ids]

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

        result_dir = os.path.join(
            config.project_result_dir, f"{model_key}_{log_analyzer_type}"
        )
        os.makedirs(result_dir, exist_ok=True)

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

            error_details.append(log_analysis_result)

            with open(os.path.join(result_dir, "log_details.json"), "w") as f:
                json.dump(error_details, f, indent=4)

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during error extraction: {e}")
            continue

        try:
            changed_files_info = collect_changed_files_for_fail_and_parent(
                owner=benchmark_owner,
                repo=repo_name,
                repo_path=repo_path,
                sha_fail=sha_fail,
                workflow_rel_path=workflow_path,
                workflow_yaml_from_dataset=workflow
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
        # 3) FAULT LOCALIZATION (later we can inject changed_files_info)
        # ------------------------------------------------------------------
        try:
            fault_localizer = FaultLocalization(
                sha_fail=sha_fail,
                repo_path=repo_path,
                error_logs=log_analysis_result,
                workflow=workflow,
                llm=llm,
                model_name=model_key,
                changed_files_info=changed_files_info,  # add to __init__ later if you want
            ).run()

            fault_localization.append(fault_localizer)

            with open(os.path.join(result_dir, "fault_localization.json"), "w") as f:
                json.dump(fault_localization, f, indent=4)

            if not fault_localizer.get("fault_localization_data"):
                print(f"[MAIN] No suspicious files found for {sha_fail}, skipping...")
                continue

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during fault localization: {e}")
            continue

        # # ------------------------------------------------------------------
        # # 4) PATCH GENERATION
        # # ------------------------------------------------------------------
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

            if not patch_generator.get("diff"):
                print(f"[MAIN] No patch generated for {sha_fail}")
                continue

            generated_patches.append(patch_generator)

            with open(os.path.join(result_dir, "generated_patches.json"), "w") as f:
                json.dump(generated_patches, f, indent=4)

        except Exception as e:
            print(f"[ERROR] Failed processing {sha_fail} during patch generation: {e}")
            continue

    return generated_patches


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")

    model_key = "gpt-5-mini"   # or "gpt4o", "deepseek-chat", etc.
    llm = get_llm(model_key)

    # Load dataset
    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    results = process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm")

    output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[MAIN] Results saved in {output_file}")
