#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import shutil
from omegaconf import OmegaConf
from benchmark_utils import save_jsonl
from benhmark_functions import get_results
from benchmark import CIFixBenchmark

# --- Helper Functions ---
def read_jsonl(path):
    """Safely read a JSONL file, skipping malformed lines."""
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping invalid JSON at line {i} in {path}: {e}")
    return data

def save_overwrite(path, records):
    """Overwrite JSONL at path with records (no merge)."""
    # ensure parent dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_jsonl(path, records)

# --- Load main config ---
CONFIG_PATH = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/config.yaml"
config = OmegaConf.load(CONFIG_PATH)
base_dir = config.get("base_dir")

# --- Paths (SOURCE OF TRUTH + outputs) ---
results_dir       = os.path.join(base_dir, "results")
jobs_pushed_file  = os.path.join(results_dir, "jobs_ids_diff.jsonl")      # <— ONLY THIS IS READ
awaiting_file     = os.path.join(results_dir, "jobs_awaiting_diff.jsonl")

success_file      = os.path.join(results_dir, "jobs_success_diff.jsonl")
failure_file      = os.path.join(results_dir, "jobs_failure_diff.jsonl")
errors_file       = os.path.join(results_dir, "jobs_error_diff.jsonl")

results_file      = os.path.join(results_dir, "jobs_results_diff.jsonl")  # success + failure only

config_path       = os.path.join(base_dir, "config.yaml")

# --- Initialize benchmark environment ---
model_name = "diff"
bench = CIFixBenchmark(model_name, config_path)

print("\n===============================")
print(" Rechecking pushed CI jobs (ONLY jobs_ids_diff.jsonl)...")
print("===============================\n")

# --- Load pushed jobs (single source of truth) ---
pushed_jobs = read_jsonl(jobs_pushed_file)
if not pushed_jobs:
    print(f"No pushed jobs found in {jobs_pushed_file}.")
    raise SystemExit(1)

print(f"Found {len(pushed_jobs)} pushed job(s) to check.\n")

# --- Process all pushed jobs, no merging with past data ---
success = []
failure = []
errors  = []
waiting = []

for job in pushed_jobs:
    if not isinstance(job, dict) or "id" not in job:
        continue

    try:
        job_url, conclusion = get_results(job, bench.config, bench.credentials)
        job["url"] = job_url
        job["conclusion"] = conclusion

        repo   = job.get("repo_name", "unknown")
        branch = job.get("branch_name", "unknown")

        if conclusion in ("waiting", None):
            waiting.append(job)
            print(f"[{repo} | {branch}] still waiting...")
        elif conclusion == "success":
            success.append(job)
            print(f"[{repo} | {branch}] → SUCCESS")
        elif conclusion == "failure":
            failure.append(job)
            print(f"[{repo} | {branch}] → FAILURE")
        elif conclusion == "error":
            errors.append(job)
            print(f"[{repo} | {branch}] → ERROR")
        else:
            # Unknown → treat as waiting to retry later
            waiting.append(job)
            print(f"[{repo} | {branch}] unknown conclusion '{conclusion}', keeping as waiting")

    except Exception as e:
        print(f"Error checking {job.get('repo_name', 'unknown')} (id={job.get('id')}): {e}")
        waiting.append(job)

# --- Overwrite split outputs (NO reading of existing files) ---
save_overwrite(success_file, success)
save_overwrite(failure_file, failure)
save_overwrite(errors_file,  errors)
save_overwrite(awaiting_file, waiting)

print("\n[Write-out]")
print(f"  success: {len(success)}  → {success_file}")
print(f"  failure: {len(failure)}  → {failure_file}")
print(f"  error:   {len(errors)}   → {errors_file}")
print(f"  waiting: {len(waiting)}  → {awaiting_file}")

# --- Build combined results for analysis (success + failure ONLY), overwrite ---
combined_results = success + failure
save_overwrite(results_file, combined_results)

print(f"\nCombined results (success+failure) written → {results_file}")
print(f"[Sanity] Combined row count: {len(combined_results)}")

# --- Analyze using an ISOLATED copy to avoid accidental globbing ---
analysis_dir = os.path.join(base_dir, "analysis_isolated")
os.makedirs(analysis_dir, exist_ok=True)
tmp_results_for_analysis = os.path.join(analysis_dir, "jobs_results_diff_SINGLE_RUN.jsonl")
shutil.copy2(results_file, tmp_results_for_analysis)

print("\n===============================")
print(" Running Evaluation Summary (isolated file)...")
print("===============================\n")
print(f"[Analyze] Passing: {tmp_results_for_analysis}")

bench.analyze_results(jobs_results_file=tmp_results_for_analysis)

print("\nEvaluation complete — metrics printed above.")
print("\nRecheck cycle finished.\n")
