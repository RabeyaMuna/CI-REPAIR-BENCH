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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_jsonl(path, records)

# --- Failure-only fast-fail normalization (no success inference) -----------
def normalize_failure_only(records):
    """
    Normalize ONLY failures (leave success as reported by the API).

    Embedded-jobs case (r['jobs'] / r['job_list']):
      1) If ANY job has status=='completed' AND conclusion=='failure' => run 'failure'
      2) Else, fast-fail heuristic: if ANY job is 'cancelled' AND at least one job is 'completed' => run 'failure'
      3) Else, if ANY job has conclusion=='failure' (even if not completed) => run 'failure'
      4) Else unchanged

    Flat per-job rows case (group by run_id/workflow_run_id/runId):
      1) If ANY row has status=='completed' AND conclusion=='failure' in group => run 'failure'
      2) Else, fast-fail heuristic: ANY row 'cancelled' AND at least one row 'completed' => run 'failure'
      3) Else, if ANY row has conclusion=='failure' in group => run 'failure'
      4) Else unchanged
    """
    # ---- Case A: at least one record embeds a jobs list ----
    if any(isinstance(r, dict) and (r.get("jobs") or r.get("job_list")) for r in records):
        out = []
        for r in records:
            if not isinstance(r, dict):
                out.append(r)
                continue

            jobs = r.get("jobs") or r.get("job_list") or []
            if not jobs:
                out.append(r)
                continue

            statuses    = [((j or {}).get("status") or "").lower() for j in jobs]
            conclusions = [((j or {}).get("conclusion") or "").lower() for j in jobs]

            # 1) completed failure wins immediately
            if any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions)):
                rr = dict(r); rr["conclusion"] = "failure"; out.append(rr); continue

            # 2) fast-fail heuristic: cancellations visible alongside at least one completed job
            if ("cancelled" in conclusions) and ("completed" in statuses):
                rr = dict(r); rr["conclusion"] = "failure"; out.append(rr); continue

            # 3) any failure (even if not completed yet)
            if any(co == "failure" for co in conclusions):
                rr = dict(r); rr["conclusion"] = "failure"; out.append(rr); continue

            # 4) unchanged (do NOT set success here)
            out.append(r)
        return out

    # ---- Case B: flat per-job rows → group by a run id ----
    run_key = None
    for k in ("run_id", "workflow_run_id", "runId"):
        if any(isinstance(r, dict) and k in r for r in records):
            run_key = k
            break
    if run_key is None:
        return records  # nothing to group by; leave unchanged

    by_run = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        by_run.setdefault(r.get(run_key), []).append(r)

    runs_fail = set()
    for rid, rows in by_run.items():
        if rid is None:
            continue

        statuses    = [((row.get("status") or "").lower()) for row in rows]
        conclusions = [((row.get("conclusion") or "").lower()) for row in rows]

        # 1) completed failure in the group
        if any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions)):
            runs_fail.add(rid)
            continue

        # 2) fast-fail heuristic: cancelled present + at least one completed row
        if ("cancelled" in conclusions) and ("completed" in statuses):
            runs_fail.add(rid)
            continue

        # 3) any row already marked failure (not necessarily completed)
        if any(co == "failure" for co in conclusions):
            runs_fail.add(rid)

    out = []
    for r in records:
        if isinstance(r, dict) and r.get(run_key) in runs_fail:
            rr = dict(r); rr["conclusion"] = "failure"; out.append(rr)
        else:
            out.append(r)
    return out
# --------------------------------------------------------------------------

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

# --- Process all pushed jobs (collect first; then normalize) ---
checked = []

for job in pushed_jobs:
    if not isinstance(job, dict) or "id" not in job:
        continue
    try:
        job_url, conclusion = get_results(job, bench.config, bench.credentials)
        job["url"] = job_url
        job["conclusion"] = conclusion
    except Exception as e:
        print(f"Error checking {job.get('repo_name', 'unknown')} (id={job.get('id')}): {e}")
        job.setdefault("conclusion", "waiting")
    checked.append(job)

# --- Apply FAILURE-ONLY fast-fail normalization BEFORE splitting ---
normalized = normalize_failure_only(checked)

# --- Split after normalization ---
success, failure, errors, waiting = [], [], [], []
for job in normalized:
    repo   = job.get("repo_name", "unknown")
    branch = job.get("branch_name", "unknown")
    concl  = (job.get("conclusion") or "").lower()

    # keep your original buckets; treat transient states as waiting
    if concl in ("waiting", "queued", "in_progress", ""):
        waiting.append(job)
        print(f"[{repo} | {branch}] still waiting...")
    elif concl == "success":
        success.append(job)
        print(f"[{repo} | {branch}] → SUCCESS")
    elif concl == "failure":
        failure.append(job)
        print(f"[{repo} | {branch}] → FAILURE")
    elif concl == "error":
        errors.append(job)
        print(f"[{repo} | {branch}] → ERROR")
    else:
        waiting.append(job)
        print(f"[{repo} | {branch}] unknown conclusion '{concl}', keeping as waiting")

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
