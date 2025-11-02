#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cancel all queued or running GitHub Actions workflow runs
for repositories listed in the local dataset (Parquet).

Owner: RabeyaMuna
"""

import os
import time
import requests
import pandas as pd
from omegaconf import OmegaConf
from dotenv import load_dotenv
from load_config import load_config
# ----------------------------------------------------
# Load configuration and environment
# ----------------------------------------------------
load_dotenv()

config, CONFIG_PATH = load_config()
DATASET_PATH = os.path.join(config.get("base_dir"), "dataset", "lca_dataset.parquet")

OWNER = config.get("benchmark_owner")  # <-- Force override to your GitHub username
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set your GITHUB_TOKEN environment variable first!")

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def github_get(url):
    """Perform GET request with retry and rate-limit handling."""
    for attempt in range(3):
        r = requests.get(url, headers=HEADERS)
        if r.status_code == 200:
            return r
        if r.status_code == 403 and "rate limit" in r.text.lower():
            reset_time = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
            sleep_for = max(0, reset_time - time.time()) + 5
            print(f"[RATE LIMIT] Sleeping {sleep_for:.0f}s before retry...")
            time.sleep(sleep_for)
        else:
            print(f"[WARN] GET failed ({r.status_code}) for {url}: {r.text[:150]}")
            time.sleep(2)
    return None


def list_runs(owner, repo, status="queued"):
    """Fetch all workflow runs with given status (queued or in_progress)."""
    runs = []
    base_url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
    url = f"{base_url}?status={status}"

    while url:
        r = github_get(url)
        if not r:
            break
        data = r.json()
        runs.extend(data.get("workflow_runs", []))
        url = r.links.get("next", {}).get("url")

    return runs


def cancel_run(owner, repo, run_id):
    """Cancel a specific workflow run by ID."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/cancel"
    r = requests.post(url, headers=HEADERS)
    if r.status_code == 202:
        print(f"[CANCEL] {owner}/{repo} → Run {run_id} cancellation accepted.")
        return True
    elif r.status_code == 403:
        print(f"[WARN] Permission denied to cancel run {run_id} in {owner}/{repo}.")
    elif r.status_code == 404:
        print(f"[WARN] Run {run_id} not found in {owner}/{repo}.")
    else:
        print(f"[WARN] Failed to cancel run {run_id} in {owner}/{repo}: {r.status_code}")
    return False


def cancel_all_waiting_jobs_for_repo(owner, repo):
    """Cancel all queued or in-progress workflow runs for a single repo."""
    total_canceled = 0
    for status in ("queued", "in_progress"):
        runs = list_runs(owner, repo, status=status)
        if not runs:
            print(f"[INFO] No {status} runs found for {owner}/{repo}.")
            continue

        print(f"[INFO] Found {len(runs)} {status} runs for {owner}/{repo}.")
        for run in runs:
            run_id = run.get("id")
            workflow_name = run.get("name", "unknown")
            print(f"[INFO] Attempting to cancel: {workflow_name} (run_id={run_id})")
            if cancel_run(owner, repo, run_id):
                total_canceled += 1
            time.sleep(0.5)  # gentle rate control

    return total_canceled


def cancel_all_dataset_repos():
    """Traverse all repos from dataset and cancel queued/running runs."""
    df = pd.read_parquet(DATASET_PATH)
    if "repo_name" not in df.columns:
        raise ValueError("Dataset must include a 'repo_name' column (e.g., diffusers).")

    repos = df["repo_name"].dropna().unique()
    print(f"[INFO] Loaded {len(repos)} repositories from dataset.")
    grand_total = 0

    for repo in repos:
        repo = str(repo).strip()
        if not repo:
            continue

        print(f"\n[REPO] Checking {OWNER}/{repo} for active runs...")
        try:
            canceled = cancel_all_waiting_jobs_for_repo(OWNER, repo)
            grand_total += canceled
        except Exception as e:
            print(f"[ERROR] Failed for {OWNER}/{repo}: {e}")

    print(f"\n[SUMMARY] Total runs canceled across all repos: {grand_total}")


# ----------------------------------------------------
# Main
# ----------------------------------------------------
if __name__ == "__main__":
    cancel_all_dataset_repos()
