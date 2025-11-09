#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# ========= CONFIGURATION =========
DATASET_PATH = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet")
OUTPUT_PATH = Path("/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/commit_diffs.json")

# Optional: Retry logic configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
TIMEOUT = 20     # seconds
# =================================

# ========= LOAD TOKEN =========
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise EnvironmentError("⚠️ GITHUB_TOKEN not found. Please set it in your .env file.")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff"
}
# =================================

def fetch_diff(repo_owner: str, repo_name: str, sha_fail: str, sha_success: str) -> str:
    """Fetches diff text between two commits using GitHub Compare API."""
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/compare/{sha_fail}...{sha_success}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 403 and "X-RateLimit-Remaining" in response.headers:
                print(f"[RateLimit] Sleeping 60s... Remaining={response.headers.get('X-RateLimit-Remaining')}")
                time.sleep(60)
            else:
                print(f"[WARN] {repo_owner}/{repo_name} diff fetch failed (status={response.status_code})")
                return None
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Network issue ({attempt}/{MAX_RETRIES}): {e}")
            time.sleep(RETRY_DELAY)
    return None

def main():
    print(f" Loading dataset from: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    print(f" Loaded {len(df)} rows")

    results = []
    for _, row in df.iterrows():
        repo_owner = row.get("repo_owner")
        repo_name = row.get("repo_name")
        sha_fail = row.get("sha_fail")
        sha_success = row.get("sha_success")
        record_id = int(row.get("id"))

        if not all([repo_owner, repo_name, sha_fail, sha_success]):
            print(f"[SKIP] Missing fields for ID={record_id}")
            continue

        print(f"🔍 Fetching diff for {repo_owner}/{repo_name} ({sha_fail[:7]} → {sha_success[:7]})")

        diff_text = fetch_diff(repo_owner, repo_name, sha_fail, sha_success)
        if diff_text:
            results.append({
                "id": record_id,
                "sha_fail": sha_fail,
                "diff": diff_text
            })
        else:
            print(f"[WARN] No diff found for ID={record_id}")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(results)} diffs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
