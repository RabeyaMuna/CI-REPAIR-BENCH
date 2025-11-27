#!/usr/bin/env python3
"""
For each row in missing_success_diffs.json, fetch ALL commits that come
after sha_fail up to sha_success (or default branch head), using the GitHub
compare API.

Input:
  /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/missing_success_diffs.json

Output:
  /Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/commits_between_fail_and_head.json
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# ========= PATHS =========
DATASET_PATH = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/missing_success_diffs.json"
)
OUT_PATH = Path(
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/commits_between_fail_and_head.json"
)

# ========= LOGGING =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ========= GITHUB AUTH =========
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise EnvironmentError("GITHUB_TOKEN not found in environment.")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}
GITHUB_API_URL = "https://api.github.com"


def github_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 3,
    timeout: int = 20,
) -> Optional[requests.Response]:
    """
    Simple GET wrapper with a few retries.
    Returns Response or None if all retries fail.
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
            return resp
        except requests.exceptions.RequestException as e:
            logging.warning(
                "GitHub GET error (%s), attempt %d/%d for %s",
                e,
                attempt,
                max_retries,
                url,
            )
            if attempt < max_retries:
                time.sleep(3 * attempt)
    logging.error("Giving up on %s after %d attempts.", url, max_retries)
    return None


def get_default_branch(owner: str, repo: str) -> Optional[str]:
    """
    Fetch default branch name for a repo.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}"
    resp = github_get(url)
    if resp is None or resp.status_code != 200:
        logging.warning(
            "Failed to get repo info for %s/%s: %s %s",
            owner,
            repo,
            resp.status_code if resp else "NO_RESP",
            resp.text[:200] if resp else "",
        )
        return None
    data = resp.json()
    return data.get("default_branch")


def get_commits_between_fail_and_head(
    owner: str,
    repo: str,
    sha_fail: str,
    sha_success: Optional[str],
) -> List[Dict[str, Any]]:
    """
    Use GitHub compare API to get ALL commits that come
    after sha_fail, up to sha_success (if provided) or the default branch head.

    - base (left side)  = sha_fail
    - head (right side) = sha_success OR default branch

    GitHub returns commits from OLDEST -> NEWEST (base to head).
    """
    # Decide head commit for comparison
    if sha_success:
        head = sha_success
    else:
        default_branch = get_default_branch(owner, repo)
        if not default_branch:
            logging.warning(
                "No default branch found for %s/%s, skipping.", owner, repo
            )
            return []
        head = default_branch

    compare_url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/compare/{sha_fail}...{head}"
    logging.info("Compare %s/%s: %s...%s", owner, repo, sha_fail, head)

    resp = github_get(compare_url)
    if resp is None:
        return []

    if resp.status_code != 200:
        logging.warning(
            "Compare failed for %s/%s (%s...%s): %s %s",
            owner,
            repo,
            sha_fail,
            head,
            resp.status_code,
            resp.text[:200],
        )
        return []

    data = resp.json()
    commits = data.get("commits", []) or []

    total_commits = data.get("total_commits")
    if total_commits is not None and total_commits > len(commits):
        logging.warning(
            "Compare for %s/%s (%s...%s) is TRUNCATED: total_commits=%s, returned=%s",
            owner,
            repo,
            sha_fail,
            head,
            total_commits,
            len(commits),
        )

    cleaned: List[Dict[str, Any]] = []
    for c in commits:
        sha = c.get("sha")
        commit_obj = c.get("commit", {}) or {}
        author = commit_obj.get("author", {}) or {}
        committer = commit_obj.get("committer", {}) or {}

        cleaned.append(
            {
                "sha": sha,
                "html_url": c.get("html_url"),
                "author_date": author.get("date"),
                "committer_date": committer.get("date"),
                # First line of commit message only
                "message": (commit_obj.get("message") or "").split("\n", 1)[0],
            }
        )

    return cleaned


def main():
    logging.info("Loading %s", DATASET_PATH)
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    results: List[Dict[str, Any]] = []

    for row in entries:
        repo_full = row["repo"]  # e.g. "kornia/kornia"
        sha_fail = row["sha_fail"]
        sha_success = row.get("sha_success")

        try:
            owner, repo = repo_full.split("/", 1)
        except ValueError:
            logging.error("Invalid repo format '%s' for id=%s", repo_full, row.get("id"))
            continue

        logging.info(
            "Processing id=%s repo=%s sha_fail=%s sha_success=%s",
            row.get("id"),
            repo_full,
            sha_fail,
            sha_success,
        )

        commits_between = get_commits_between_fail_and_head(
            owner=owner,
            repo=repo,
            sha_fail=sha_fail,
            sha_success=sha_success,
        )

        results.append(
            {
                "id": row.get("id"),
                "repo": repo_full,
                "sha_fail": sha_fail,
                "sha_success": sha_success,
                "fail_commit_url": row.get("fail_commit_url"),
                "success_commit_url": row.get("success_commit_url"),
                "reason": row.get("reason"),
                # All commits from OLDEST -> NEWEST between fail and head
                "commits_between_fail_and_head": commits_between,
            }
        )

    logging.info("Saving results to %s", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("Done. Processed %d entries.", len(results))


if __name__ == "__main__":
    main()
