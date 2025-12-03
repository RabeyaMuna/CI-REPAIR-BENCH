#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ci_backfill_utils.py

Algorithm:

Given a failed commit `sha_fail` (from your dataset):

1) Always collect changed files + diffs for `sha_fail` (relative to its parent).

2) Walk backwards through parents (up to `max_previous_commits`):

   For each commit `current_sha`:
     - Find `parent_sha`.
     - Look up GitHub Actions workflow runs in the REAL repo
       for this exact workflow file (workflow_rel_path) and this commit.

     - If there is at least one COMPLETED run for that workflow:
         * If ANY completed run has conclusion == "failure":
               => treat parent_sha as a FAILED commit.
                  - Collect its changed files + diffs.
                  - Continue backwards with current_sha = parent_sha.
         * Otherwise (no "failure" conclusion among completed runs):
               => treat as PASSED commit, STOP traversal.

     - If there are NO runs for that workflow, or none are completed:
         => no reliable info, STOP traversal.

Return structure:

{
  "sha_fail": <sha_fail>,   # top-level failed commit from dataset
  "changed_files": [
    {
      "commit": <commit_sha>,      # sha_fail or one of its failed parents
      "file_path": "<path/to/file>",
      "diff": "<unified diff vs parent>",
    },
    ...
  ]
}
"""

import os
import subprocess
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
import requests

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"

# Ignore config-ish / metadata files
IGNORED_EXTENSIONS = (".yaml", ".yml", ".xml", ".toml", ".lock", ".md", "json", "jsonl", "json5")


# =====================================================================
# Helpers
# =====================================================================

def _normalize_sha(sha: Any) -> str:
    """
    Ensure we always work with a string commit SHA.

    Accepts:
      - plain string
      - 1-element tuple/list like ('abc123',)
    """
    if isinstance(sha, (tuple, list)):
        if len(sha) == 1:
            return str(sha[0])
        raise TypeError(f"Expected single SHA, got multiple: {sha!r}")
    return str(sha)


# =====================================================================
# Git helpers
# =====================================================================

def _run_git(args: List[str], repo_path: str, check: bool = True) -> subprocess.CompletedProcess:
    """Small wrapper for running git commands."""
    return subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=check,
    )


def _try_get_parent_once(repo_path: str, sha: str) -> Optional[str]:
    """Try to get parent of sha once, without deepening."""
    sha = _normalize_sha(sha)
    try:
        proc = _run_git(["rev-parse", f"{sha}^"], repo_path, check=True)
        parent_sha = proc.stdout.strip()
        return parent_sha or None
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip()
        print(f"[WARN] Could not find parent of {sha}: {msg}")
        return None


def ensure_parent_commit_available(
    repo_path: str,
    sha: str,
    remote: str = "origin",
    deepen_step: int = 500,
    max_deepen_rounds: int = 3,
) -> Optional[str]:
    """
    Ensure the REAL parent commit of `sha` is available locally.

    - First, try rev-parse sha^.
    - If that fails, repeatedly:
        git fetch <remote> --deepen=<deepen_step>
      and retry up to max_deepen_rounds.

    Returns parent_sha if found, else None.
    """
    sha = _normalize_sha(sha)
    parent_sha = _try_get_parent_once(repo_path, sha)
    if parent_sha:
        return parent_sha

    for attempt in range(1, max_deepen_rounds + 1):
        print(
            f"[INFO] Parent of {sha} not available; deepening history "
            f"(attempt {attempt}/{max_deepen_rounds})..."
        )
        try:
            _run_git(
                ["fetch", remote, f"--deepen={deepen_step}"],
                repo_path,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(
                f"[WARN] git fetch --deepen failed for {sha}: "
                f"{(e.stderr or '').strip()}"
            )

        parent_sha = _try_get_parent_once(repo_path, sha)
        if parent_sha:
            print(f"[INFO] Parent of {sha} found after deepening.")
            return parent_sha

    print(f"[ERROR] Could not ensure parent of {sha} in local repo.")
    return None


def get_commit_changed_files(
    repo_path: str,
    sha: str,
    remote: str = "origin",
) -> List[str]:
    """
    Fetch the REAL parent of `sha`, then return changed files:

        git diff --name-only <parent_sha> <sha>

    Filters out config-style / doc / JSON-family files:
    - .yaml, .yml, .xml, .toml, .lock, .md
    - anything with extension starting with '.json' (.json, .jsonl, .json5, ...)
    """
    sha = _normalize_sha(sha)
    parent_sha = ensure_parent_commit_available(repo_path, sha, remote=remote)
    if not parent_sha:
        print(
            f"[ERROR] Cannot get changed files for {sha} because parent is not available."
        )
        return []

    try:
        proc = _run_git(
            ["diff", "--name-only", parent_sha, sha],
            repo_path,
            check=True,
        )
        all_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

        filtered_files = []
        for fp in all_files:
            lower_fp = fp.lower()
            _, ext = os.path.splitext(lower_fp)  # e.g. ".py", ".json", ".jsonl"

            # Ignore config/doc files
            if ext in IGNORED_EXTENSIONS:
                print(f"[INFO] Skipping ignored config/doc file: {fp}")
                continue

            # Ignore ANY JSON-like extension: .json, .jsonl, .json5, etc.
            if ext.startswith(".json"):
                print(f"[INFO] Skipping JSON-family file: {fp}")
                continue

            filtered_files.append(fp)

        return filtered_files

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to get changed files for {sha}: {e.stderr.strip()}")
        return []


def get_file_diff_for_commit(
    repo_path: str,
    sha: str,
    file_path: str,
    remote: str = "origin",
) -> str:
    """
    Get unified diff for `file_path` in commit `sha` vs its REAL parent:

        git diff <parent_sha> <sha> -- <file_path>
    """
    sha = _normalize_sha(sha)
    parent_sha = ensure_parent_commit_available(repo_path, sha, remote=remote)
    if not parent_sha:
        print(
            f"[ERROR] Cannot get diff for {file_path} in {sha} because parent is not available."
        )
        return ""

    try:
        proc = _run_git(
            ["diff", parent_sha, sha, "--", file_path],
            repo_path,
            check=True,
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to get diff for {file_path} in {sha}: {e.stderr.strip()}"
        )
        return ""


# =====================================================================
# GitHub Actions helpers (REAL repo CI status – workflow-specific)
# =====================================================================

def get_runs_for_commit(
    owner: str,
    repo: str,
    sha: str,
    per_page: int = 50,
    workflow_rel_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return workflow runs in the REAL repo where head_sha == sha.

    If workflow_rel_path is provided, only keep runs whose `.path` matches
    that workflow (by full path or by basename).
    """
    sha = _normalize_sha(sha)
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    params = {
        "head_sha": sha,
        "per_page": per_page,
        # NOTE: we do NOT filter by 'event'; we want ANY event for that workflow.
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    if resp.status_code != 200:
        print(
            f"[WARN] Failed to fetch runs for {owner}/{repo}@{sha}: "
            f"HTTP {resp.status_code} {resp.text[:200]}"
        )
        return []

    runs = resp.json().get("workflow_runs", [])

    if workflow_rel_path:
        target_base = os.path.basename(workflow_rel_path)
        filtered = []
        for r in runs:
            r_path = (r.get("path") or "").strip()
            if not r_path:
                continue
            r_base = os.path.basename(r_path)
            # match by full rel path or just the filename
            if r_path == workflow_rel_path or r_base == target_base:
                filtered.append(r)

        print(
            f"[INFO] Filtered runs for {owner}/{repo}@{sha} "
            f"by workflow='{workflow_rel_path}' → {len(filtered)} run(s)."
        )
        runs = filtered

    return runs


def get_commit_ci_status_for_push(  # name kept for compatibility
    owner: str,
    repo: str,
    sha: str,
    workflow_rel_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    CI status for REAL repo owner/repo and commit sha,
    considering **ALL events** but **ONLY** for the given workflow file.

    Rule you requested:

    - Take all workflow runs whose head_sha == sha and whose .path
      matches workflow_rel_path (by full path or basename).
    - Consider only runs with status == 'completed'.

      * If ANY completed run has conclusion == 'failure'  -> commit is FAILED.
      * Otherwise (no completed 'failure' for that workflow) -> commit is PASSED.

    If there are no runs at all, or no completed runs for that workflow:
      -> conclusion = None (unknown / no-info).
    """
    sha = _normalize_sha(sha)
    runs = get_runs_for_commit(
        owner=owner,
        repo=repo,
        sha=sha,
        per_page=50,
        workflow_rel_path=workflow_rel_path,
    )

    if not runs:
        return {"sha": sha, "has_runs": False, "conclusion": None, "latest_run": None}

    completed = [r for r in runs if r.get("status") == "completed"]
    if not completed:
        # Runs exist but none are completed yet
        return {"sha": sha, "has_runs": True, "conclusion": None, "latest_run": None}

    # Sort by created_at descending to pick a canonical "latest"
    completed.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    latest = completed[0]

    # If ANY completed run for this workflow is 'failure' => FAILED commit.
    has_failure = any(r.get("conclusion") == "failure" for r in completed)

    if has_failure:
        conclusion = "failure"
    else:
        # Your rule: otherwise treat as passed.
        conclusion = "success"

    return {
        "sha": sha,
        "has_runs": True,
        "conclusion": conclusion,
        "latest_run": latest,
    }


# =====================================================================
# High-level function: sha_fail + previous failed commits (backwards)
# =====================================================================

def collect_changed_files_for_fail_and_parent(
    owner: str,
    repo: str,
    repo_path: str,
    sha_fail: str,
    workflow_rel_path: str,
    workflow_yaml_from_dataset: str,  # kept for signature compatibility (unused)
    max_previous_commits: int = 5,
) -> Dict[str, Any]:
    """
    Backward CI-failure chain based on a single workflow file:

    1) Given a failed commit sha_fail (from dataset):
       - Fetch changed_files + diffs for sha_fail (once).

    2) Then go backwards up to `max_previous_commits`:

       For each current_sha:
         * parent_sha = parent(current_sha)
         * Get CI status for parent_sha in REAL repo, but ONLY for
           the workflow at `workflow_rel_path`.

         * If has_runs and conclusion == 'failure':
               - Treat parent_sha as FAILED, collect its changed_files + diffs.
               - current_sha = parent_sha and continue.

         * If has_runs and conclusion == 'success':
               - Treat parent_sha as PASSED, STOP traversal.

         * If no runs or no completed runs:
               - STOP traversal (no reliable info).

    3) Return:
        {
          "sha_fail": sha_fail,
          "changed_files": [
            { "commit": <commit_sha>, "file_path": <path>, "diff": "<unified diff>" },
            ...
          ]
        }

    All changed files (sha_fail + failed parents) live under one top-level sha_fail.
    """
    sha_fail = _normalize_sha(sha_fail)

    file_records: List[Dict[str, Any]] = []
    seen_pairs = set()  # (commit_sha, file_path)

    # --- Step 1: ALWAYS collect changed files for sha_fail ---
    print(f"[INFO] Collecting changed files for sha_fail={sha_fail}...")
    fail_changed_files = get_commit_changed_files(repo_path, sha_fail)

    if not fail_changed_files:
        print(
            f"[WARN] No changed files found for sha_fail={sha_fail}. "
            "This is unexpected for a failed commit."
        )
    else:
        print(f"[INFO] Found {len(fail_changed_files)} changed files for sha_fail.")
        for fp in fail_changed_files:
            diff_text = get_file_diff_for_commit(repo_path, sha_fail, fp)
            key = (sha_fail, fp)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            file_records.append(
                {
                    "commit": sha_fail,
                    "file_path": fp,
                    "diff": diff_text,
                }
            )

    # --- Step 2: walk backwards through previous commits (only if FAILED) ---
    current_sha = sha_fail
    steps_back = 0

    while steps_back < max_previous_commits:
        parent_sha = ensure_parent_commit_available(repo_path, current_sha)
        if not parent_sha:
            print(
                f"[INFO] Commit {current_sha} has no parent (or parent unavailable). "
                "Stopping parent traversal."
            )
            break

        steps_back += 1
        print(f"[INFO] Checking parent #{steps_back}: {parent_sha} of {current_sha}...")

        status = get_commit_ci_status_for_push(
            owner=owner,
            repo=repo,
            sha=parent_sha,
            workflow_rel_path=workflow_rel_path,
        )
        conclusion = status["conclusion"]
        has_runs = status["has_runs"]

        if not has_runs or conclusion is None:
            print(
                f"[INFO] Parent {parent_sha} has no usable runs "
                "for this workflow (no runs or none completed). Stopping traversal."
            )
            break

        print(
            f"[INFO] REAL repo parent {parent_sha} has CI conclusion={conclusion} "
            f"for workflow='{workflow_rel_path}'."
        )

        if conclusion == "failure":
            # Failed parent: collect its changed files and continue further back
            print(
                f"[INFO] Parent {parent_sha} is FAILED. "
                "Collecting its changed files..."
            )
            parent_changed_files = get_commit_changed_files(repo_path, parent_sha)
            for fp in parent_changed_files:
                key = (parent_sha, fp)
                if key in seen_pairs:
                    continue
                diff_text = get_file_diff_for_commit(repo_path, parent_sha, fp)
                seen_pairs.add(key)
                file_records.append(
                    {
                        "commit": parent_sha,
                        "file_path": fp,
                        "diff": diff_text,
                    }
                )

            current_sha = parent_sha
            continue

        # Any non-failure conclusion => treat as passed and stop
        print(
            f"[INFO] Parent {parent_sha} is not failed "
            f"(conclusion={conclusion}). Stopping traversal."
        )
        break

    return {
        "sha_fail": sha_fail,
        "changed_files": file_records,
    }
