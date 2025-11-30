#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ci_backfill_utils.py

Logic:

1. Always collect file changes for `sha_fail` commit.
2. Look at the parent of `sha_fail`:
   - If push CI runs already exist for the parent:
       * Use their conclusion (success/failure).
   - If NO push runs exist:
       * Use provided workflow path + YAML (from dataset),
         overwrite that workflow file at the parent commit on a temp branch,
         enforce `on: push`,
         push branch to trigger CI,
         wait for a completed push run for that temp commit,
         read its conclusion (success/failure).
   - If the parent is (or behaves as) FAILED:
       * Collect file changes for the parent commit as well
         (but do NOT overwrite newer info from `sha_fail`).
   - If the parent is SUCCESS or not clearly failed:
       * Stop and just return what we have.

Return format:

{
  "sha_fail": <sha_fail>,
  "changed_files": [
    {
      "commit": <commit_sha>,
      "file_path": <path/to/file>,
      "diff": "<file-level unified diff>"
    },
    ...
  ]
}
"""

import os
import time
import subprocess
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import requests
import yaml

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API = "https://api.github.com"



# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def get_parent_commit_sha(repo_path: str, sha: str) -> Optional[str]:
    """
    Return the parent commit SHA of `sha` (sha^).
    Returns None if this is the root commit or parent cannot be found.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", f"{sha}^"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        parent_sha = proc.stdout.strip()
        if parent_sha:
            return parent_sha
        return None
    except subprocess.CalledProcessError as e:
        print(f"[WARN] Could not find parent of {sha}: {e}")
        return None


def get_commit_changed_files(repo_path: str, sha: str) -> List[str]:
    """
    Get the list of changed files for a commit relative to its parent.

    Uses:
        git diff --name-only sha^ sha
    """
    files_proc = subprocess.run(
        ["git", "diff", "--name-only", f"{sha}^", sha],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in files_proc.stdout.splitlines() if line.strip()]


def get_file_diff_for_commit(repo_path: str, sha: str, file_path: str) -> str:
    """
    Get the unified diff for a single file in a given commit,
    relative to its parent.

    Uses:
        git diff sha^ sha -- <file_path>
    """
    diff_proc = subprocess.run(
        ["git", "diff", f"{sha}^", sha, "--", file_path],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )
    return diff_proc.stdout


# ---------------------------------------------------------------------------
# GitHub Actions helpers
# ---------------------------------------------------------------------------

def get_runs_for_commit(
    owner: str,
    repo: str,
    sha: str,
    per_page: int = 30,
    event: str = "push",
) -> List[Dict[str, Any]]:
    """
    Return all workflow runs for this repository whose head_sha == sha and event == `event`.

    Uses:
        GET /repos/{owner}/{repo}/actions/runs?head_sha=sha&event=push
    """
    url = f"{GITHUB_API}/repos/{owner}/{repo}/actions/runs"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    params = {
        "head_sha": sha,
        "per_page": per_page,
        "event": event,
    }

    resp = requests.get(url, headers=headers, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("workflow_runs", [])


def get_commit_ci_status_for_push(
    owner: str,
    repo: str,
    sha: str,
) -> Dict[str, Any]:
    """
    Inspect CI status for a commit for push workflows ONLY.

    Logic:
    - Fetch all runs with head_sha == sha and event == "push".
    - If none: has_runs=False, conclusion=None.
    - Otherwise:
        - Consider only 'completed' runs.
        - Take the LATEST completed run by created_at as the canonical one.
        - conclusion = that run's 'conclusion' ('success', 'failure', etc.).

    Returns:
        {
          "sha": sha,
          "has_runs": bool,
          "latest_run": { ... } or None,
          "conclusion": "success" | "failure" | "cancelled" | None,
        }
    """
    runs = get_runs_for_commit(owner, repo, sha, event="push")

    if not runs:
        return {
            "sha": sha,
            "has_runs": False,
            "latest_run": None,
            "conclusion": None,
        }

    completed = [r for r in runs if r.get("status") == "completed"]
    if not completed:
        # Runs exist but none completed yet
        return {
            "sha": sha,
            "has_runs": True,
            "latest_run": None,
            "conclusion": None,
        }

    completed.sort(key=lambda r: r["created_at"], reverse=True)
    latest = completed[0]
    conclusion = latest.get("conclusion")

    return {
        "sha": sha,
        "has_runs": True,
        "latest_run": latest,
        "conclusion": conclusion,
    }


def wait_for_push_ci_conclusion(
    owner: str,
    repo: str,
    sha: str,
    max_wait_seconds: int = 600,
    poll_interval: int = 15,
) -> Dict[str, Any]:
    """
    Poll until we get a completed push run (success/failure) for this commit,
    or until timeout.

    Returns the same dict as get_commit_ci_status_for_push().
    """
    deadline = time.time() + max_wait_seconds

    while time.time() < deadline:
        status = get_commit_ci_status_for_push(owner, repo, sha)
        if status["has_runs"] and status["conclusion"] in ("success", "failure"):
            print(f"[INFO] Completed push run for {sha} with conclusion={status['conclusion']}.")
            return status

        if status["has_runs"]:
            print(f"[INFO] Push runs exist for {sha} but not completed with success/failure yet.")
        else:
            print(f"[INFO] Still no push runs for {sha}...")

        time.sleep(poll_interval)

    print(f"[WARN] Timed out waiting for push run for {sha}. Returning last known status.")
    return get_commit_ci_status_for_push(owner, repo, sha)


# ---------------------------------------------------------------------------
# Workflow rewrite + trigger (for parent commit when no runs exist)
# ---------------------------------------------------------------------------

def rewrite_workflow_and_push_temp_branch(
    repo_path: str,
    base_sha: str,
    workflow_rel_path: str,
    workflow_yaml_from_dataset: str,
    remote: str = "origin",
    branch_prefix: str = "ci-repair-parent",
) -> Optional[str]:
    """
    Create a temporary branch at `base_sha`, replace the specified workflow file
    with the given YAML (forcing `on: push`), commit, and push.

    Returns:
        The new commit SHA (HEAD of the temp branch) if everything succeeds,
        otherwise None.
    """
    temp_branch = f"{branch_prefix}-{base_sha[:7]}"

    try:
        # Create or reset local branch at base_sha
        subprocess.run(
            ["git", "checkout", "-B", temp_branch, base_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create/reset temp branch {temp_branch}: {e.stderr}")
        return None

    # Build full workflow path
    workflow_full_path = os.path.join(repo_path, workflow_rel_path)
    os.makedirs(os.path.dirname(workflow_full_path), exist_ok=True)

    # Load YAML from dataset and enforce `on: push`
    try:
        data = yaml.safe_load(workflow_yaml_from_dataset) or {}
        # Force event to push
        # You can choose either this form:
        #   on:
        #     push: {}
        # or simply:
        #   on: push
        # Here we use the mapping form:
        data["on"] = {"push": {}}
        new_yaml_str = yaml.safe_dump(data, sort_keys=False)
    except Exception as e:
        print(f"[WARN] Failed to parse dataset workflow YAML, writing raw content. Error: {e}")
        # Fallback: just write the raw YAML from dataset
        new_yaml_str = workflow_yaml_from_dataset

    # Write workflow file
    with open(workflow_full_path, "w", encoding="utf-8") as f:
        f.write(new_yaml_str)

    try:
        # Stage and commit
        subprocess.run(
            ["git", "add", workflow_rel_path],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Temp CI workflow for parent {base_sha}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # Push the temp branch
        subprocess.run(
            ["git", "push", remote, f"{temp_branch}:{temp_branch}"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # Get the new commit SHA
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        new_sha = proc.stdout.strip()
        print(f"[INFO] Pushed temp branch '{temp_branch}' with new commit {new_sha}.")
        return new_sha

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to commit/push temp branch {temp_branch}: {e.stderr}")
        return None


# ---------------------------------------------------------------------------
# High-level function: sha_fail + its parent (one step back)
# ---------------------------------------------------------------------------

def collect_changed_files_for_fail_and_parent(
    owner: str,
    repo: str,
    repo_path: str,
    sha_fail: str,
    workflow_rel_path: str,
    workflow_yaml_from_dataset: str,
    max_wait_seconds: int = 600,
    poll_interval: int = 4,
) -> Dict[str, Any]:
    """
    1) First, collect changed files for the given `sha_fail` commit.
    2) Then, inspect the parent commit of `sha_fail`:

       - Check if push CI runs exist for the parent:
           * If YES, inspect the conclusion:
               - If conclusion == 'failure':
                   -> collect changed files for the parent (without overriding
                      any files already collected from sha_fail).
               - If conclusion == 'success' or anything else:
                   -> stop and return current results.
           * If NO runs exist:
               - Use the given workflow path + workflow YAML from dataset,
                 overwrite the workflow file for the parent on a temp branch
                 (forcing `on: push`), push that branch to trigger CI,
                 wait for a completed push run, and interpret its conclusion:
                   - If 'failure': collect changed files for the parent as above.
                   - Otherwise: stop and return current results.

    Dedup rule:
    - If the same file_path appears in both `sha_fail` and parent,
      we keep ONLY the info from `sha_fail` (more recent commit).

    Return:
        {
          "sha_fail": sha_fail,
          "changed_files": [
            {
              "commit": <commit_sha>,
              "file_path": <path/to/file>,
              "diff": "<file-level unified diff>",
            },
            ...
          ]
        }
    """
    # For dedup: file_path -> record (newest first)
    file_records: Dict[str, Dict[str, Any]] = {}

    # --- Step 1: collect changed files for sha_fail ---
    print(f"[INFO] Collecting changed files for sha_fail={sha_fail}...")
    fail_changed_files = get_commit_changed_files(repo_path, sha_fail)
    for fp in fail_changed_files:
        diff_text = get_file_diff_for_commit(repo_path, sha_fail, fp)
        file_records[fp] = {
            "commit": sha_fail,
            "file_path": fp,
            "diff": diff_text,
        }

    # --- Step 2: look at the parent commit of sha_fail ---
    parent_sha = get_parent_commit_sha(repo_path, sha_fail)
    if not parent_sha:
        print(f"[INFO] Commit {sha_fail} has no parent. Returning only sha_fail data.")
        return {
            "sha_fail": sha_fail,
            "changed_files": list(file_records.values()),
        }

    print(f"[INFO] Checking parent commit {parent_sha}...")

    status = get_commit_ci_status_for_push(owner, repo, parent_sha)

    if status["has_runs"]:
        # There are already push CI runs for the parent
        conclusion = status["conclusion"]
        print(f"[INFO] Parent {parent_sha} has push runs with conclusion={conclusion}.")

        if conclusion == "failure":
            # Collect parent changed files (without overriding sha_fail data)
            print(f"[INFO] Parent {parent_sha} is FAILED. Collecting its changes...")
            parent_changed_files = get_commit_changed_files(repo_path, parent_sha)
            for fp in parent_changed_files:
                if fp in file_records:
                    # sha_fail already changed this file -> keep sha_fail info
                    print(f"[INFO] Skipping '{fp}' from parent {parent_sha} "
                          f"because sha_fail already has it.")
                    continue
                diff_text = get_file_diff_for_commit(repo_path, parent_sha, fp)
                file_records[fp] = {
                    "commit": parent_sha,
                    "file_path": fp,
                    "diff": diff_text,
                }

        else:
            # success or cancelled/skipped/None -> stop and return
            print(f"[INFO] Parent {parent_sha} is not a failed commit "
                  f"(conclusion={conclusion}). Stopping here.")

        # Either way (failure or not), we are done (only one step back)
        return {
            "sha_fail": sha_fail,
            "changed_files": list(file_records.values()),
        }

    # No push runs exist for parent_sha: we need to rewrite workflow and trigger
    print(f"[INFO] No push runs exist for parent {parent_sha}. "
          f"Rewriting workflow '{workflow_rel_path}' from dataset and triggering CI...")

    new_ci_sha = rewrite_workflow_and_push_temp_branch(
        repo_path=repo_path,
        base_sha=parent_sha,
        workflow_rel_path=workflow_rel_path,
        workflow_yaml_from_dataset=workflow_yaml_from_dataset,
    )

    if not new_ci_sha:
        print(f"[WARN] Failed to create/push temp CI branch for parent {parent_sha}. "
              f"Returning only sha_fail data.")
        return {
            "sha_fail": sha_fail,
            "changed_files": list(file_records.values()),
        }

    # Wait for CI result on the new temp commit
    ci_status = wait_for_push_ci_conclusion(
        owner=owner,
        repo=repo,
        sha=new_ci_sha,
        max_wait_seconds=max_wait_seconds,
        poll_interval=poll_interval,
    )
    conclusion = ci_status["conclusion"]
    print(f"[INFO] Temp CI commit {new_ci_sha} for parent {parent_sha} "
          f"completed with conclusion={conclusion}.")

    if conclusion == "failure":
        # Parent behaves as failed under this workflow. Collect its changed files.
        print(f"[INFO] Treating parent {parent_sha} as FAILED under dataset workflow. "
              f"Collecting its changes...")
        parent_changed_files = get_commit_changed_files(repo_path, parent_sha)
        for fp in parent_changed_files:
            if fp in file_records:
                print(f"[INFO] Skipping '{fp}' from parent {parent_sha} "
                      f"because sha_fail already has it.")
                continue
            diff_text = get_file_diff_for_commit(repo_path, parent_sha, fp)
            file_records[fp] = {
                "commit": parent_sha,
                "file_path": fp,
                "diff": diff_text,
            }
    else:
        print(f"[INFO] Under dataset workflow, parent {parent_sha} is not failed "
              f"(conclusion={conclusion}). Not adding its changes.")

    return {
        "sha_fail": sha_fail,
        "changed_files": list(file_records.values()),
    }
