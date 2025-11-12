# fast_fail_detail.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import requests
from benhmark_functions import get_results

# -----------------------------
# Module-level helpers (no nesting)
# -----------------------------
def normalize_run_level_conclusion(concl: str) -> str:
    c = (concl or "").lower()
    if c in ("cancelled", "timed_out", "timeout"):
        return "failure"
    return c or ""

def parse_owner_repo_run_id(url: str):
    """
    Parse .../github.com/<owner>/<repo>/actions/runs/<run_id> (no regex).
    Returns (owner, repo, run_id) or (None, None, None).
    """
    if not url:
        return None, None, None
    s = url.strip().strip("/")
    parts = s.split("/")
    # Handle schema-present URLs
    if "//" in s and "github.com" not in parts:
        s2 = s.split("//", 1)[1]
        parts = s2.strip("/").split("/")
    # Prefer actions/runs pattern
    if "actions" in parts and "runs" in parts:
        a = parts.index("actions")
        if a >= 2 and a + 2 < len(parts):
            return parts[a - 2], parts[a - 1], parts[a + 2]
    # Fallback using 'github.com' anchor
    if "github.com" in parts and len(parts) >= parts.index("github.com") + 6:
        i = parts.index("github.com")
        return parts[i + 1], parts[i + 2], parts[i + 5]
    return None, None, None

def build_github_headers(token: str | None):
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


# -----------------------------
# Detail pass (flat; call from your class)
# -----------------------------
def finalize_after_last_poll(
    self,
    *,
    jobs_results: list,
    jobs_ids_await: list,
    jobs_ids_invalid: list,
    stream_results_path: str,
) -> None:
    """
    One-time detail pass to run AFTER the polling loop.

    Mutates (in place):
      - jobs_results: extends with any newly-resolved rows (incl. inferred failures)
      - jobs_ids_invalid: extends with any error rows discovered now
      - jobs_ids_await: replaced with the final still-waiting rows

    Also appends any newly-resolved rows to the streaming results file.
    """
   # local import to avoid cycles

    REQ_DELAY = 0.8

    # ---------- 1) Re-check all still-waiting once via get_results() ----------
    resolved_now = []
    still_waiting = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for job in list(jobs_ids_await):
            try:
                job_url, conclusion = get_results(job, self.config, self.credentials)
            except Exception:
                job_url, conclusion = None, "waiting"

            conclusion = normalize_run_level_conclusion(conclusion)

            if conclusion in ("waiting", "queued", "in_progress", ""):
                row = dict(job)
                if job_url:
                    row["url"] = job_url
                row["conclusion"] = "waiting"
                still_waiting.append(row)

            elif conclusion == "error":
                rr = dict(job)
                rr["url"] = job_url
                rr["conclusion"] = "error"
                jobs_ids_invalid.append(rr)

            else:
                rr = dict(job)
                rr["url"] = job_url
                rr["conclusion"] = conclusion  # success or failure
                resolved_now.append(rr)
                json.dump(rr, streamf); streamf.write("\n")

            time.sleep(REQ_DELAY)

    if resolved_now:
        jobs_results.extend(resolved_now)

    # ---------- 2) For still-waiting rows, use GH API to infer fast-fail ----------
    token = (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or self.credentials.get("token")
    )
    headers = build_github_headers(token)

    inferred_resolved = []   # will include inferred failures and successes
    undecided_after_api = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for row in still_waiting:
            owner, repo, run_id = parse_owner_repo_run_id(row.get("url"))
            if not (owner and repo and run_id):
                undecided_after_api.append(row)
                continue

            base = f"https://api.github.com/repos/{owner}/{repo}/actions"
            try:
                # Fetch jobs in the run
                r_jobs = requests.get(
                    f"{base}/runs/{run_id}/jobs?per_page=100",
                    headers=headers,
                    timeout=20,
                )
                jobs = (r_jobs.json().get("jobs", []) if r_jobs.ok else []) or []
                statuses    = [str(j.get("status") or "").lower() for j in jobs]
                conclusions = [str(j.get("conclusion") or "").lower() for j in jobs]

                fast_fail = (
                    any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions))
                    or ("cancelled" in conclusions and "completed" in statuses)
                    or any(co == "failure" for co in conclusions)
                )

                if fast_fail:
                    rr = dict(row); rr["conclusion"] = "failure"
                    inferred_resolved.append(rr)
                    json.dump(rr, streamf); streamf.write("\n")
                    time.sleep(REQ_DELAY)
                    continue

                # Run-level fallback (normalize cancelled/timed_out -> failure)
                r_run = requests.get(f"{base}/runs/{run_id}", headers=headers, timeout=20)
                rj = r_run.json() if r_run.ok else {}
                run_concl = normalize_run_level_conclusion(rj.get("conclusion"))

                if run_concl in ("failure", "success"):
                    rr = dict(row); rr["conclusion"] = run_concl
                    inferred_resolved.append(rr)
                    json.dump(rr, streamf); streamf.write("\n")
                else:
                    undecided_after_api.append(row)

            except requests.RequestException:
                undecided_after_api.append(row)

            time.sleep(REQ_DELAY)

    if inferred_resolved:
        jobs_results.extend(inferred_resolved)

    # ---------- 3) Replace caller's waiting list with final undecided rows ----------
    jobs_ids_await[:] = undecided_after_api
