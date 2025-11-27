#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import requests
from benhmark_functions import get_results

# -----------------------------
# Debug helper
# -----------------------------
DEBUG_FAST_FAIL = True

def debug_log(msg: str, *args) -> None:
    """Simple debug logger for this module."""
    if not DEBUG_FAST_FAIL:
        return
    try:
        formatted = msg.format(*args)
    except Exception:
        # Fallback if formatting fails
        formatted = msg
    print(f"[fast_fail_detail] {formatted}", flush=True)

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
    REQ_DELAY = 0.8

    debug_log(
        "Starting finalize_after_last_poll: jobs_results={}, jobs_ids_await={}, jobs_ids_invalid={}",
        len(jobs_results),
        len(jobs_ids_await),
        len(jobs_ids_invalid),
    )
    debug_log("Streaming results path: {}", stream_results_path)

    # ---------- 1) Re-check all still-waiting once via get_results() ----------
    resolved_now = []
    still_waiting = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for job in list(jobs_ids_await):
            job_id = job.get("id")
            debug_log("Re-checking job (step1): id={}, raw_job={}", job_id, job)

            try:
                job_url, conclusion = get_results(job, self.config, self.credentials)
                debug_log(
                    "get_results -> job_id={}, url={}, conclusion={}",
                    job_id,
                    job_url,
                    conclusion,
                )
            except Exception as e:
                debug_log(
                    "get_results EXCEPTION for job_id={}: {} (marking as waiting)",
                    job_id,
                    repr(e),
                )
                job_url, conclusion = None, "waiting"

            conclusion = normalize_run_level_conclusion(conclusion)
            debug_log(
                "Normalized conclusion for job_id={}: {}",
                job_id,
                conclusion,
            )

            if conclusion in ("waiting", "queued", "in_progress", ""):
                row = dict(job)
                if job_url:
                    row["url"] = job_url
                row["conclusion"] = "waiting"
                still_waiting.append(row)
                debug_log(
                    "Job still waiting after get_results: id={}, url={}",
                    job_id,
                    row.get("url"),
                )

            elif conclusion == "error":
                rr = dict(job)
                rr["url"] = job_url
                rr["conclusion"] = "error"
                jobs_ids_invalid.append(rr)
                debug_log(
                    "Job marked as INVALID (error) after get_results: id={}, url={}",
                    job_id,
                    job_url,
                )

            else:
                rr = dict(job)
                rr["url"] = job_url
                rr["conclusion"] = conclusion  # success or failure
                resolved_now.append(rr)
                debug_log(
                    "Job resolved after get_results: id={}, conclusion={}, url={}",
                    job_id,
                    conclusion,
                    job_url,
                )
                json.dump(rr, streamf); streamf.write("\n")

            time.sleep(REQ_DELAY)

    if resolved_now:
        jobs_results.extend(resolved_now)
        debug_log(
            "Step1 complete: newly resolved count={}, jobs_results now={}",
            len(resolved_now),
            len(jobs_results),
        )
    else:
        debug_log("Step1 complete: no jobs resolved by get_results")

    debug_log("Jobs still waiting after step1: {}", len(still_waiting))

    # ---------- 2) For still-waiting rows, use GH API to infer fast-fail ----------
    token = (
        os.environ.get("GH_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or self.credentials.get("token")
    )
    if token:
        debug_log("GitHub token detected (not printing for safety).")
    else:
        debug_log("NO GitHub token detected, API calls may be rate-limited or fail.")

    headers = build_github_headers(token)

    inferred_resolved = []   # will include inferred failures and successes
    undecided_after_api = []

    with open(stream_results_path, "a", encoding="utf-8") as streamf:
        for row in still_waiting:
            job_id = row.get("id")
            debug_log("Step2 GH API check for job_id={}, row={}", job_id, row)

            owner, repo, run_id = parse_owner_repo_run_id(row.get("url"))
            debug_log(
                "Parsed owner/repo/run_id for job_id={}: owner={}, repo={}, run_id={}",
                job_id,
                owner,
                repo,
                run_id,
            )

            if not (owner and repo and run_id):
                debug_log(
                    "Could NOT parse run_id for job_id={} (url={}), keeping undecided.",
                    job_id,
                    row.get("url"),
                )
                undecided_after_api.append(row)
                continue

            base = f"https://api.github.com/repos/{owner}/{repo}/actions"
            try:
                # Fetch jobs in the run
                jobs_url = f"{base}/runs/{run_id}/jobs?per_page=100"
                debug_log("Requesting jobs list for job_id={} -> {}", job_id, jobs_url)
                r_jobs = requests.get(jobs_url, headers=headers, timeout=20)
                debug_log(
                    "Jobs HTTP status for job_id={}: {}",
                    job_id,
                    r_jobs.status_code,
                )

                jobs = (r_jobs.json().get("jobs", []) if r_jobs.ok else []) or []
                statuses    = [str(j.get("status") or "").lower() for j in jobs]
                conclusions = [str(j.get("conclusion") or "").lower() for j in jobs]

                debug_log(
                    "Job details for job_id={}: jobs_count={}, statuses={}, conclusions={}",
                    job_id,
                    len(jobs),
                    statuses,
                    conclusions,
                )

                fast_fail = (
                    any(st == "completed" and co == "failure" for st, co in zip(statuses, conclusions))
                    or ("cancelled" in conclusions and "completed" in statuses)
                    or any(co == "failure" for co in conclusions)
                )
                debug_log(
                    "Fast-fail inference for job_id={}: fast_fail={}",
                    job_id,
                    fast_fail,
                )

                if fast_fail:
                    rr = dict(row); rr["conclusion"] = "failure"
                    inferred_resolved.append(rr)
                    json.dump(rr, streamf); streamf.write("\n")
                    debug_log(
                        "Job inferred as FAILURE via fast-fail: job_id={}, row={}",
                        job_id,
                        rr,
                    )
                    time.sleep(REQ_DELAY)
                    continue

                # Run-level fallback (normalize cancelled/timed_out -> failure)
                run_url = f"{base}/runs/{run_id}"
                debug_log(
                    "Requesting run-level details for job_id={} -> {}",
                    job_id,
                    run_url,
                )
                r_run = requests.get(run_url, headers=headers, timeout=20)
                debug_log(
                    "Run HTTP status for job_id={}: {}",
                    job_id,
                    r_run.status_code,
                )

                rj = r_run.json() if r_run.ok else {}
                raw_run_concl = rj.get("conclusion")
                run_concl = normalize_run_level_conclusion(raw_run_concl)
                debug_log(
                    "Run-level conclusion for job_id={}: raw={}, normalized={}",
                    job_id,
                    raw_run_concl,
                    run_concl,
                )

                if run_concl in ("failure", "success"):
                    rr = dict(row); rr["conclusion"] = run_concl
                    inferred_resolved.append(rr)
                    json.dump(rr, streamf); streamf.write("\n")
                    debug_log(
                        "Job inferred from run-level: job_id={}, conclusion={}, row={}",
                        job_id,
                        run_concl,
                        rr,
                    )
                else:
                    undecided_after_api.append(row)
                    debug_log(
                        "Job still UNDECIDED after GH API: job_id={}, keeping in await",
                        job_id,
                    )

            except requests.RequestException as e:
                undecided_after_api.append(row)
                debug_log(
                    "RequestException for job_id={} during GH API checks: {}",
                    job_id,
                    repr(e),
                )

            time.sleep(REQ_DELAY)

    if inferred_resolved:
        jobs_results.extend(inferred_resolved)
        debug_log(
            "Step2 complete: inferred_resolved_count={}, jobs_results now={}",
            len(inferred_resolved),
            len(jobs_results),
        )
    else:
        debug_log("Step2 complete: no jobs inferred via GH API")

    debug_log(
        "Final undecided_after_api count (will remain in jobs_ids_await): {}",
        len(undecided_after_api),
    )

    # ---------- 3) Replace caller's waiting list with final undecided rows ----------
    jobs_ids_await[:] = undecided_after_api
    debug_log(
        "End finalize_after_last_poll: jobs_results={}, jobs_ids_await={}, jobs_ids_invalid={}",
        len(jobs_results),
        len(jobs_ids_await),
        len(jobs_ids_invalid),
    )
