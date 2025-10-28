import os
import requests
from omegaconf import OmegaConf
from dotenv import load_dotenv

# ----------------------------------------------------
# Load configuration and environment
# ----------------------------------------------------
load_dotenv()

CONFIG_PATH = "/Users/rabeyakhatunmuna/Documents/Automated-CI-Build-Repair_with_benchmark/ci-builds-repair-benchmark/config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

config = OmegaConf.load(CONFIG_PATH)

OWNER = config.get("benchmark_owner") or config.get("username_gh")
REPO = "OpenHands"  # Change if needed (e.g., "cloud-init", etc.)

if not OWNER:
    raise ValueError("Missing 'benchmark_owner' or 'username_gh' in config.yaml")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("Please set your GITHUB_TOKEN environment variable first!")

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/actions/runs"
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2022-11-28",
}

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def list_runs(status: str = "queued"):
    """
    Fetch all workflow runs with given status (e.g., queued, in_progress).
    """
    runs = []
    url = f"{BASE_URL}?status={status}"
    while url:
        r = requests.get(url, headers=HEADERS)
        if r.status_code != 200:
            print(f"[ERROR] Failed to fetch runs: {r.status_code} - {r.text}")
            break
        data = r.json()
        runs.extend(data.get("workflow_runs", []))
        url = r.links.get("next", {}).get("url")  # handle pagination
    return runs

def cancel_run(run_id: int):
    """
    Cancel a specific workflow run by ID.
    """
    url = f"{BASE_URL}/{run_id}/cancel"
    r = requests.post(url, headers=HEADERS)
    if r.status_code == 202:
        print(f"[CANCEL] Run {run_id} cancellation accepted.")
        return True
    elif r.status_code == 403:
        print(f"[WARN] Permission denied to cancel run {run_id}. (Possibly forked repo or insufficient token scope.)")
    else:
        print(f"[WARN] Failed to cancel run {run_id}: {r.status_code} - {r.text}")
    return False

def cancel_all_waiting_jobs():
    """
    Cancels all queued or in-progress workflow runs.
    """
    total_canceled = 0

    for status in ("queued", "in_progress"):
        runs = list_runs(status=status)
        if not runs:
            print(f"[INFO] No {status} runs found for {OWNER}/{REPO}.")
            continue

        print(f"[INFO] Found {len(runs)} runs with status='{status}' in {OWNER}/{REPO}.")
        for run in runs:
            run_id = run["id"]
            workflow_name = run.get("name", "unknown")
            print(f"[INFO] Attempting to cancel: {workflow_name} (run_id={run_id})")
            if cancel_run(run_id):
                total_canceled += 1

    print(f"[SUMMARY] Total canceled: {total_canceled} queued/running jobs.")

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------
if __name__ == "__main__":
    cancel_all_waiting_jobs()
