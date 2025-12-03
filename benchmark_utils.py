import json
import os
import shutil
from omegaconf import OmegaConf
from load_config import load_config

config, CONFIG_PATH = load_config()

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(file_path, data):
    with open(file_path, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def _safe_read_jsonl(path):
    """Read JSONL if present; otherwise return empty list."""
    if not os.path.exists(path):
        return []
    
    return read_jsonl(path)


def filter_out_res(
    data_folder,
    out_folder,
    jobs_error_path=os.path.join(config.out_folder, "jobs_error_diff.jsonl")
):
    """
    Filter according to results benchmarks.

    Keeps datapoints whose sha_original (7 chars) appears as:
    - failure in jobs_results_none.jsonl  (original set), AND
    - success in jobs_results_diff.jsonl  OR error in jobs_error_diff.jsonl (fixed/error set)

    Copies matching <sha7>.json from datapoints_json_verified → datapoints_json_filtered.
    """
    # Inputs
    results_none_path = os.path.join(out_folder, "jobs_results_none.jsonl")
    results_diff_path = os.path.join(out_folder, "jobs_results_diff.jsonl")

    results_none = _safe_read_jsonl(results_none_path)
    results_diff = _safe_read_jsonl(results_diff_path)
    results_error = _safe_read_jsonl(jobs_error_path)

    # Folders
    orig_path = os.path.join(data_folder, "datapoints_json_verified")
    filtered_path = os.path.join(data_folder, "datapoints_json_filtered")
    os.makedirs(filtered_path, exist_ok=True)

    # Sets
    original_sha = {
        r["sha_original"][:7]
        for r in results_none
        if r.get("conclusion") == "failure" and "sha_original" in r
    }

    fixed_sha_success = {
        r["sha_original"][:7]
        for r in results_diff
        if r.get("conclusion") == "success" and "sha_original" in r
    }

    fixed_sha_error = {
        r["sha_original"][:7]
        for r in results_error
        if r.get("conclusion") == "error" and "sha_original" in r
    }

    # Consider success OR error as "fixed/error side present"
    fixed_or_error = fixed_sha_success | fixed_sha_error

    sha_valid = original_sha & fixed_or_error

    copied = 0
    missing = []
    for sha in sha_valid:
        dp_file = os.path.join(orig_path, f"{sha}.json")
        dp_filtered = os.path.join(filtered_path, f"{sha}.json")
        if os.path.exists(dp_file):
            shutil.copy2(dp_file, dp_filtered)
            copied += 1
        else:
            missing.append(sha)

    # Optional: simple summary print (remove if you don’t want stdout)
    print(
        f"[filter_out_res] originals(failure)={len(original_sha)}, "
        f"success={len(fixed_sha_success)}, error={len(fixed_sha_error)}, "
        f"kept={len(sha_valid)}, copied={copied}, missing_source={len(missing)}"
    )
    if missing:
        print(f"[filter_out_res] Missing source JSON for: {sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
    