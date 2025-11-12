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
        
# import json
# import os
# import shutil
# from omegaconf import OmegaConf
# from load_config import load_config

# config, CONFIG_PATH = load_config()

# def _normalize_fastfail_conclusion(records):
#     """
#     Fast-fail policy (failure-only normalization):
#       - If ANY job is completed AND conclusion == 'failure' => mark run 'failure'.
#       - Else, if fast-fail pattern observed: ANY job is 'cancelled' AND at least one job is 'completed' => mark run 'failure'.
#       - Else, if ANY job (even if not completed) has conclusion == 'failure' => mark run 'failure'.
#       - Else, leave record unchanged.  (No success normalization here.)
#     """
#     out = []
#     for r in records:
#         jobs = r.get("jobs") or r.get("job_list") or []
#         if not jobs:
#             out.append(r)
#             continue

#         # Normalize values (case-insensitive)
#         statuses    = [((j or {}).get("status") or "").lower() for j in jobs]
#         conclusions = [((j or {}).get("conclusion") or "").lower() for j in jobs]

#         # 1) Immediate failure if any completed job failed
#         completed_failure = any(st == "completed" and co == "failure"
#                                 for st, co in zip(statuses, conclusions))
#         if completed_failure:
#             rr = dict(r)
#             rr["conclusion"] = "failure"
#             out.append(rr)
#             continue

#         # 2) Fast-fail pattern: jobs cancelled because a sibling failed
#         #    (often shows up as some 'cancelled' and at least one 'completed' in the same snapshot)
#         has_cancelled  = "cancelled" in conclusions
#         any_completed  = "completed" in statuses
#         if has_cancelled and any_completed:
#             rr = dict(r)
#             rr["conclusion"] = "failure"
#             out.append(rr)
#             continue

#         # 3) Otherwise, any (not-yet-completed) job already marked as failure
#         any_failure = any(co == "failure" for co in conclusions)
#         if any_failure:
#             rr = dict(r)
#             rr["conclusion"] = "failure"
#             out.append(rr)
#             continue

#         # 4) No failures detected -> leave as-is (success logic remains unchanged)
#         out.append(r)

#     return out


# def read_jsonl(file_path):
#     data = []
#     with open(file_path, "r") as f:
#         for line in f:
#             data.append(json.loads(line))
#     return data


# def save_jsonl(file_path, data):
#     with open(file_path, "w") as f:
#         for entry in data:
#             json.dump(entry, f)
#             f.write("\n")


# def _safe_read_jsonl(path):
#     """Read JSONL if present; otherwise return empty list."""
#     if not os.path.exists(path):
#         return []
#     return read_jsonl(path)


# def filter_out_res(
#     data_folder,
#     out_folder,
#     jobs_error_path=os.path.join(config.out_folder, "jobs_error_diff.jsonl")
# ):
#     """
#     Filter according to results benchmarks.

#     Keeps datapoints whose sha_original (7 chars) appears as:
#     - failure in jobs_results_none.jsonl  (original set), AND
#     - success in jobs_results_diff.jsonl  OR error in jobs_error_diff.jsonl (fixed/error set)

#     Copies matching <sha7>.json from datapoints_json_verified → datapoints_json_filtered.
#     """
#     # Inputs
#     results_none_path = os.path.join(out_folder, "jobs_results_none.jsonl")
#     results_diff_path = os.path.join(out_folder, "jobs_results_diff.jsonl")

#     results_none = _safe_read_jsonl(results_none_path)
#     results_none = _normalize_fastfail_conclusion(results_none)
#     results_diff = _safe_read_jsonl(results_diff_path)
#     results_error = _safe_read_jsonl(jobs_error_path)

#     # Folders
#     orig_path = os.path.join(data_folder, "datapoints_json_verified")
#     filtered_path = os.path.join(data_folder, "datapoints_json_filtered")
#     os.makedirs(filtered_path, exist_ok=True)

#     # Sets
#     original_sha = {
#         r["sha_original"][:7]
#         for r in results_none
#         if r.get("conclusion") == "failure" and "sha_original" in r
#     }

#     fixed_sha_success = {
#         r["sha_original"][:7]
#         for r in results_diff
#         if r.get("conclusion") == "success" and "sha_original" in r
#     }

#     fixed_sha_error = {
#         r["sha_original"][:7]
#         for r in results_error
#         if r.get("conclusion") == "error" and "sha_original" in r
#     }

#     # Consider success OR error as "fixed/error side present"
#     fixed_or_error = fixed_sha_success | fixed_sha_error

#     sha_valid = original_sha & fixed_or_error

#     copied = 0
#     missing = []
#     for sha in sha_valid:
#         dp_file = os.path.join(orig_path, f"{sha}.json")
#         dp_filtered = os.path.join(filtered_path, f"{sha}.json")
#         if os.path.exists(dp_file):
#             shutil.copy2(dp_file, dp_filtered)
#             copied += 1
#         else:
#             missing.append(sha)

#     # Optional: simple summary print (remove if you don’t want stdout)
#     print(
#         f"[filter_out_res] originals(failure)={len(original_sha)}, "
#         f"success={len(fixed_sha_success)}, error={len(fixed_sha_error)}, "
#         f"kept={len(sha_valid)}, copied={copied}, missing_source={len(missing)}"
#     )
#     if missing:
#         print(f"[filter_out_res] Missing source JSON for: {sorted(missing)[:10]}{' ...' if len(missing) > 10 else ''}")
