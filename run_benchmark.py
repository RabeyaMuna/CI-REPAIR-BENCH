# ============================================================
#  run_benchmark.py — CI-Builds-Repair Benchmark Runner
# ============================================================

from benchmark import CIFixBenchmark
from benhmark_functions import fix_apply_generated_patch

# ============================================================
#  Configuration
# ============================================================
model_name = "diff"
config_path = (
    "/Users/rabeyakhatunmuna/Documents/"
    "Automated-CI-Build-Repair_with_benchmark/"
    "ci-builds-repair-benchmark/config.yaml"
)

# Initialize benchmark object
CIBenchPython = CIFixBenchmark(model_name, config_path)

# ============================================================
#  CHOOSE ONE DATASET OPTION
# ============================================================

# ---------- OPTION 1: Local Dataset ----------
# Uncomment this block if you already have a dataset locally
dataset_info = (
    "/Users/rabeyakhatunmuna/Documents/Automated-CI-Build-Repair_with_benchmark/dataset/lca_dataset.parquet"
)

# ---------- OPTION 2: Online Dataset ----------
# Uncomment this block if you want to fetch dataset from an online source (e.g., Hugging Face)
# dataset_info = "JetBrains-Research/lca-ci-builds-repair"  # or any other dataset name/id

# ============================================================
#  Run the Benchmark
# ============================================================
print(" Starting benchmark evaluation...")

CIBenchPython.eval_dataset(
    fix_repo_function=fix_apply_generated_patch,
    dataset_info=dataset_info,
    num_dp=None,           # Limit number of datapoints (optional)
    ids_list=None,         # Provide specific IDs if needed
    force_download=False   # Set True to re-download from online
)

# ============================================================
#  Get and Analyze Results
# ============================================================
CIBenchPython.get_results()

# ============================================================
#  Evaluate Jobs (Optional)
# ============================================================
# job_ids_file = "examples/jobs_ids.jsonl"
# job_results = CIBenchPython.eval_jobs(
#     job_ids_file=job_ids_file,
#     result_filename="jobs_results_test.jsonl",
# )

# ============================================================
#  Analyze Existing Results (Optional)
# ============================================================
# job_results_file = "examples/jobs_results.jsonl"
# CIBenchPython.analyze_results(jobs_results_file=job_results_file)

# ============================================================
#  End of Script
# ============================================================
pass
