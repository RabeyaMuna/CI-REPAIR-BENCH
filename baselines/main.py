import os
import pandas as pd
import json
import subprocess
from omegaconf import OmegaConf
from dotenv import load_dotenv
from datasets import load_dataset
from ci_repair.ci_log_analyzer import CILogAnalyzer
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()

def process_entire_dataset(dataset, config):
    error_details = []
    results = []
    
    for datapoint in dataset:
        task_id = datapoint["id"]
        repo_name = datapoint["repo_name"]
        repo_owner = datapoint["repo_owner"]
        repo_path = os.path.join(config.repos_folder, repo_name)
        head_branch = datapoint["head_branch"]
        sha_fail = datapoint["sha_fail"]
        benchmark_owner = config.benchmark_owner  
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]
        sha_success = datapoint["sha_success"]
        
        print(f" Proceed with the commit: {sha_fail}")
        
        ensure_repo_at_commit(repo_url, repo_path, sha_fail)

        try:
            log_analysis_result = CILogAnalyzer(repo_path, logs, sha_fail, workflow, workflow_path).run()
            
            error_details.append(log_analysis_result)

            with open(os.path.join(config.project_result_dir, "log_details.json"), "w") as f:
                json.dump(error_details, f, indent=4)

        except Exception as e:
            print(f" Failed processing {sha_fail} during error extraction: {e}")
            continue


    return results


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)
    # Construct dataset path dynamically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from ci-build-repair-project
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")

    # Load dataset
    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    results = process_entire_dataset(dataset, config)

    output_file = os.path.join(config.result_dir, "generated_patches.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved in {output_file}")
