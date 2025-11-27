import os
import pandas as pd
import json
import subprocess
from omegaconf import OmegaConf
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from utilities.llm_provider import get_llm
from ci_repair.ci_log_analyzer_bm25 import CILogAnalyzerBM25
from ci_repair.ci_log_analyzer_llm import CILogAnalyzerLLM
from ci_repair.fault_localization import FaultLocalization
from ci_repair.patch_generation import PatchGeneration
from utilities.ensure_repo import ensure_repo_at_commit

load_dotenv()

def process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm"):
    error_details = []
    fault_localization = []
    generated_patches = []
    results = []
    
    subset = dataset[318:]
    # target_ids = {241, 243, 281, 323}
    # subset = [dp for dp in dataset if dp.get("id") in target_ids]
    for datapoint in subset:
        task_id = datapoint["id"]
        repo_name = datapoint["repo_name"]
        repo_owner = datapoint["repo_owner"]
        repo_path = os.path.join(config.baseline_repo_folder, repo_name)
        head_branch = datapoint["head_branch"]
        sha_fail = datapoint["sha_fail"]
        benchmark_owner = config.benchmark_owner  
        repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
        logs = datapoint["logs"]
        workflow = datapoint["workflow"]
        workflow_path = datapoint["workflow_path"]
        sha_success = datapoint["sha_success"]
        
        result_dir = os.path.join(config.project_result_dir, model_key+"_"+log_analyzer_type)
        if not os.path.exists(result_dir):
          os.makedirs(result_dir, exist_ok=True)
        
        print(f" Proceed with the commit: {sha_fail}")
        
        ensure_repo_at_commit(repo_url, repo_path, sha_fail)
        try:
            if log_analyzer_type=="llm":
                log_analysis_result = CILogAnalyzerLLM(repo_path, logs, sha_fail, workflow, workflow_path, llm=llm, model_name=model_key).run()
            else:
                log_analysis_result = CILogAnalyzerBM25(repo_path, logs, sha_fail, workflow, workflow_path, llm=llm, model_name=model_key).run()
            
            error_details.append(log_analysis_result)

            with open(os.path.join(result_dir, "log_details.json"), "w") as f:
                json.dump(error_details, f, indent=4)
                
            if log_analysis_result ["relevant_files"] == []:
                print(f" No relevant files found for {sha_fail}, skipping...")
                continue

        except Exception as e:
            print(f" Failed processing {sha_fail} during error extraction: {e}")
            continue
        
        try:
            fault_localizer = FaultLocalization(
                                                sha_fail=sha_fail,
                                                repo_path=repo_path,
                                                error_logs=log_analysis_result,
                                                workflow=workflow,
                                                llm=llm
                                            ).run()
            
            fault_localization.append(fault_localizer)

            with open(os.path.join(result_dir, "fault_localization.json"), "w") as f:
                json.dump(fault_localization, f, indent=4)
                
            if fault_localizer["fault_localization_data"] == []:
                print(f" No suspicious files found for {sha_fail}, skipping...")
                continue

        except Exception as e:
            print(f" Failed processing {sha_fail} during error extraction: {e}")
            continue
        
        try:
            patch_generator = PatchGeneration(bug_report=fault_localizer, repo_path=repo_path, task_id=task_id,
            error_details=log_analysis_result, workflow_path=workflow_path, workflow=workflow, llm=llm).run()
            
            if patch_generator["diff"] =="":
                print(f" No patch generated for {sha_fail}")
                continue

            generated_patches.append(patch_generator)

            with open(os.path.join(result_dir, "generated_patches.json"), "w") as f:
                json.dump(generated_patches, f, indent=4)

        except Exception as e:
            print(f" Failed processing {sha_fail} during error extraction: {e}")
            continue
        
    results = generated_patches
    return results

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    config = OmegaConf.load(config_path)
    # Construct dataset path dynamically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # one level up from ci-build-repair-project
    dataset_path = os.path.join(base_dir, "dataset", "lca_dataset.parquet")
    model_key = "gpt-5-mini"   # or "gpt4o", "deepseek-chat", etc.
    llm = get_llm(model_key)
    # Load dataset
    dataset_df = pd.read_parquet(dataset_path)
    dataset = dataset_df.to_dict(orient="records")

    results = process_entire_dataset(dataset, config, llm, model_key, log_analyzer_type="llm")

    output_file = os.path.join(config.project_result_dir, "generated_patches.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved in {output_file}")
