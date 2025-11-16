import json
import os
import time
from typing import Optional, List
from dotenv import load_dotenv
import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm
from typing import List
from fast_fail_detail import finalize_after_last_poll
from benchmark_utils import read_jsonl, save_jsonl
from benhmark_functions import get_results, process_datapoint

load_dotenv()

def filter_files(directory, files):
    return [file for file in files if file != "meta_info.json"]

def filter_by_id(example, ids):
    return example['id'] in ids

class CIFixBenchmark:
    def __init__(self, model_name, config_path):

        self.dataset_id = "JetBrains-Research/lca-ci-builds-repair"

        # Loads a YAML/JSON configuration file using OmegaConf, a part of the Hydra library.
        self.config = OmegaConf.load(config_path)
        benchmark_owner = self.config.get("benchmark_owner", "RabeyaMuna")
        
        if not "test_username" in self.config:
            self.config.test_username = self.config.username_gh
            
        language = self.config.language
        
        self.credentials = {
            "username": self.config.username_gh,
            "token": os.environ.get("GITHUB_TOKEN"),
            "model": model_name,
        }

        os.makedirs(self.config.out_folder, exist_ok=True)
        os.makedirs(self.config.repos_folder, exist_ok=True)
        OmegaConf.update(
            self.config, "benchmark_owner", benchmark_owner, force_add=True
        )
        
        if hasattr(self.config, "data_cache_dir"):
            self.cache_dir = self.config.data_cache_dir
        else:
            self.cache_dir = None
        self.model_name = model_name

    def get_dataset(
        self,
        dataset_info: Optional[str] = None,
        num_dp: Optional[int] = None,
        force_download: bool = False,
    ):
        """
        Load dataset dynamically:
        - If dataset_info points to a valid local path → load locally.
        - If dataset_info is an online dataset name → fetch online.
        - If None → load default online dataset.
        """

        if dataset_info:
            # Case 1: Local file or directory
            if os.path.exists(dataset_info) and os.path.isfile(dataset_info):
                print(f"Loading local dataset from: {dataset_info}")
                df = pd.read_parquet(dataset_info)
                self.dataset = pd.DataFrame(df).to_dict(orient="records")

            # Case 2: Online dataset
            else:
                print(f" Fetching online dataset: {dataset_info}")
                self.dataset = load_dataset(
                    dataset_info,
                    cache_dir=self.cache_dir,
                    download_mode="force_redownload" if force_download else None,
                    split="test",
                )

        return self.dataset

    def run_dataset(self, fix_repo_function, test_dataset=None):        
        if test_dataset is None:
            test_dataset = self.dataset
        self.jobs_ids = []
        jobs_ids_file_path = os.path.join(
            self.config.out_folder, f"jobs_ids_{self.model_name}.jsonl"
        )
        
        with open(jobs_ids_file_path, "w") as writer:
            for datapoint in tqdm(test_dataset):
                job_identificator = process_datapoint(
                    datapoint, fix_repo_function, self.config, self.credentials
                )

                self.jobs_ids.append(job_identificator)
                json.dump(job_identificator, writer)
                writer.write("\n")
                
        return self.jobs_ids

    def eval_jobs(self, jobs_ids=None, job_ids_file=None, result_filename=None):
        """
        Evaluate all submitted jobs by polling all GitHub Actions results periodically.
        Fetches all jobs each cycle but includes delay and rate-limit protection
        to avoid exceeding GitHub’s 5000 req/hour limit.
        """

        WAIT_INTERVAL = 900       # 10 minutes between polling cycles
        MAX_ATTEMPTS = 12         # total 2-hour window
        REQ_DELAY = 0.8           # ~0.8s between requests → 4500 req/hour safe margin

        if result_filename is None:
            result_filename = f"jobs_results_{self.model_name}.jsonl"

        jobs_results_file_path = os.path.join(self.config.out_folder, result_filename)
        jobs_awaiting_file_path = os.path.join(
            self.config.out_folder, f"jobs_awaiting_{self.model_name}.jsonl"
        )
        jobs_invalid_file_path = os.path.join(
            self.config.out_folder, f"jobs_invalid_{self.model_name}.jsonl"
        )

        result_file = open(jobs_results_file_path, "w")

        # Load job IDs
        if job_ids_file is not None:
            jobs_ids = read_jsonl(job_ids_file)
        elif jobs_ids is None:
            jobs_ids = self.jobs_ids

        jobs_ids_await = jobs_ids
        n_attempts = 0
        jobs_results = []
        jobs_ids_invalid = []

        print(f"Starting evaluation: {len(jobs_ids_await)} jobs to check")

        while len(jobs_ids_await) > 0 and n_attempts < MAX_ATTEMPTS:
            n_attempts += 1
            print(f"\nCycle {n_attempts}: polling all {len(jobs_ids_await)} jobs")

            new_waiting = []
            start_time = time.time()

            for job_id in jobs_ids_await:
                try:
                    job_url, conclusion = get_results(job_id, self.config, self.credentials)
                except Exception as e:
                    print(f"Warning: API error for {job_id.get('repo_name')}: {e}")
                    new_waiting.append(job_id)
                    time.sleep(REQ_DELAY)
                    continue

                # Detect API rate limit response
                if isinstance(job_url, dict) and "API rate limit" in str(job_url):
                    print("Rate limit reached. Sleeping 15 minutes before retrying...")
                    time.sleep(900)
                    new_waiting.append(job_id)
                    continue

                # Categorize job state
                if conclusion == "waiting":
                    new_waiting.append(job_id)
                elif conclusion == "error":
                    jobs_ids_invalid.append(job_id)
                else:
                    job_id["url"] = job_url
                    job_id["conclusion"] = conclusion
                    jobs_results.append(job_id)
                    json.dump(job_id, result_file)
                    result_file.write("\n")

                # Delay between requests to avoid hitting limit
                time.sleep(REQ_DELAY)

            jobs_ids_await = new_waiting

            # Save intermediate states
            save_jsonl(jobs_awaiting_file_path, jobs_ids_await)
            save_jsonl(jobs_invalid_file_path, jobs_ids_invalid)

            elapsed = (time.time() - start_time) / 60
            print(f"Cycle {n_attempts} done in {elapsed:.1f} minutes.")
            print(f"Results: {len(jobs_results)} success, "
                f"{len(jobs_ids_invalid)} invalid, {len(jobs_ids_await)} waiting.")

            # Wait before next cycle if needed
            if len(jobs_ids_await) > 0 and n_attempts < MAX_ATTEMPTS:
                print(f"Sleeping {WAIT_INTERVAL/60:.0f} minutes before next cycle...")
                time.sleep(WAIT_INTERVAL)

        result_file.close()
        
        finalize_after_last_poll(
            self,
            jobs_results=jobs_results,
            jobs_ids_await=jobs_ids_await,
            jobs_ids_invalid=jobs_ids_invalid,
            stream_results_path=jobs_results_file_path,
        )

        print("\nFinal summary:")
        print(f"Completed: {len(jobs_results)}")
        print(f"Invalid: {len(jobs_ids_invalid)}")
        print(f"Still waiting: {len(jobs_ids_await)} after {MAX_ATTEMPTS} attempts")

        self.jobs_results = jobs_results
        return jobs_results

    def get_results(self, job_ids_file=None, result_filename=None):
        if job_ids_file is None:
            job_ids_file = os.path.join(
                self.config.out_folder, f"jobs_ids_{self.model_name}.jsonl"
            )
        
        if result_filename is None:
            result_filename = f"jobs_results_{self.model_name}.jsonl"
            result_file = os.path.join(self.config.out_folder, result_filename)
            
        self.analyze_results(jobs_results_file=result_file)

    def analyze_results(self, jobs_results=None, jobs_results_file=None):
        """
        Analyze benchmark results and print summary:
        1. Pass/fail summary for evaluated subset
        2. Pass ratio relative to total dataset size (if dataset loaded)
        """
        # Load results
        if jobs_results_file is not None:
            jobs_results = read_jsonl(jobs_results_file)
        if jobs_results is None:
            jobs_results = self.jobs_ids

        results_df = pd.DataFrame(jobs_results)
        total_evaluated = len(results_df)
        total_dataset_size = len(self.dataset) if hasattr(self, "dataset") and self.dataset is not None else None

        total_counts = results_df["conclusion"].value_counts()
        total_pass = total_counts.get("success", 0)
        total_fail = total_counts.get("failure", 0)
        total_ratio = total_pass / total_evaluated if total_evaluated > 0 else 0

        print("\n===============================")
        print("Evaluation Summary")
        print("===============================")
        print(f"Evaluated datapoints: {total_evaluated}")
        print(f"Passed: {total_pass}")
        print(f"Failed: {total_fail}")
        print(f"Pass ratio (subset): {total_ratio:.2%}")

        # Global dataset context
        if total_dataset_size:
            global_ratio = total_pass / total_dataset_size
            print("\nOverall Context")
            print(f"Total dataset size: {total_dataset_size}")
            print(f"Passes relative to full dataset: {total_pass}/{total_dataset_size}")
            print(f"Global pass ratio: {global_ratio:.2%}")

    def eval_dataset(
        self,
        fix_repo_function,
        dataset_info: Optional[str] = None,
        num_dp: Optional[int] = None,
        ids_list: Optional[List] = None,
        force_download: bool = False,
        result_filename: Optional[str] = None,
    ):
        """
        Evaluate the benchmark on either a local or online dataset.
        """
        print("---------------- Loading dataset -------------------")
        self.get_dataset(
            dataset_info=dataset_info,
            num_dp=num_dp,
            force_download=force_download,
        )

        if ids_list is not None:
            self.dataset = self.dataset.filter(lambda example: filter_by_id(example, ids_list))

        print(f"Got {len(self.dataset)} datapoints to evaluate")
        print("---------------- Running datapoints -------------------")
        self.run_dataset(fix_repo_function)

        print("---------------- Getting results -------------------")
        self.eval_jobs(result_filename=result_filename)
        self.analyze_results()

    def run_datapoint(self, datapoint, fix_repo_function):
        # This method is for debugging reasons
        jobs_ids_file_path = os.path.join(
            self.config.out_folder, f"jobs_ids_{self.model_name}.jsonl"
        )
        with open(jobs_ids_file_path, "w") as writer:
            job_identificator = process_datapoint(
                datapoint, fix_repo_function, self.config, self.credentials
            )
            json.dump(job_identificator, writer)
            writer.write("\n")
            
        return job_identificator

    def eval_datapoint(self, job_identificator):
        # This method is for debugging reasons
        pass
