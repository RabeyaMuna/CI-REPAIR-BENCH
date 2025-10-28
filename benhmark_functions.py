import os
import re
import json
import git
import requests
import subprocess
from git import GitCommandError
from ruamel.yaml import YAML


def edit_workflow_push(workflow_file):
    """
    editing workflow.yaml so, that it would be run on push
    """

    yaml = YAML()
    with open(workflow_file, "r") as file:
        yaml_data = yaml.load(file)

    yaml_data["on"] = "push"

    with open(workflow_file, "w") as file:
        yaml.dump(yaml_data, file)

def extract_referenced_workflows(workflow_file):
    """Find reusable workflow references to preserve them."""
    yaml = YAML()
    with open(workflow_file, "r") as file:
        yaml_data = yaml.load(file)
    referenced = set()

    def _scan(value):
        if isinstance(value, str):
            matches = re.findall(r"\.github/workflows/([\w\-.\/]+(?:\.yml|\.yaml))", value)
            for m in matches:
                referenced.add(os.path.basename(m))
        elif isinstance(value, list):
            for item in value:
                _scan(item)
        elif isinstance(value, dict):
            for _, v in value.items():
                _scan(v)

    _scan(yaml_data)
    return referenced

def delete_unreferenced_workflows(workflow_dir, referenced_files):
    """Delete only workflows not referenced anywhere."""
    existing_files = [
        f for f in os.listdir(workflow_dir)
        if os.path.isfile(os.path.join(workflow_dir, f)) and f.endswith((".yml", ".yaml"))
    ]
    for filename in existing_files:
        if filename not in referenced_files:
            try:
                os.remove(os.path.join(workflow_dir, filename))
            except Exception as e:
                print(f"[WARN] Could not delete {filename}: {e}")

def detect_and_normalize_runners(workflow_file, replace=True):
    """
    Detects deprecated GitHub-hosted runners in a workflow YAML file.
    If `replace=True`, fixes them in place while preserving YAML formatting.
    Returns:
        True  → if deprecated runners were found (and optionally replaced)
        False → if no deprecated runners were found
    """
    yaml = YAML()

    with open(workflow_file, "r", encoding="utf-8") as f:
        yaml_data = yaml.load(f)

    if not isinstance(yaml_data, dict) or "jobs" not in yaml_data:
        return False

    replacement_map = {
        "ubuntu-16.04": "ubuntu-22.04",
        "ubuntu-18.04": "ubuntu-22.04",
        "ubuntu-20.04": "ubuntu-latest",
        "windows-2019": "windows-latest",
        "macos-13": "macos-latest",
        "ubuntu-latest-unit-tester": "ubuntu-latest"
    }

    deprecated_found = False

    for job_name, job in yaml_data.get("jobs", {}).items():
        # --- Case 1: runs-on ---
        runner = job.get("runs-on")
        if isinstance(runner, str):
            lower = runner.lower()
            if lower in replacement_map:
                print(f"[FOUND] Deprecated runner in job '{job_name}': {runner}")
                deprecated_found = True
                if replace:
                    job["runs-on"] = replacement_map[lower]
                    print(f"[FIXED] '{job_name}' runs-on → {job['runs-on']}")
        elif isinstance(runner, list):
            for i, r in enumerate(runner):
                lower = str(r).lower()
                if lower in replacement_map:
                    print(f"[FOUND] Deprecated runner in job '{job_name}': {r}")
                    deprecated_found = True
                    if replace:
                        job["runs-on"][i] = replacement_map[lower]
                        print(f"[FIXED] '{job_name}' runs-on → {job['runs-on'][i]}")

        # --- Case 2: strategy.matrix.os ---
        strategy = job.get("strategy")
        if strategy and isinstance(strategy, dict):
            matrix = strategy.get("matrix")
            if matrix and isinstance(matrix, dict) and "os" in matrix:
                os_list = matrix["os"]
                if isinstance(os_list, list):
                    for i, os_val in enumerate(os_list):
                        lower = str(os_val).lower()
                        if lower in replacement_map:
                            print(f"[FOUND] Deprecated matrix OS in job '{job_name}': {os_val}")
                            deprecated_found = True
                            if replace:
                                matrix["os"][i] = replacement_map[lower]
                                print(f"[FIXED] '{job_name}' matrix.os → {matrix['os'][i]}")

    # Only rewrite file if replacements were made
    if deprecated_found:
        with open(workflow_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f)

    return deprecated_found



def copy_and_edit_workflow_file(datapoint, repo):
    """
    Copies and minimally edits a workflow file:
      - Adds 'push' trigger if missing.
      - Normalizes deprecated runners only if found.
      - Keeps formatting, comments, and structure identical.
    """

    workflow_dir = os.path.join(repo.working_dir, ".github", "workflows")
    os.makedirs(workflow_dir, exist_ok=True)

    workflow_path = datapoint.get("workflow_path")
    if not workflow_path or not os.path.isfile(os.path.join(repo.working_dir, workflow_path)):
        print(f"[WARN] Workflow path invalid or missing for {datapoint['id']}: {workflow_path}")
        return

    target_file = os.path.join(workflow_dir, os.path.basename(workflow_path))

    # --- 1. Normalize runners only if deprecated found
    # yaml_data, deprecated_found = detect_and_normalize_runners(target_file, replace=True)
    # --- 2. Add push trigger (preserves all formatting)
    edit_workflow_push(target_file)
    reference_files = extract_referenced_workflows(target_file)

    reference_files.add(os.path.basename(target_file))

    delete_unreferenced_workflows(workflow_dir, reference_files)

    print(f"[INFO] Workflow updated: {target_file}")


def rename_precommit_files(repo_path):
    """
    rename pre-commit.yaml, so it will be run on push
    """
    workflow_dir = os.path.join(repo_path, ".github/workflows")
    
    for filename in os.listdir(workflow_dir):
        file_path = os.path.join(workflow_dir, filename)
        if os.path.isfile(file_path):
            if "pre-commit" in filename.lower():
                os.rename(
                    file_path, file_path.lower().replace("pre-commit", "precommit")
                )


def push_repo(repo, credentials, benchmark_owner, user_branch_name):
    """
    Pushes the corrected repo, return commit sha to use it for getting results
    """

    # TODO think about adding only changed files
    repo.git.add(".")
    repo.git.add(update=True)
    repo.index.commit(user_branch_name)
    username = credentials["username"]
    token = credentials["token"]
    try:
        repo.delete_remote("origin")
    except:
        pass
    origin_url = (
        f"https://{username}:{token}@github.com/{benchmark_owner}/{repo.name}.git"
    )
    origin = repo.create_remote("origin", url=origin_url)
    repo.git.push("--force", "--set-upstream", origin, repo.head.ref)
    # Tried this, but it did not work - returned an error
    """
    cmdline: git push -u origin test_user
    stderr: 'gh auth git-credential: "erase" operation not supported
    remote: Invalid username or password.
    """
    # origin = repo.remote("origin")
    # with repo.git.custom_environment(GIT_USERNAME=username, GIT_PASSWORD=token):
    #     repo.git.push("-u", "origin", "test_user")
    #     origin.push()
    
    commit_hash = repo.head.commit.hexsha
    return commit_hash


def get_repo(datapoint, repos_folder, test_username, benchmark_owner, credentials):
    """
    clones repo and switches it to the required commit
    creates branch with username
    """
     
    id = datapoint["id"]
    username = credentials["username"]
    token = credentials["token"]
    model_name = credentials["model"]
    repo_name, repo_owner = datapoint["repo_name"], datapoint["repo_owner"]
    # TODO add original branch name to new_branch_name?
    new_branch_name = f"{test_username}__{model_name}__id_{id}"
    commit_hash = datapoint["sha_fail"]
    repo_path = os.path.join(repos_folder, f"{repo_owner}__{repo_name}")
    repo_url = f"https://github.com/{benchmark_owner}/{repo_name}.git"
    origin_url = (
        f"https://{username}:{token}@github.com/{benchmark_owner}/{repo_name}.git"
    )
    if (not os.path.exists(repo_path)) or (not os.listdir(repo_path)):
        repo = git.Repo.clone_from(repo_url, repo_path, depth=1)  # branch=commit_hash
    else:
        repo = git.Repo(repo_path)
    try:
        origin = repo.remote("origin")
    except:
        origin = repo.create_remote("origin", url=origin_url)
    repo.git.fetch("origin", commit_hash)
    try:
        repo.git.reset("--hard", commit_hash)
    except Exception as e:
        print(e)
        repo.git.checkout(commit_hash)
    # remove excessive files
    repo.git.clean("-fdx")
    if not any((h for h in repo.heads if h.name == new_branch_name)):
        # repo.delete_head("test_user", force=True)
        repo.create_head(new_branch_name, force=True)
    # TODO note that you should ban usage of the .git folder.
    # You need flag "-B" to checkout to the current state. Otherwise, the old brach state would be used
    repo.git.checkout("-B", new_branch_name)
    repo.name, repo.owner = repo_name, repo_owner
    
    return repo, new_branch_name


def get_run_data(repo_name, commit_sha, credentials, config):
    token = credentials["token"]
    headers = {"Authorization": f"token {token}"}

    jobs_url = f"https://api.github.com/repos/{config.benchmark_owner}/{repo_name}/commits/{commit_sha}/check-runs"
    response = requests.get(jobs_url, headers=headers)
    data = response.json()
    try:
        run_url = data["check_runs"][0]["html_url"]
        job_url = "/".join(run_url.split("/")[:-2])
        conclusions = [run["conclusion"] for run in data["check_runs"]]
        statuses = [run["status"] for run in data["check_runs"]]
        completed = [status == "completed" for status in statuses]
    except:
        print(f"Error in requesting jobs url {jobs_url}")
        print(data)
        job_url = ""
        conclusion = "error"
        return job_url, conclusion

    if not all(completed):
        conclusion = "waiting"
    elif "failure" in conclusions:
        conclusion = "failure"
    elif all([conclusion == "success" for conclusion in conclusions]):
        conclusion = "success"
    else:
        log_file_path = os.path.join(config.out_folder, "out_logs.txt")
        with open(log_file_path, "a") as f:
            f.write("--------------------DP BEGIN----------------------- \n")
            f.write(str(statuses) + "\n")
            f.write(str(conclusions) + "\n")
            f.write(str(data) + "\n")
            f.write("---------------------DP END------------------------- \n")
            conclusion = "error"
    
    return job_url, conclusion


def fix_none(datapoint, repo_path, repo=None, out_folder=None):
    return None


def fix_apply_diff(datapoint, repo_path, repo, out_folder):
    commit_sha = datapoint["sha_fail"][:7]
    diff_path = os.path.join(out_folder, f"{commit_sha}.diff")
    with open(diff_path, "w") as f:
        f.write(datapoint["diff"])
    
    try:
        repo.git.apply(diff_path)
    except GitCommandError as err:
        print(f"Sha = {datapoint['sha_fail']}")
        print(f"An error occurred while running the git command: {err}")
    os.remove(diff_path)
    return None


def process_datapoint(datapoint, fix_repo_function, config, credentials):
    """
    fix_repo_function - function that takes repo path and datapoint, repo object and out_folder.
    it should edit the repo in the folder, nothing to return
    credentials are passed in the following format:
    {'token': token, 'username': username}
    """

    # TODO think, what to do if test_username (which converts to a branch) is already present
    repo, user_branch_name = get_repo(
        datapoint,
        config.repos_folder,
        config.test_username,
        config.benchmark_owner,
        credentials,
    )

    # Prepares workflow file Moves target workflow file to the .github/workflows
    copy_and_edit_workflow_file(datapoint, repo)
    # Fixing the repo. fix_repo_function is provided by user.
    fix_repo_function(datapoint, repo.working_dir, repo, config.out_folder)
    # Push the corrected repo
    commit_sha = push_repo(repo, credentials, config.benchmark_owner, user_branch_name)
    job_identificator = {
        "repo_name": repo.name,
        "commit": commit_sha,
        "id": datapoint["id"],
        "sha_original": datapoint["sha_fail"],
        "branch_name": user_branch_name,
        "workflow": datapoint.get("workflow_path", "")
    }

    return job_identificator


def get_results(job_identificator, config, credentials):
    # We have to make some pause to get result or even url, unless it sees no runs
    repo_name = job_identificator["repo_name"]
    commit_sha = job_identificator["commit"]
    job_url, conclusion = get_run_data(repo_name, commit_sha, credentials, config)
    
    return job_url, conclusion


def dataset_to_json(dataset):
    json_list = []
    for item in dataset:
        json_list.append(item)
    
    return json_list

def fix_apply_generated_patch(datapoint, repo_path, repo, out_folder):
    patch_file = os.path.join(out_folder, "generated_patches.json")

    if not os.path.exists(patch_file):
        raise FileNotFoundError(f"[ERROR] Patch file not found: {patch_file}")

    # Load all patches
    with open(patch_file, "r", encoding="utf-8") as f:
        patches = json.load(f)

    current_id = datapoint["id"]
    patch_data = next((p for p in patches if p["id"] == current_id and p.get("diff", "").strip()), None)

    if not patch_data:
        print(f"[SKIP] No patch found for ID {current_id}")
        return

    temp_diff_path = os.path.join(out_folder, f"temp_{current_id}.diff")

    # Write patch to temp file
    with open(temp_diff_path, "w", encoding="utf-8") as f:
        f.write(patch_data["diff"])

    # Pre-check patch validity using --3way
    check = subprocess.run(
        ["git", "apply", "--check", "--3way", temp_diff_path],
        cwd=repo_path,
        capture_output=True,
        text=True
    )

    if check.returncode != 0:
        print(f"[SKIP] Patch for ID {current_id} failed `git apply --check --3way`")
        print(f"[DEBUG] Patch Content:\n{patch_data['diff']}")
        print(f"[DEBUG] Git Error:\n{check.stderr.strip()}")
        os.remove(temp_diff_path)
        return

    # Apply patch if valid
    try:
        subprocess.run(["git", "apply", "--3way", temp_diff_path], cwd=repo_path, check=True)
        print(f"[SUCCESS] Applied patch for ID {current_id}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to apply patch for ID {current_id}")
        print(f"[DEBUG] {e.stderr}")
    finally:
        if os.path.exists(temp_diff_path):
            os.remove(temp_diff_path)

