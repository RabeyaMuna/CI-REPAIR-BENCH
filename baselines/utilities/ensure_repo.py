import os
import pandas as pd
import json
import subprocess


def run_cmd(args, cwd=None):
    """Run a command and raise with readable stderr on failure."""
    result = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout.strip()

def ensure_repo_at_commit(repo_url, repo_path, commit_sha):
    if not os.path.exists(repo_path):
        run_cmd(["git", "clone", repo_url, repo_path])
    else:
        # Make sure it's a git repository
        if not os.path.exists(os.path.join(repo_path, ".git")):
            raise RuntimeError(f"Directory exists but is not a git repository: {repo_path}")
        # Pull latest changes
        run_cmd(["git", "fetch", "--all"], cwd=repo_path)
    
    # **Reset local changes to make repo clean**
    run_cmd(["git", "reset", "--hard"], cwd=repo_path)
    run_cmd(["git", "clean", "-fd"], cwd=repo_path)  # remove untracked files/directories
    
    # Checkout the desired commit
    run_cmd(["git", "checkout", "--detach", commit_sha], cwd=repo_path)
