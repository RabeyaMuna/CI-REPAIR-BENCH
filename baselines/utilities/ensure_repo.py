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

def ensure_repo_at_commit(repo_url: str, repo_path: str, sha_fail: str):
    """
    Minimal: clone (shallow), fetch just the target commit (shallow), checkout it.
    """
    parent = os.path.dirname(repo_path)
    os.makedirs(parent, exist_ok=True)

    if not os.path.exists(repo_path):
        # Shallow clone (depth=1)
        run_cmd(["git", "clone", "--depth", "1", repo_url, repo_path])

    # Shallow-fetch exactly the failed commit so it’s available locally
    run_cmd(["git", "fetch", "--depth", "1", "origin", sha_fail], cwd=repo_path)

    # Checkout that commit cleanly (detached HEAD)
    run_cmd(["git", "checkout", "--detach", sha_fail], cwd=repo_path)
    run_cmd(["git", "reset", "--hard", sha_fail], cwd=repo_path)
    run_cmd(["git", "clean", "-fdx"], cwd=repo_path)
