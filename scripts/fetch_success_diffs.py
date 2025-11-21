import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any

import pandas as pd
import requests
from dotenv import load_dotenv

# ========== PATHS & CONSTANTS ==========

DATASET_PATH = (
    "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset/lca_dataset.parquet"
)
OUT_DIR = "/Users/rabeyakhatunmuna/Documents/CI-REPAIR-BENCH/dataset"

PATCHES_OUT_PATH = os.path.join(OUT_DIR, "generated_patches.json")
MISSING_DIFFS_OUT_PATH = os.path.join(OUT_DIR, "missing_success_diffs.json")

REQUIRED_COLS = ["id", "repo_owner", "repo_name", "sha_fail", "sha_success"]

GITHUB_API_URL = "https://api.github.com"

# === EXCLUSION RULES ===
EXCLUDED_EXTS = {
    ".json", ".ts", ".rst", ".js", ".html", ".toml", ".yaml", ".yml",
    ".tsx", ".lock", ".md", ".sh", ".onnx", ".pbtxt", ".uts", ".robot",
    ".css", ".po", ".example", ".ini", ".cfg", ".conf", ".csv", ".xml"
}
EXCLUDED_NAMES = {".gitignore"}
EXCLUDED_DIRS_PREFIX = (".vscode/",)  # extend if needed, e.g. ("docs/", ".vscode/")

# ========== ENV & AUTH ==========

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

if not GITHUB_TOKEN:
    raise SystemExit("ERROR: GITHUB_TOKEN not found in environment (.env).")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3.diff",
}

# ========== LOGGING ==========

def setup_logging() -> None:
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "fetch_success_diffs.log"),
            logging.StreamHandler(),
        ],
    )

# ========== DIFF FILTERING HELPERS ==========

def should_exclude_file(path: str) -> bool:
    """
    Decide if a file path should be excluded based on extension, file name, or directory.
    """
    path = path.strip().lstrip("./")
    lower = path.lower()

    # Check directory prefixes
    for prefix in EXCLUDED_DIRS_PREFIX:
        if lower.startswith(prefix):
            return True

    # Check file name
    name = Path(lower).name
    if name in EXCLUDED_NAMES:
        return True

    # Check extension
    ext = Path(lower).suffix
    if ext in EXCLUDED_EXTS:
        return True

    return False


def filter_diff_blocks(raw_diff: str) -> str:
    """
    Take a full unified diff (as returned by GitHub v3.diff) and remove
    entire file blocks whose paths match the exclusion rules.
    """
    if not raw_diff:
        return ""

    lines = raw_diff.splitlines(keepends=True)
    kept_blocks: List[str] = []
    current_block: List[str] = []
    current_paths = (None, None)  # (a_path, b_path)

    def flush_block():
        nonlocal current_block, current_paths
        if not current_block:
            return

        a_path, b_path = current_paths
        filename = (b_path or a_path or "").strip()
        if filename and should_exclude_file(filename):
            logging.debug("Excluding diff block for %s", filename)
        else:
            kept_blocks.append("".join(current_block))

        current_block = []
        current_paths = (None, None)

    for line in lines:
        if line.startswith("diff --git "):
            # New file block starts; flush previous
            flush_block()

            # Start new block and parse paths
            current_block.append(line)
            parts = line.strip().split()
            # Expected: diff --git a/path b/path
            a_path = parts[2] if len(parts) > 2 else None
            b_path = parts[3] if len(parts) > 3 else None
            if a_path and a_path.startswith("a/"):
                a_path = a_path[2:]
            if b_path and b_path.startswith("b/"):
                b_path = b_path[2:]
            current_paths = (a_path, b_path)
        else:
            current_block.append(line)

    # Flush last block
    flush_block()

    final_diff = "".join(kept_blocks)
    if final_diff and not final_diff.endswith("\n"):
        final_diff += "\n"
    return final_diff

# ========== GITHUB API ==========

def fetch_commit_diff(owner: str, repo: str, sha_success: str) -> Optional[str]:
    """
    Fetch the raw unified diff for a *single* commit (sha_success) from GitHub.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{repo}/commits/{sha_success}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
    except requests.RequestException as e:
        logging.warning(
            "Request error for %s/%s commit %s: %s",
            owner, repo, sha_success, e
        )
        return None

    if resp.status_code != 200:
        logging.warning(
            "GitHub API error for %s/%s commit %s: %s %s",
            owner, repo, sha_success, resp.status_code, resp.text[:200],
        )
        return None

    return resp.text

# ========== DATASET LOADING ==========

def load_dataset() -> pd.DataFrame:
    """
    Load the dataset from the fixed parquet path.
    Expects columns: id, repo_owner, repo_name, sha_fail, sha_success.
    """
    p = Path(DATASET_PATH)
    if not p.exists():
        raise FileNotFoundError(DATASET_PATH)

    logging.info("Loading dataset from %s", DATASET_PATH)
    df = pd.read_parquet(p)

    # Normalize column names (case-insensitive)
    cmap = {c.lower(): c for c in df.columns}
    missing = [c for c in REQUIRED_COLS if c not in cmap]
    if missing:
        raise KeyError(
            f"Required columns missing: {missing}. Available: {list(df.columns)}"
        )

    df = df[
        [
            cmap["id"],
            cmap["repo_owner"],
            cmap["repo_name"],
            cmap["sha_fail"],
            cmap["sha_success"],
        ]
    ].rename(
        columns={
            cmap["id"]: "id",
            cmap["repo_owner"]: "repo_owner",
            cmap["repo_name"]: "repo_name",
            cmap["sha_fail"]: "sha_fail",
            cmap["sha_success"]: "sha_success",
        }
    )
    logging.info("Loaded %d rows", len(df))
    return df

# ========== JSON WRITER ==========

def write_json_array_stream(objs_iter: Iterator[Dict[str, Any]], out_path: str) -> int:
    """
    Stream-safe writer to avoid half-written JSON on crashes.
    Writes to a temp file then renames.
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path + ".tmp"

    written = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("[")
        first = True
        for obj in objs_iter:
            if obj is None:
                continue
            if not first:
                f.write(",")
            f.write("\n")
            f.write(json.dumps(obj, ensure_ascii=False))
            first = False
            written += 1
        f.write("\n]\n")

    Path(tmp_path).replace(out_path)
    return written

# ========== MISSING DIFF COLLECTOR ==========

class MissingDiffCollector:
    """
    Collects entries with no usable diff, then writes them
    as a JSON array to a separate file.
    """

    def __init__(self):
        self.entries: List[Dict[str, Any]] = []
        self.count_by_reason: Counter[str] = Counter()

    def add(
        self,
        reason: str,
        rid: int,
        owner: str,
        repo: str,
        sha_fail: str,
        sha_success: str,
    ) -> None:
        entry = {
            "id": rid,
            "repo": f"{owner}/{repo}",
            "sha_fail": sha_fail,
            "sha_success": sha_success,
            "fail_commit_url": (
                f"https://github.com/{owner}/{repo}/commit/{sha_fail}"
                if sha_fail else None
            ),
            "success_commit_url": (
                f"https://github.com/{owner}/{repo}/commit/{sha_success}"
                if sha_success else None
            ),
            "reason": reason,
        }
        self.entries.append(entry)
        self.count_by_reason[reason] += 1

    def write_json(self, out_path: str) -> int:
        """
        Write all collected missing-diff entries as a JSON array to out_path.
        """
        def _iter():
            for e in self.entries:
                yield e

        return write_json_array_stream(_iter(), out_path)

# ========== MAIN LOGIC ==========

def build_records(df: pd.DataFrame, missing: MissingDiffCollector) -> Iterator[Dict[str, Any]]:
    for _, row in df.iterrows():
        rid = int(row["id"])
        owner = str(row["repo_owner"])
        repo = str(row["repo_name"])
        sha_fail = str(row["sha_fail"])
        sha_success = str(row["sha_success"])

        logging.info(
            "Processing id=%s %s/%s (success commit %s)",
            rid, owner, repo, sha_success[:7] if sha_success else "<none>"
        )

        if not sha_success:
            logging.warning("Missing sha_success for id=%s; skipping", rid)
            missing.add(
                reason="missing_sha_success",
                rid=rid,
                owner=owner,
                repo=repo,
                sha_fail=sha_fail,
                sha_success=sha_success,
            )
            continue

        raw_diff = fetch_commit_diff(owner, repo, sha_success)
        if raw_diff is None:
            logging.warning("No diff fetched for id=%s; skipping", rid)
            missing.add(
                reason="fetch_failed",
                rid=rid,
                owner=owner,
                repo=repo,
                sha_fail=sha_fail,
                sha_success=sha_success,
            )
            continue

        filtered_diff = filter_diff_blocks(raw_diff)
        if not filtered_diff.strip():
            logging.warning("Filtered diff empty for id=%s; skipping", rid)
            missing.add(
                reason="filtered_diff_empty",
                rid=rid,
                owner=owner,
                repo=repo,
                sha_fail=sha_fail,
                sha_success=sha_success,
            )
            continue

        # Only successful ones get yielded as patches
        yield {
            "id": rid,
            "sha_fail": sha_fail,
            "diff": filtered_diff,
        }

def main() -> None:
    setup_logging()
    df = load_dataset()

    missing = MissingDiffCollector()

    logging.info(
        "Fetching success-commit diffs from GitHub and writing patches to %s",
        PATCHES_OUT_PATH,
    )
    written_patches = write_json_array_stream(
        build_records(df, missing),
        PATCHES_OUT_PATH,
    )

    logging.info("Wrote %d patch objects → %s", written_patches, PATCHES_OUT_PATH)

    missing_count = missing.write_json(MISSING_DIFFS_OUT_PATH)
    logging.info(
        "Wrote %d missing-diff entries → %s (by_reason=%s)",
        missing_count,
        MISSING_DIFFS_OUT_PATH,
        dict(missing.count_by_reason),
    )

if __name__ == "__main__":
    main()
