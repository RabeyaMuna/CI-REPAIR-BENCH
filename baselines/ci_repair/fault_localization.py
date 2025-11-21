import os
import json
import subprocess
import sys
import math
import re
import yaml
import time
from pathlib import Path
import json, re, demjson3
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from utilities.load_config import load_config
from utilities.snippet_extractor import extract_snippet_from_line_range, find_line_range
from utilities.chunking_logic import chunk_log_by_tokens
from utilities.symbols_outline import build_outline, format_outline

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

api_key = os.getenv("OPENAI_API_KEY")


class FaultLocalization:
    def __init__(self, sha_fail: str, repo_path: str, error_logs: dict, workflow: str, llm: ChatOpenAI):
        # Unpack OmegaConf + project root if your loader returns a tuple
        cfg_result = load_config()
        if isinstance(cfg_result, tuple) and len(cfg_result) == 2:
            self.config, self.project_root = cfg_result
        else:
            self.config, self.project_root = cfg_result, None

        self.error_logs = error_logs or {}
        # Use correct defaults/types
        self.error_context = self.error_logs.get("error_context", [])         # list
        self.error_types = self.error_logs.get("error_types", [])             # list
        # Handle both singular/plural to be robust
        self.failed_jobs = self.error_logs.get("failed_jobs",
                           self.error_logs.get("failed_job", []))              # list
        # Fix tuple-as-key bug
        self.relevant_files = self.error_logs.get("relevant_files", [])       # list
        self._has_checked_out = False
        self.workflow = workflow
        self.repo_path = repo_path
        self.failed_commit = sha_fail

        # Make sure api_key is defined/imported where this runs
        self.llm = llm

        self.parser = JsonOutputParser()


    def run(self) -> Dict:
        try:
            self._checkout_failed_commit_once()

            print("[Step 3] Running final fault localization...")
            result = self._final_fault_localization()

            return result

        except Exception as e:
            base_dir = os.path.join(self.config["exception_dir"], "interrupted_fault_localization")
            os.makedirs(base_dir, exist_ok=True)
            filepath = os.path.join(base_dir, f"{self.failed_commit}_bug.json")
            error_info = {
                "sha_fail": self.failed_commit,
                "error": str(e),
                "tool": "FaultLocalization"
            }
            with open(filepath, "w") as f:
                json.dump(error_info, f, indent=4)
            return error_info
    
          
    def _checkout_failed_commit_once(self):
        try:
            subprocess.run(["git", "checkout", self.failed_commit], cwd=self.repo_path, check=True, capture_output=True)
            self._has_checked_out = True
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git checkout failed: {e.stderr.decode()}")
        
    def _final_fault_localization(self) -> List[Dict]:
        print("[Tool] read_error_file called")

        fault_localization = []
        for item in self.relevant_files:
            
            file_path = (item.get("file") or item.get("path") or "").strip()
            if not file_path:
                continue
            
            ext = Path(file_path).suffix.lower()
            
            if ext not in {".py", ".toml", ".txt"}:
                continue
            
            resolved = self.find_full_file_path(file_path)
            if resolved["status"] != "found":
                continue

            full_file_path = resolved["full_path"]
            
            content = self._read_file_content(full_file_path)
            if not content:
                continue
            
            file_type = self.detect_file_type(file_path)
            outline = build_outline(content) if file_type == "python" else ""
            numbered_full_content = self._numbered_file_content(content)
            chunks = self._chunk_file(numbered_full_content)
            num_chunks = len(chunks)

            # Per-chunk strict FL
            all_faults: List[Dict[str, Any]] = []
            
            for idx, ch in enumerate(chunks):
                file_summary = (
                    f"File: {file_path}\n"
                    f"Full Path: {full_file_path}\n\n"
                    f"**File Outline (symbol → [start–end] lines)**\n{outline}\n\n"
                    f"File Type: {file_type}\n"
                    f"Chunk index: {idx+1} of {num_chunks}\n"
                    f"Chunk lines ({ch['valid_start']}–{ch['valid_end']}):\n\n"
                    f"```{file_type}\n{ch['content']}\n```"
                )
                
                faults = self.fault_localization_based_on_ci_log(
                    file_path=file_path,
                    full_file_path=full_file_path,
                    file_summary=file_summary,
                    valid_start=ch["valid_start"],
                    valid_end=ch["valid_end"],
                    original_content=content,  # unnumbered
                    chunk_idx=idx,
                    num_chunks=num_chunks,
                    faults=all_faults,
                )
                
                if faults:
                    all_faults.extend(faults)
                    
            fault_localization.append({
                "file_path": file_path,
                "full_file_path": full_file_path,
                "faults": all_faults
            })
            
        results = {
            "sha_fail": self.failed_commit,
            "fault_localization_data": fault_localization
        }
        
        return results

    def fault_localization_based_on_ci_log(
    self,
    *,
    file_path: str,
    full_file_path: str,
    file_summary: str,
    valid_start: int,
    valid_end: int,
    original_content: str,
    chunk_idx: int,
    num_chunks: int,
    faults: list,
) -> list:
        """
        Runs the strict FL prompt for a single chunk and returns a list of fault objects.
        Expects the LLM to return [] or [ { ... }, ... ] per your original schema.
        """

        fault_locations = []
        
        prompt = f"""
You are a **Strict Fault Localization Agent**.

Your task:
Given a numbered source-code CHUNK, the CI error context, workflow jobs, and a file outline, identify **all** distinct faults that explain the CI failure — not just the first one. 
Use the outline to expand detected faults to full method/class/import/file scopes where appropriate. 
Output must be **valid JSON only** (array or empty array). No markdown, no commentary, no extra text.

==============================================================================
INPUT
------------------------------------------------------------------------------
SOURCE CODE (numbered lines; 1-based) with Outline of the file is given:
{file_summary}

Each outline entry contains:
  - name: symbol or construct name (function/class/import)
  - type: one of "method" | "class" | "import_block"
  - start_line: first numbered line of the element
  - end_line: last numbered line of the element
Use this outline to determine which scope a fault belongs to and to expand line ranges correctly.

FAILED JOBS:
{self.failed_jobs}

ERROR TYPES:
{self.error_types}

ERROR CONTEXT:
{self.error_context}

WORKFLOW (high-level summary):
{self.workflow}

ALREADY DETECTED FAULTS (skip/avoid duplicates):
{faults}

CHUNK WINDOW: lines {valid_start}–{valid_end}

==============================================================================
RULES
------------------------------------------------------------------------------
R1. Detect All Matches
  - Read every numbered line between {valid_start}–{valid_end}.
  - Add every new fault that directly explains CI messages or rule codes (ruff, pylint, mypy, pytest, etc.).
  - Include formatting, linting, typing, runtime, and test failures if indicated by the logs.

R2. Verify in Code
  - Each fault must be observable in these lines or provably absent (for missing imports/symbols).
  - Confirm CI log claims (missing symbol, unused import, annotation absence, etc.) against code shown.

R3. Outline-Based Scope Expansion
  - For each detected fault, check the file outline to identify which structural element it belongs to.
  - Expand the line_range to cover the full element boundaries:
      • If inside a method: return the full method range.
      • If inside a class: return the full class range.
      • If within an import block: return all contiguous import lines + 2 lines after.
      • If spanning multiple non-overlapping outline elements, escalate to "file".
  - If faults occur in the same outline element, merge them into one JSON object and combine reasons.

R4. Reason–Snippet Consistency
  - Each "reason" must cite concrete CI evidence (messages or rule codes such as F401, E1101, I001, etc.).
  - Reference actual code or confirm omission explicitly. Avoid vague speculation.

R5. Line Range Integrity
  - "line_range" must match **exact first and last lines** of the chosen scope according to the outline.
  - Always use the displayed (numbered) line indices, not inferred offsets.

R6. Fault Type & Level
  - Choose "issue_type" precisely (formatting, linting, type_error, runtime_error, test_failure, dependency_error, docstring, complexity, other).
  - Set "fault_localization_level" to reflect the expanded scope: line | method | class | import_block | file.

R7. Extended Reason Context
  - In "reason", you may mention related decorators, helper calls, or affected functions that clarify the cause.
  - Do NOT expand snippet scope beyond the chosen element — just mention those links textually.

R8. Missing Elements
  - If CI cites a missing construct (e.g., import, symbol, type hint), confirm absence and record it with the correct scope, usually "import_block" or "method".

R9. Output Contract (Hard)
  - Return **strict JSON only**:
      • Either [] or a JSON array of objects matching the schema below.
      • No markdown fences, comments, or trailing commas.
  - Do NOT include "code_snippet" — it will be added later by the caller.

==============================================================================
OUTPUT SCHEMA (JSON array)
------------------------------------------------------------------------------
[
  {{
    "file_path": "{file_path}",
    "full_file_path": "{full_file_path}",
    "line_range": [start_line, end_line],
    "reason": "Comprehensive explanation citing CI log messages and rule codes. If merged, include concise bullet-like sub-fault summaries.",
    "issue_type": "formatting | linting | type_error | runtime_error | test_failure | dependency_error | docstring | complexity | other",
    "fault_localization_level": "line | method | class | import_block | file"
  }},
  ...
]

==============================================================================
CHECKLIST BEFORE RETURNING
------------------------------------------------------------------------------
1) All new faults in {valid_start}–{valid_end} are included.
2) Overlapping or nested faults are merged under a common expanded scope.
3) "line_range" matches the exact outline boundaries for the chosen scope.
4) Each "reason" references concrete CI evidence or rule code.
5) No duplicates of ALREADY DETECTED FAULTS.
6) Output is valid JSON only — no markdown, prose, or trailing commas.
7) If nothing new is found, return [].
"""
        print(f"[Chunk {chunk_idx+1}/{num_chunks}] Analyzing lines {valid_start}-{valid_end}...")

        try:
            raw_response = self.llm.invoke(prompt).content.strip()
            # Clean any markdown fences
            if raw_response.startswith("```"):
                raw_response = raw_response.strip("` \n")

            # Must be pure JSON
            try:
        # Try strict JSON parsing
                parsed_result = json.loads(raw_response)
            except json.JSONDecodeError:
                # Fallback to demjson3 for messy JSON
                parsed_result = demjson3.decode(raw_response)

            if not isinstance(parsed_result, list) or not parsed_result:
                print(f"[Chunk {chunk_idx}] No faults found.")
                return []

            # --------------------------------------------------------
            # Fetch local snippets using the expanded line ranges
            # -------------------------------------------------------
            for fault in parsed_result:
                line_range = fault.get("line_range")
                
                if not line_range:
                    continue  # Skip if no snippet present

                start, end = line_range
                if valid_start <= start and end <= valid_end:
                    # Normal case: attach found range
                    snippet = extract_snippet_from_line_range(original_file_content=original_content, line_range=line_range)
                    
                    fault["code_snippet"] = snippet
                    
                    fault_locations.append(fault)
                    print("\n--- Fault Detected ---")
                    print("Code Snippet:\n", snippet)
                    print("Line Range:", line_range)
                    print("---------------------\n")
                else:
                    # Skip fault outside this chunk
                    print(f"[Chunk {chunk_idx+1}] Skipping fault outside chunk range {valid_start}-{valid_end}: {line_range}")


        except json.JSONDecodeError as e:
            # Save parsing error for analysis
            self._save_fault_localization_error(
                self.failed_commit, e,
                tool_name="fault_localization_based_on_ci_log",
                prompt_name="json_parsing"
            )
            print(f"[Chunk {chunk_idx}] JSON parse error: {e}")
            print(f"[Chunk {chunk_idx}] Raw response:\n{raw_response}")
            return []

        except Exception as e:
            # Save unexpected errors
            self._save_fault_localization_error(
                self.failed_commit, e,
                tool_name="fault_localization_based_on_ci_log",
                prompt_name="chunk_processing"
            )
            print(f"[Chunk {chunk_idx}] Error processing chunk: {e}")
            return []

        return fault_locations
                
    def find_full_file_path(self, file_path: str) -> dict:
        """
        Find the best matching full path for a given relative file_path inside repo_path.
        Prioritizes the candidate whose relative path best matches the requested file_path.
        """
        normalized = os.path.normpath(file_path.split(":", 1)[0].strip())
        abs_path = os.path.join(self.repo_path, normalized)
        file_name = os.path.basename(normalized)

        try:
            # Direct full path exists
            if os.path.exists(abs_path):
                return {"status": "found", "full_path": abs_path}

            candidates = []
            # Walk the repo and collect all candidates
            for root, _, files in os.walk(self.repo_path):
                if file_name in files:
                    candidate = os.path.join(root, file_name)
                    rel_candidate = os.path.relpath(candidate, self.repo_path)
                    # Compute similarity: longest common suffix with the requested path
                    score = len(os.path.commonprefix([rel_candidate[::-1], normalized[::-1]]))
                    candidates.append((score, candidate))

            if candidates:
                # Pick the highest scoring match
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_match = candidates[0][1]
                return {"status": "found", "full_path": best_match, "all_candidates": [c[1] for c in candidates]}

        except Exception as e:
            return {"status": "error", "error": str(e)}

        return {"status": "not_found"}

    def _read_file_content(self, resolved_path: str) -> str:
        """Return file content as a string, or '' if file is missing/unreadable."""
        
        if not os.path.exists(resolved_path):
            print(f"[WARN] File not found: {resolved_path}")
            return ""
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[WARN] Could not read {resolved_path}: {e}")
            return ""

    def _call_llm_directly(self, prompt: str) -> dict:
        chunks = chunk_log_by_tokens(prompt, max_tokens=90000)

        for chunk in chunks:
            try:
                raw_response = self.llm.invoke([HumanMessage(content=chunk)]).content.strip()

                # Remove markdown fences if present
                raw_response = re.sub(r"^```(?:json)?\s*|```$", "", raw_response.strip(), flags=re.DOTALL)

                return self.safe_parse_json(raw_response)
            except Exception as e:
                print(f"[ERROR] LLM call failed while parsing: {e}")
                raise

    def safe_parse_json(self, text: str) -> Dict:
        try:
            return self.parser.parse(text)
        except Exception as e:
            print("[!] Primary parser failed. Attempting cleanup...")
            try:
                cleaned = text.strip().strip("```json").strip("```").strip()
                return json.loads(cleaned)
            except Exception as second_error:
                print("[!] Fallback parsing failed.")
                raise ValueError(f"JSON parsing failed: {e}\n\nRaw:\n{text}")

    def _numbered_file_content(self, content: str, offset: int = 0) -> str:
        return "\n".join(f"{idx+1:04d}: {line}" for idx, line in enumerate(content.splitlines()))

    def detect_file_type(self, file_path: str) -> str:
        """Detect programming/config language based on file extension."""
        ext = Path(file_path).suffix.lower()
        mapping = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".sh": "bash",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".dockerfile": "dockerfile",
            ".rst": "restructuredtext",
            ".md": "markdown",
        }
        
        # Handle Dockerfile without extension
        if ext == "" and Path(file_path).name.lower() == "dockerfile":
            return "dockerfile"
        return mapping.get(ext, "text")  # default to plain text if unknown

    # Helper method for all other file types
    def _chunk_file(self, file_content):
        chunk_size = 400
        overlap = 50
        lines = file_content.splitlines()
        total_lines = len(lines)
        chunks = []
        
        num_chunks = math.ceil(total_lines / (chunk_size - overlap)) if chunk_size > overlap else 1

        for chunk_idx in range(num_chunks):
            start_idx = max(0, chunk_idx * chunk_size - overlap)
            end_idx = min(start_idx + chunk_size + overlap, total_lines)
            
            valid_start = start_idx + overlap if chunk_idx != 0 else start_idx
            valid_end = end_idx - overlap if chunk_idx != num_chunks - 1 else end_idx

            chunk_lines = lines[start_idx:end_idx]
            
            chunks.append({
                "content": "\n".join(chunk_lines),
                "line_range": (start_idx + 1, end_idx),
                "valid_start": valid_start + 1,
                "valid_end": valid_end
            })

        return chunks
    
    def _save_fault_localization_error(
    self,
    sha: str,
    error: Exception,
    tool_name: str = "",
    prompt_name: str = "",
    extra_context: dict | None = None
    ):
        """
        Save detailed fault localization error info to JSON file.

        Args:
            sha (str): Commit SHA where the error occurred.
            error (Exception): The raised exception.
            tool_name (str): The internal tool/method where failure occurred.
            prompt_name (str): The specific prompt/stage (e.g., 'json_parsing', 'line_range').
            extra_context (dict): Optional contextual info (chunk_idx, file_path, etc.)
        """

        base_dir = os.path.join(self.config["exception_dir"], "interrupted_fault_localization")
        os.makedirs(base_dir, exist_ok=True)

        # timestamped filename to avoid overwriting
        fname = f"{self.failed_commit}.json"
        filepath = os.path.join(base_dir, fname)

        error_info = {
            "sha_fail": sha or self.failed_commit,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "tool": tool_name,
            "prompt_name": prompt_name,
            "Agent": "FaultLocalization",
        }

        # attach any extra context (e.g., chunk_idx, file_path)
        if extra_context:
            error_info["context"] = extra_context

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=4)

        return error_info
    