import os
import re
import json
import subprocess
import logging
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from utilities.chunking_logic import chunk_log_by_tokens
from utilities.load_config import load_config
from utilities.symbols_outline import get_outline_for_file

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
api_key = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

automated_commands_available = [
  {
    "tool": "ruff",
    "purpose": "Unified linter and formatter for Python (replaces flake8, pyflakes, pycodestyle, etc.)",
    "check_command": "ruff check {{file_or_dir}}",
    "fix_command": "ruff check --fix {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "black",
    "purpose": "Code formatter enforcing uniform style and indentation",
    "check_command": "black --check {{file_or_dir}}",
    "fix_command": "black {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "isort",
    "purpose": "Sort and group imports automatically",
    "check_command": "isort --check-only {{file_or_dir}}",
    "fix_command": "isort {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "flake8",
    "purpose": "Linting tool for style and logical errors (PEP8, unused imports, etc.)",
    "check_command": "flake8 {{file_or_dir}}",
    "fix_command": "ruff check --fix {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "autopep8",
    "purpose": "Automatically formats Python code to be PEP8 compliant",
    "check_command": "autopep8 --diff {{file_or_dir}}",
    "fix_command": "autopep8 --in-place --aggressive {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "yapf",
    "purpose": "Code formatter (alternative to black/autopep8)",
    "check_command": "yapf --diff {{file_or_dir}}",
    "fix_command": "yapf -i {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "pylint",
    "purpose": "Comprehensive linter for detecting code smells and logical issues",
    "check_command": "pylint {{file_or_dir}}",
    "fix_command": "ruff check --fix {{file_or_dir}}",
    "file_pattern": "*.py"
  },
  {
    "tool": "mypy",
    "purpose": "Static type checker for Python",
    "check_command": "mypy {{file_or_dir}}",
    "fix_command": "",
    "file_pattern": "*.py"
  },
  {
    "tool": "pytest",
    "purpose": "Run unit and integration tests",
    "check_command": "pytest {{file_or_dir}}",
    "fix_command": "",
    "file_pattern": "tests/*.py"
  },
  {
    "tool": "bandit",
    "purpose": "Security linter (checks for hardcoded secrets, dangerous calls, etc.)",
    "check_command": "bandit -r {{file_or_dir}}",
    "fix_command": "",
    "file_pattern": "*.py"
  },
  {
    "tool": "codespell",
    "purpose": "Spell checker for comments and docstrings",
    "check_command": "codespell {{file_or_dir}}",
    "fix_command": "codespell -w {{file_or_dir}}",
    "file_pattern": ["*.py", "*.md", "*.rst"]
  },
  {
    "tool": "docformatter",
    "purpose": "Format docstrings to follow PEP 257 conventions",
    "check_command": "docformatter --check {{file_or_dir}}",
    "fix_command": "docformatter --in-place {{file_or_dir}}",
    "file_pattern": "*.py"
  }
]



class DiffOutput(BaseModel):
    diff: str


class PatchGeneration:
    """
    PatchGeneration applies a two-phase CI repair pipeline:
    1️ Try automated tool-based fixes (ruff, black, isort, etc.)
    2️ If unavailable or unsuccessful, generate localized patches via LLM
    3️ Produce a validated unified diff and reset repo state
    """

    def __init__(
        self,
        bug_report: Dict,
        repo_path: str,
        task_id: str,
        error_details: Optional[Dict] = None,
        workflow_path: str = "",
        workflow: str = "",
    ):
        self.config = load_config()
        self.bug_report = bug_report
        self.repo_path = repo_path
        self.task_id = task_id
        self.error_details = error_details or {}
        self.sha_fail = bug_report.get("sha_fail")
        self.workflow_path = workflow_path
        self.workflow = workflow
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        self.parser = JsonOutputParser()
        self.patch_results: List[Dict[str, Any]] = []
        self.original_content = None

    # =========================================================
    # --------------------- CORE METHODS ----------------------
    # =========================================================

    def _call_llm_directly(self, prompt: str) -> Any:
        """Chunk long prompts and call the LLM, parsing valid JSON output."""
        chunks = chunk_log_by_tokens(prompt, max_tokens=50000)
        for chunk in chunks:
            try:
                response = self.llm.invoke([HumanMessage(content=chunk)]).content.strip()
                match = re.search(r"```(?:json)?\s*\n(.*?)```", response, re.DOTALL)
                extracted = match.group(1).strip() if match else response
                return self.parser.parse(extracted)
            except Exception as e:
                logger.error(f"Failed to parse LLM response: {e}")
                logger.debug(f"Raw LLM Output:\n{response}")
        return None
    
    def _extract_outline_block_for_fault(
    self,
    outline: List[Dict[str, Any]],
    file_content: str,
    line_range: List[int]
    ) -> Dict[str, Any]:
        """
        Given an outline and a fault's line_range, find the enclosing block (class/func/const)
        and extract its full code from file_content. Returns a dict with snippet and metadata.
        """
        if not line_range or not isinstance(line_range, (list, tuple)) or len(line_range) != 2:
            return {"original_snippet": "", "block_info": None}

        fault_start, fault_end = line_range
        lines = file_content.splitlines()
        best_match = None
        best_span = float("inf")

        def search_outline(nodes: List[Dict[str, Any]]):
            nonlocal best_match, best_span
            for node in nodes:
                start, end = node["start"], node["end"]
                if start <= fault_start <= end or start <= fault_end <= end:
                    # Candidate node
                    span = end - start
                    if span < best_span:  # prefer smaller enclosing scope
                        best_match = node
                        best_span = span
                # recurse into children
                if node.get("children"):
                    search_outline(node["children"])

        search_outline(outline)

        if best_match:
            start, end = best_match["start"], best_match["end"]
            snippet = "\n".join(lines[start - 1:end])
            return {
                "original_snippet": snippet,
                "block_info": {
                    "name": best_match["name"],
                    "kind": best_match["kind"],
                    "start": start,
                    "end": end,
                },
            }

        # Fallback: not inside any outline block
        snippet = "\n".join(lines[fault_start - 1:fault_end])
        return {
            "original_snippet": snippet,
            "block_info": {"name": "top_level", "kind": "unknown", "start": fault_start, "end": fault_end},
        }


    # =========================================================
    # ---------------- AUTOMATED TOOL FIX ---------------------
    # =========================================================

    def _try_automated_fix(self, faults: List[Dict[str, Any]], full_path: str) -> bool:
        """Try to fix the file automatically using appropriate tools (ruff, black, isort, etc.)"""
        prompt = f"""
You are a **CI Repair Analyst AI**, specialized in automated build and validation repair.

Your goal is to determine whether the following Python file can be automatically fixed using
the tools and validation commands that are actually part of this repository's CI workflow
(e.g., ruff, black, isort, flake8, pylint, mypy, pytest).

---

### CONTEXT INFORMATION

#### 1. FILE INFO
Full Path: {full_path}

#### 2. FAULT CONTEXT (Detected by Fault Localization)
{json.dumps(faults, indent=2)}

#### 3. CI ERROR LOGS
{json.dumps(self.error_details, indent=2)}

#### 4. WORKFLOW VALIDATION COMMANDS
These are the exact validation tools or commands extracted from the project's CI workflow file:
{json.dumps(self.workflow, indent=2)}

#### 5. AUTOMATED FIXATION TOOLS AND COMMAND REFERENCE
Below is a verified reference of all available automated tools and their valid fix/check commands.
You must select only those that align with the CI workflow and detected error type.
{json.dumps(automated_commands_available, indent=2)}

---

### OBJECTIVE
Select the most accurate automated fix strategy that:
- Matches the actual **tools used in the CI workflow**.
- Addresses the specific **errors found in fault localization or CI logs**.
- Uses **correct and compatible CLI syntax** for each tool (from the provided command reference).
- Ensures the commands target the provided file path only.

---

### INSTRUCTIONS
1. Identify which automated tools from the provided reference can fix the detected issues.
2. Provide the **installation commands** for those tools (`pip install <tool>`).
3. Provide the **automated fix commands**, each including the full file path.
4. If multiple tools apply, order them logically (e.g., `isort` → `black` → `ruff` → `flake8`).
5. Only return tools that are explicitly mentioned in the CI workflow or are compatible with it.
6. If no tool can fix the issue automatically, return empty lists.

---

### RESPONSE FORMAT (MUST BE STRICTLY VALID JSON)
{{
  "installation_commands": [
    "pip install <tool1>",
    "pip install <tool2>"
  ],
  "fix_commands": [
    "<command1>",
    "<command2>"
  ],
  "tool_explanation": "Briefly explain which tools were chosen, why they apply to this error type, and how they align with the CI workflow."
}}

---

### EXAMPLES

#### Example 1: Import Sorting Error
Error: "Import block is un-sorted or un-formatted"
Workflow uses: `ruff`, `black`, `isort`
Output:
{{
  "installation_commands": ["pip install ruff black isort"],
  "fix_commands": [
    "isort {full_path}",
    "black {full_path}",
    "ruff check --fix {full_path}"
  ],
  "tool_explanation": "The workflow uses ruff and black. The issue is an unsorted import block, so isort + black + ruff --fix will correct and reformat the imports."
}}

#### Example 2: Line-Length Violation
Error: "line too long (E501)"
Workflow uses: `ruff`, `flake8`
Output:
{{
  "installation_commands": ["pip install ruff"],
  "fix_commands": ["ruff check --fix {full_path}"],
  "tool_explanation": "Ruff is part of the CI workflow and can automatically fix E501 violations via --fix."
}}

#### Example 3: Logical or Type Error
Error: "Function missing a return statement"
Workflow uses: `pytest`, `mypy`
Output:
{{
  "installation_commands": [],
  "fix_commands": [],
  "tool_explanation": "This is a logical or semantic error that cannot be fixed automatically."
}}
""".strip()


        try:
            result = self._call_llm_directly(prompt)
            if not (result and isinstance(result, dict)):
                logger.warning(f"No valid response from LLM for automated fix in {full_path}")
                return False

            install_cmds = result.get("installation_commands", [])
            fix_cmds = result.get("fix_commands", [])
            explanation = result.get("tool_explanation", "")

            logger.info(f"Tool explanation: {explanation}")

            # Step 1: install tools
            for cmd in install_cmds:
                try:
                    cmd_parts = cmd.split()
                    logger.info(f"Installing tool: {cmd}")
                    subprocess.run(
                        cmd_parts, cwd=self.repo_path, capture_output=True, text=True, timeout=120
                    )
                except Exception as e:
                    logger.error(f"Tool installation failed ({cmd}): {e}")

            # Step 2: execute automated fix commands
            success = False
            for cmd in fix_cmds:
                try:
                    cmd_parts = cmd.split()
                    logger.info(f"Running automated fix command: {cmd}")
                    fix_proc = subprocess.run(
                        cmd_parts, cwd=self.repo_path, capture_output=True, text=True, timeout=120
                    )
                    if fix_proc.returncode == 0:
                        logger.info(f"Automated fix succeeded: {cmd}")
                        success = True
                    else:
                        success = False
                        logger.warning(f"Command failed: {cmd}\n{fix_proc.stderr}")
                except Exception as e:
                    logger.error(f"Failed running command {cmd}: {e}")

            return success

        except Exception as e:
            logger.error(f"Automated fix analysis failed: {e}")
            return False

    # =========================================================
    # ------------------- LLM PATCH FIX ------------------------
    # =========================================================

    def _generate_llm_patch(
        self, faults: List[Dict[str, Any]], file_path: str, full_path: str, original_content: str
    ) -> bool:
        """
        Generate snippet-level patches using LLM and apply them to the file.
        Each patch includes 'original_snippet' and 'fixed_snippet'.
        """
        outline = get_outline_for_file(full_path) or "No outline available."

        prompt = f"""
You are a Senior Software Engineer responsible for repairing code faults.
Each fault entry describes a specific issue detected by CI validation tools.

---

### FILE PATH
{full_path}

### FILE OUTLINE
{outline}

### FAULT LOCALIZATION DATA
{json.dumps(faults, indent=2)}

---

### INSTRUCTIONS
- For each issue, identify the **exact original snippet** to fix.
- Return a **list** of patches, each containing:
  - `original_snippet`: exact code to replace
  - `fixed_snippet`: corrected version
  - `explanation`: short summary
- Keep structure and indentation consistent.
- Do NOT return the entire file content.

---

### RESPONSE FORMAT
[
  {{
    "original_snippet": "<exact code block that was modified>",
    "fixed_snippet": "<corrected version of that code block>",
    "explanation": "Brief explanation of what was fixed and why"
  }}
]
""".strip()

        try:
            result = self._call_llm_directly(prompt)
            if not (result and isinstance(result, list)):
                logger.warning(f"No valid snippet patches returned by LLM for {file_path}")
                return False

            updated_content = original_content
            success = False

            for idx, patch in enumerate(result, start=1):
                original_snippet = patch.get("original_snippet", "")
                fixed_snippet = patch.get("fixed_snippet", "")
                explanation = patch.get("explanation", "")

                if not original_snippet.strip() or not fixed_snippet.strip():
                    logger.warning(f"Patch {idx} missing snippet data, skipping.")
                    continue

                replaced_content = self._replace_snippet_in_code(
                    updated_content, original_snippet, fixed_snippet
                )

                if replaced_content == updated_content:
                    logger.warning(f"Patch {idx}: snippet not found in {file_path}, skipping.")
                    continue

                self._write_updated_file(full_path, replaced_content)
                success = True
                updated_content = replaced_content

            return success

        except Exception as e:
            logger.error(f"LLM patch generation failed for {file_path}: {e}")
            return False

    # =========================================================
    # ---------------------- REPLACEMENT ----------------------
    # =========================================================

    def _replace_snippet_in_code(self, code: str, original_snippet: str, fixed_snippet: str) -> str:
        """Replace a snippet of code safely with normalization."""
        if not original_snippet.strip():
            return code

        if original_snippet in code:
            return code.replace(original_snippet, fixed_snippet, 1)

        norm_code = "\n".join(line.rstrip() for line in code.splitlines())
        norm_snippet = "\n".join(line.rstrip() for line in original_snippet.splitlines())
        if norm_snippet in norm_code:
            logger.info("Snippet found via normalized search.")
            return norm_code.replace(norm_snippet, fixed_snippet, 1)

        logger.debug("Snippet not found after normalization.")
        return code

    def _write_updated_file(self, full_path: str, content: str) -> None:
        """Write updated content to disk with newline normalization."""
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                for line in content.splitlines(keepends=True):
                    f.write(line if line.endswith("\n") else line + "\n")
            logger.info(f"Updated file written successfully: {full_path}")
        except Exception as e:
            logger.error(f"Failed to write updated file {full_path}: {e}")

    # =========================================================
    # -------------------- PATCH PROCESS ----------------------
    # =========================================================

    def _fix_code(self) -> List[Dict[str, Any]]:
        """Apply automated + LLM fixes sequentially."""
        faults_data = self.bug_report.get("fault_localization_data", [])
        for faults in faults_data:
            file_path = faults["file_path"]
            full_path = faults["full_file_path"]

            if not os.path.exists(full_path):
                logger.warning(f"Invalid file path: {full_path}")
                continue

            original = self._read_file(full_path)
            if not original:
                logger.warning(f"Could not read file: {full_path}")
                continue

            if not self._try_automated_fix(faults, full_path):
                logger.info(f"Falling back to LLM patch for {file_path}")
                self._generate_llm_patch(faults, file_path, full_path, original)
            
            
            modified_content = self._read_file(full_path)
                
            if modified_content != original:
                self.patch_results.append({
                    "file_path": file_path,
                    "full_file_path": full_path,
                    "original_content": original,
                    "fixed_content": modified_content,
                    "fix_method": "automated_tool"})

        return self.patch_results

    # =========================================================
    # ---------------------- DIFF CREATION --------------------
    # =========================================================

    def _format_diff(self, patch_results) -> Optional[DiffOutput]:
        if not patch_results:
            raise RuntimeError("No valid files after patching.")

        valid_files = []
        for r in patch_results:
            try:
                rel = os.path.relpath(r["full_file_path"], self.repo_path)
                valid_files.append(rel)
            except Exception as e:
                logger.warning(f"Could not compute relative path: {e}")

        subprocess.run(["git", "add"] + valid_files, cwd=self.repo_path, check=True)
        proc = subprocess.run(
            ["git", "diff", "--cached"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        diff = proc.stdout
        if not self._is_diff_valid(diff):
            raise RuntimeError("Diff failed validation (git apply --check).")

        logger.info("Diff successfully validated.")
        return DiffOutput(diff=diff)

    def _is_diff_valid(self, diff_text: str) -> bool:
        temp = os.path.join(self.repo_path, "temp_validation.diff")
        with open(temp, "w", encoding="utf-8") as f:
            f.write(diff_text)
        proc = subprocess.run(
            ["git", "apply", "--check", "--3way", temp],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        os.remove(temp)
        return proc.returncode == 0

    # =========================================================
    # ---------------------- UTILITIES ------------------------
    # =========================================================

    def _read_file(self, full_path: str) -> str:
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Cannot read file {full_path}: {e}")
            return ""

    # =========================================================
    # ------------------ GIT OPERATIONS ------------------------
    # =========================================================

    def _reset_to_failed_commit(self):
        logger.info(f"Resetting repo to commit {self.sha_fail}")
        subprocess.run(["git", "reset", "--hard"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "clean", "-fd"], cwd=self.repo_path, check=True)
        subprocess.run(["git", "checkout", self.sha_fail], cwd=self.repo_path, check=True)

    def _check_current_commit(self) -> bool:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() == self.sha_fail

    # =========================================================
    # ---------------------- MAIN ENTRY -----------------------
    # =========================================================

    def run(self) -> Dict[str, Any]:
        """Main entrypoint for patch generation and diff validation."""
        if not self._check_current_commit():
            self._reset_to_failed_commit()

        patches = self._fix_code()

        try:
            diff_output = self._format_diff(patches)
        except Exception as e:
            logger.error(f"Failed diff creation: {e}")
            diff_output = None

        self._reset_to_failed_commit()

        return {
            "id": self.task_id,
            "sha_fail": self.sha_fail,
            "diff": diff_output.diff if diff_output else "",
        }
