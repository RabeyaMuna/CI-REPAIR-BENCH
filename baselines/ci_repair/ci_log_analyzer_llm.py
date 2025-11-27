from typing import List, Dict, Any
import json
import os
import re
import time
import demjson3
import tiktoken
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from utilities.constant import ERROR_KEYWORDS
from utilities.load_config import load_config
from utilities.chunking_logic import chunk_log_by_tokens

load_dotenv()

MAX_TOKENS_SUMMARY = 90_000

class CILogAnalyzerLLM:
    def __init__(
        self,
        repo_path: str,
        ci_log: List[Dict[str, Any]],
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: ChatOpenAI,
         model_name: str
    ):
        self.config = load_config()
        self.repo_path = repo_path
        self.ci_log = ci_log
        self.sha_fail = sha_fail
        self.workflow = workflow
        self.workflow_path = workflow_path

        self.llm = llm
        self.model_name = model_name
        self._encoder = self._get_encoder()

        self.error_details: List[Dict[str, Any]] = []

    def ci_log_analysis(self) -> List[Dict[str, Any]]:
        """
        Analyze CI logs using LLM
        """
        print("Running Tool: LL M-based CI Log Analysis")
        results: List[Dict[str, Any]] = []
        THRESHOLD = 80_000
        chunk_tracker = []
        for step in self.ci_log:
            step_name = step.get("step_name", "unknown_step")
            log = step.get("log", "")
            print(f"\nProcessing Step: {step_name}")

            try:
                log_text = log if isinstance(log, str) else "\n".join(log)
                enc = tiktoken.encoding_for_model(self.model_name)
                total_tokens = len(enc.encode(log_text))

                print(f"Token count for '{step_name}': {total_tokens}")

                if total_tokens > THRESHOLD:
                    raw_chunks = chunk_log_by_tokens(log_text, max_tokens=60000, overlap=200)
                    print(f"Chunking activated: {len(raw_chunks)} chunks created for step '{step_name}'")
                    
                    chunk_tracker.append((step_name, len(raw_chunks)))
                    
                    if len(raw_chunks) > 10:
                        chunks = self._filter_chunks(raw_chunks)
                    else:
                        chunks = raw_chunks
                else:
                    chunks = [log_text]
                    print(f"No chunking needed for '{step_name}'")
                    
                self._save_chunk_tracker(chunk_tracker)

                step_chunks = []

                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i + 1}/{len(chunks)}...")

                    prompt = f"""
Analyze the following CI log chunk and extract comprehensive information with a focus on capturing all error context.

## INSTRUCTIONS:
1. Create an extremely detailed natural language summary that includes EVERY important piece of information from the log chunk:
   - Mention all commands, operations, and outcomes.
   - Mention all test results.
   - Mention files involved in errors, failing tests, or critical warnings (you may mention other files only briefly in the narrative).
   - Mention all errors and critical warnings.

2. Extract ONLY file paths that are directly related to failures, assertions, runtime errors, or critical warnings.
   - A file is related if it appears in:
     * a stack trace,
     * a failing test message,
     * an error or exception message,
     * or a critical warning that may cause the CI run to fail.
   - Ignore files that are only mentioned in generic setup or installation steps (e.g., installing requirements, caching).
3. Identify ALL failures, errors, and issues with their complete context
4. Extract the exact error blocks with their complete surrounding context

## OUTPUT FORMAT:
Return ONLY valid JSON with this exact structure:

{{
  "step_name": "{step_name}",
  "summary": "Extremely natural language description that essentially rewrites the entire log chunk in organized narrative form. Include ALL operations, files mentioned in the chunk, and ALL contextual information.",
  "relevant_files": [
    {{
      "file": "normalized/path/to/file1.py",
      "reason": "Exact CI log evidence why this file is mentioned in the CI Log and if there is any relevance of CI failure if so why",
    }},
    {{
      "file": "normalized/path/to/file2.py",
      "reason":  "Exact CI log evidence why this file is mentioned in the CI Log and if there is any relevance of CI failure if so why",
    }}
   ],
    "relevant_failures": [
        "Complete error block 1 from CI log with all context lines",
        "Complete error block 2 from CI log with all context lines"
    ]
}}

## CRITICAL RULES:
- The summary MUST be exhaustive and include ALL information from the log chunk
- The `"reason"` for each file MUST be derived from the log itself (quote log lines or write a precise explanation). NEVER return generic placeholders like "mentioned in log".
- Normalize file paths: convert absolute paths (e.g., `/opt/.../optuna/study/_optimize.py`) to repo-relative (`optuna/study/_optimize.py`).
- Deduplicate files but preserve unique reasons if a file appears multiple times with different contexts.
- Preserve exact wording and formatting of error/warning lines inside `relevant_failures`.
- **Do NOT wrap the JSON in ```json or ``` markers. Output plain JSON only.**

## CI LOG CHUNK:
{chunk}

## CRITICAL RULES:
- Return ONLY raw JSON in the above format — absolutely no text before or after.
- Do NOT include ```json or ``` markers.
- Do NOT add any extra keys, commentary, or filler text.
- Fill in every field based only on the provided log chunk.
- If no files or CI failures reasons exist, return empty arrays [] for those fields.
"""

                    response = self.llm.invoke([HumanMessage(content=prompt)])
                    time.sleep(1.0)  # throttle
                    content = response.content.strip()
                    
                    try:
                    # Try standard JSON decoding
                        cleaned_json = json.loads(content)
                    except json.JSONDecodeError:
                    # Fallback: tolerant decoder
                        cleaned_json = demjson3.decode(content)

                    step_chunks.append(cleaned_json)
                
                results.append({
                    "step_name": step_name,
                    "chunks": step_chunks
                })

            except Exception as e:
                print(f"[ERROR] Processing step '{step_name}': {str(e)}")
                results.append({
                    "step_name": step_name,
                    "chunks": [],
                    "error": str(e)
                })

        return results
   
    # ------------------------------------------------------------------
    def generate_log_summary(self, log_details) -> Dict[str, Any]:
        """
        Generate a structured final error summary from error details,
        workflow tools, and validation checks.
        """
        print(" Running Tool: _generate_summary")
        log_details = []
        for step in self.all_step_outputs:
            step_name = step.get("step_name", "UNKNOWN_STEP")
            chunks = step.get("chunks", [])
            
            step_payload = {
                "step_name": step_name,
                "chunks": chunks,
            }
            step_payload_json = json.dumps(step_payload, indent=2, ensure_ascii=False)
            prompt = f"""
You are a CI log analyzer.

You receive pre-processed CI log information for a FAILED CI RUN, for a SINGLE CI STEP.

## Input Details
--- PRE-PROCESSED Log information for Given STEP DATA (JSON) ---
{step_payload_json}

Using ONLY this information for THIS STEP (all its chunks), produce a structured summary
of the CI failure for this step using the following STRICT JSON schema
(**do not add or remove top-level keys**)

{{
"step_name": {step_name},
"sha_fail": "{self.sha_fail}",
"log_content": "<Explain overall details of the CI log in natural language>",
"error_context": [
    "English explanation(s) of the root cause(s) visible of CI workflow failed supported by log evidence. Provide detail and evident reasons of the given step failure."
],
"relevant_files": [
    {{
    "file": "path/to/file.py",
    "line_number": 123,
    "reason": "Short explanation of why this file is tied to the failure."
    }}
],
"error_types": [
    {{
    "category": "High-level category, e.g. 'Test Failure', 'Runtime Error', 'Dependency Error', 'Configuration Error', 'Code Formatting', 'Type Checking'",
    "subcategory": "More specific description, e.g. 'Runtime Error – ValueError in Optuna objective', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
    "evidence": 'Short quote or paraphrase from THIS CHUNK that justifies this classification.'
    }}
]
}}

### Rules (IMPORTANT)

- "step_name": the CI step name (use the input step_name).
- "sha_fail": the failing commit SHA (given).
- "log_content":
  - A concise but informative natural-language description of what happened in this CI STEP.
  - You may integrate information across ALL chunks for this step.

- "error_context":
  - A list of English explanations of the root cause(s) of the CI failure related to THIS STEP.
  - Each entry must be supported by evidence in the summaries and/or relevant_failures.
  - If nothing meaningful appears, use an empty list [].

- "relevant_files":
  - Consider all chunk-level data ("relevant_files", "relevant_failures", and summaries),
    but INCLUDE a file ONLY if:
      * it is clearly linked to a failing test, assertion error, runtime exception,
        dependency error, configuration error, or critical warning in THIS STEP, OR
      * the log explicitly states that the failure occurs in that file.
  - It is OK to discard files that appear in chunk-level "relevant_files" if they were only
    mentioned in setup/installation and are not clearly tied to the failure.
  - Deduplicate by "file" path. If the same file appears with different reasons, merge
    them into one concise, evidence-based "reason".
  - "line_number":
      * Use the failing line number if it is clearly shown in the logs,
      * otherwise null.
  - If no file clearly meets these conditions, return "relevant_files": [].
  - provide evidence-based "reasoning" explaining how this file is tied to the CI failure.

- "error_types":
  - Describe the kinds of errors visible in this STEP:
      * Test failures,
      * Runtime exceptions,
      * Dependency / environment issues,
      * Configuration / CI YAML problems,
      * Code formatting or linting issues,
      * Type checking errors, etc.
  - Each entry MUST include:
      * "category"  (broad bucket),
      * "subcategory" (more specific),
      * "evidence"   (short quote or paraphrase from summaries / relevant_failures).

### Global Rules

1. Use ONLY the information from the STEP JSON shown below.
2. Use null for any unknown scalar values (e.g., line_number if not visible).
3. Do NOT add extra top-level keys.
4. Return STRICT JSON ONLY — no markdown, no comments, no natural language outside JSON.
"""

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)]).content
                try:
                    summary = json.loads(response)
                except json.JSONDecodeError:
                    summary = demjson3.decode(response)
                    
                log_details.append(summary)
            except Exception as e:
                # If one chunk fails, log and continue
                self._log_error(
                    method="generate_log_summary",
                    error=e,
                    step=f"{step_name}",
                )

        return log_details

    # ------------------------------------------------------------------
    def full_content_summary(
        self, log_details: List[Dict[str, Any]], workflow_details: Any
    ) -> Dict[str, Any]:
        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read step-level CI job log summaries and workflow details, then produce a **single, structured, evidence-based JSON summary** that explains why the CI run failed and clearly classifies the errors by category and subcategory.

---

## INPUTS

1. CI Log Details (from step analysis, list of objects):

Each element in `log_details` corresponds to ONE CI step and typically includes:
- "step_name": name of the CI step.
- "sha_fail": the failing commit SHA.
- "log_content": natural-language description of what happened in this step.
- "error_context": list of step-level explanations of root causes in this step.
- "relevant_files": list of files tied to the failure in this step, each with:
  - "file"
  - "line_number" (may be null)
  - "reason"
- "error_types": list of error classifications for this step, each with:
  - "category"
  - "subcategory"
  - "evidence"

Full step-level details:
{json.dumps(log_details, indent=2, ensure_ascii=False)}

2. Workflow Details:

Parsed CI workflow (e.g., GitHub Actions YAML) including jobs, steps, and commands.
Use this to map failing steps to their jobs and commands when possible.

Full workflow details:
{json.dumps(workflow_details, indent=2, ensure_ascii=False)}

---

## OUTPUT FORMAT (strict JSON only)

Return a SINGLE aggregated summary for the entire failed run using this exact structure:

{{
  "sha_fail": "{self.sha_fail}",
  "error_context": [
    "Plain-English explanation(s) of the root cause(s), supported by log evidence. Mention all the steps involved in the failure and how and why it failed."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "reason": "Short evidence-based explanation of why this file is tied to the failure."
    }}
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Code Formatting', 'Dependency Error', 'Test Failure', 'Runtime Error', 'Type Checking', 'Configuration Error'",
      "subcategory": "More specific type under that category, e.g. 'Unused Import', 'Line Length Exceeded', 'ImportError: No module named X', 'AssertionError', 'Missing dependency', 'Mypy type mismatch'",
      "evidence": "Brief quote or paraphrase from logs that proves this classification."
    }}
  ],
  "failed_job": [
    {{
      "job": "Job name or ID",
      "step": "Step name that failed",
      "command": "Exact command or action that caused the failure"
    }}
  ]
}}

---

## INSTRUCTIONS

1. **Aggregate across ALL steps (do not treat them independently).**
   - Read every element in `log_details`.
   - Combine their information into ONE global summary for the entire CI run.

2. **Error Context (global explanation).**
   - Use English sentences summarizing the main root cause(s) of the failure.
   - Explicitly reference the step names involved and has failed in that steps, e.g.:
     - "In step 'Install dependencies', pip failed due to missing package X..."
     - "In step 'Run tests', pytest reported failing tests because..."
   - If multiple steps contribute to the failure, mention each clearly.
   - Base your explanations on `log_content`, `error_context`, and `error_types` from the steps.

3. **Relevant Files (deduplicated across steps).**
   - Consider all `relevant_files` from all steps.
   - Deduplicate by `"file"` path: each file should appear at most once in the final list.
   - If the same file appears in multiple steps, merge the reasons into a single concise,
     evidence-based `"reason"` that reflects all relevant contexts.
   - Use `"line_number":` the most specific failing line if available; otherwise `null`.
   - Include only files that are clearly tied to failures, errors, or critical warnings.
   - If no such file exists, return `"relevant_files": []`.

4. **Error Types (all distinct types found).**
   - Look at all `error_types` from all steps.
   - Aggregate them into a list of distinct (category, subcategory) pairs.
   - For each distinct pair, keep one entry with:
     - `"category"` and `"subcategory"` exactly once.
     - `"evidence"` summarizing or quoting representative log evidence (you may combine evidence from multiple steps briefly).

5. **Failed Job (use workflow + step info).**
   - Use both `log_details` and `workflow_details` to identify which job(s) and step(s) failed.
   - Match step names from `log_details` to steps in the workflow (by their `"name"` field) to infer:
     - `"job"`: the job display name or ID (from the workflow, if available; otherwise null).
     - `"step"`: the failing step name from `log_details`.
     - `"command"`:
       - The value of `"run"` if present, or `"uses"` if it is an action reference.
       - If no command can be found, use `null`.
   - If multiple jobs/steps clearly fail, include multiple entries in `"failed_job"`.

6. **Output Rules**
   - Return **only valid JSON** — no markdown, commentary, or code fences.
   - Do not hallucinate; use `null` for unknown values (e.g., line_number, job, command).
   - Merge duplicates carefully (same file paths, same category/subcategory pairs).
   - Ensure every item is concise, evidence-based, and traceable to the logs and/or workflow.
"""
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            try:
                summary = json.loads(response)
            except json.JSONDecodeError:
                summary = demjson3.decode(response)

            print(" Completed: _generate_summary")
            return summary

        except Exception as e:
            error_dir = os.path.join(
                self.config["exception_dir"], "interrupted_error_log"
            )
            os.makedirs(error_dir, exist_ok=True)

            error_data = {
                "sha_fail": self.sha_fail,
                "error": str(e),
                "tool": "ErrorContextExtractionAgent.run",
            }

            error_file = os.path.join(error_dir, f"{self.sha_fail}_error.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=4)

            return {"error": f"Failed to generate summary: {e}"}

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        print(f"Fully Autonomous Execution for Commit: {self.sha_fail}")
        log_details = self.ci_log_analysis()
        generated_summary = self.full_content_summary(log_details, workflow_details=self.workflow)
        return generated_summary

    # ------------------------------------------------------------------
    def _get_encoder(self):
        """Safely get a tiktoken encoder for the model."""
        try:
            return tiktoken.encoding_for_model("gpt-4o-mini")
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for a given text using the cached encoder."""
        if text is None:
            return 0
        return len(self._encoder.encode(text))

    def _log_error(self, method: str, error: Exception, step: str = ""):
        base_dir = os.path.join(
            self.config["exception_dir"], "interrupted_error_log"
        )
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"{self.sha_fail}_{method}_{int(time.time())}_error.json"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "commit": self.sha_fail,
                    "method": method,
                    "step": step,
                    "error": str(error),
                },
                f,
                indent=2,
            )
        print(f"[ERROR LOGGED] {file_path}")

    def _filter_chunks(self, raw_chunks: List[str]) -> List[str]:
        """
        Keep:
        - All chunks in the first (n-6) that contain any ERROR_KEYWORDS (by word)
        - Always keep the last 6 chunks (serial order preserved)
        """
        n_chunks = len(raw_chunks)

        # If we have 6 or fewer chunks, just keep everything
        if n_chunks <= 6:
            print(f"Filtered from {n_chunks} ➝ {n_chunks} chunks (<= 6, kept all)")
            return raw_chunks

        cutoff = n_chunks - 6
        filtered_chunks: List[str] = []

        # 1) Check the first (n-6) chunks and keep only those with error keywords
        for idx, chunk in enumerate(raw_chunks[:cutoff]):
            for line_no, line in enumerate(chunk.splitlines(), start=1):
                hits = self.is_line_error(line, ERROR_KEYWORDS)
                if hits:
                    # optional debug:
                    # print(f"[FILTER] chunk {idx}, line {line_no}: hits={hits}")
                    filtered_chunks.append(chunk)
                    break  # done with this chunk

        # 2) Append the last 6 chunks unconditionally (preserve serial order)
        filtered_chunks.extend(raw_chunks[cutoff:])

        print(
            f"Filtered from {n_chunks} ➝ {len(filtered_chunks)} chunks "
            f"(checked first {cutoff}, always kept last 6)"
        )

        return filtered_chunks
      
    def _save_chunk_tracker(self, chunk_tracker: List[tuple]):
        debug_dir = os.path.join(self.config["out_folder"], "chunk_tracking")
        os.makedirs(debug_dir, exist_ok=True)

        file_path = os.path.join(debug_dir, "chunk_tracker.json")

        # Load existing data if file exists
        existing: List[Dict[str, Any]] = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = []
            except Exception:
                existing = []

        # Append this run
        existing.append({"sha_fail": self.sha_fail, "chunks": chunk_tracker})

        # Write back pretty JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)

        print(f"[CHUNK TRACKER SAVED] {file_path}")
        
        
    def is_line_error(self, line: str, keywords) -> list[str]:
        """
        Return list of keywords that appear as separate words or exact phrases.
        - Single-word keywords: must match a whole word in the line (case-sensitive).
        - Multi-word keywords: matched as exact substrings (case-sensitive).
        - Special rule: if we see 'error' or 'errors' but the previous word is 'no'
        (e.g. 'no error', 'No errors'), we IGNORE that match.
        """
        hits = []

        # Very simple tokenization: split on whitespace, strip basic punctuation
        raw_tokens = line.split()
        tokens = [tok.strip("[]():,") for tok in raw_tokens]

        # 1) Phrase keywords → use substring
        for kw in keywords:
            if " " in kw:
                if kw in line:  # exact phrase, case-sensitive
                    hits.append(kw)

        # 2) Single-word keywords → must match full token
        # We also apply the "no error" rule here.
        for idx, tok in enumerate(tokens):
            for kw in keywords:
                if " " in kw:
                    continue  # phrases already handled

                if tok == kw:
                    # Special: ignore "error"/"errors" if previous token is 'no' (any case)
                    if kw in ("error", "errors") and idx > 0:
                        prev_tok = tokens[idx - 1]
                        if prev_tok.lower() == "no":
                            # e.g. "no error", "No errors" → not a real error
                            continue

                    hits.append(kw)

        # Remove duplicates while preserving order
        seen = set()
        unique_hits = []
        for h in hits:
            if h not in seen:
                seen.add(h)
                unique_hits.append(h)

        return unique_hits
