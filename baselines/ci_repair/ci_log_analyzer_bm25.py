from typing import List, Dict, Any
import json
import os
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


class CILogAnalyzerBM25:
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

    # ------------------------------------------------------------------
    def _extract_relevant_context(self, chunk_text: str, step_name: str) -> Dict[str, Any]:
        """
        Extract relevant error context lines using BM25, but driven by ERROR_KEYWORDS.

        - BM25 ranks all lines using ERROR_KEYWORDS (tokenized) as the query.
        - We only keep lines that actually contain one or more ERROR_KEYWORDS
          (substring match, case-insensitive).
        - For each hit, we capture a context window around it.
        """
        lines = chunk_text.splitlines()
        if not lines:
            return {"step_name": step_name, "relevant_failures": []}

        CONTEXT_LINES = 10
        BM25_TOP_N = 100

        # Normalized, unique keywords for matching/query
        normalized_keywords = list({
            kw.lower().strip()
            for kw in ERROR_KEYWORDS
            if kw.strip()
        })

        # --- BM25 over lines -------------------------------------------------
        # Each line is a "document" represented as tokens
        tokenized_corpus = [line.lower().split() for line in lines]
        bm25 = BM25Okapi(tokenized_corpus)

        # Build query tokens from the keywords (split phrases into tokens)
        query_tokens: List[str] = []
        for kw in normalized_keywords:
            query_tokens.extend(kw.split())

        scores = bm25.get_scores(query_tokens)

        # Indices sorted by BM25 score (high → low)
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )

        relevant_failures: List[Dict[str, Any]] = []
        seen_spans = set()

        for idx in ranked_indices:
            # Limit the number of BM25-selected windows per chunk
            if len(relevant_failures) >= BM25_TOP_N:
                break

            score = scores[idx]
            if score <= 0.0:
                # Because ranked in descending order, remaining will be <= 0 as well
                break

            line = lines[idx]
            if not line.strip():
                continue

            lower_line = line.lower()

            # Only keep lines that actually contain one or more ERROR_KEYWORDS
            matched_keywords = [
                kw for kw in normalized_keywords
                if kw and kw in lower_line
            ]
            if not matched_keywords:
                # BM25 scored it high, but it doesn't contain our keywords → skip
                continue

            # Context window
            start = max(0, idx - CONTEXT_LINES)
            end = min(len(lines), idx + CONTEXT_LINES + 1)
            span = (start, end)

            # Avoid duplicate/overlapping windows
            if span in seen_spans:
                continue
            seen_spans.add(span)

            context_window = [l for l in lines[start:end] if l.strip()]

            primary_keyword = matched_keywords[0]

            # Extract "message" (everything after first ':' if present)
            if ":" in line:
                message_part = line.split(":", 1)[1].strip()
                message = message_part if message_part else line
            else:
                message = line

            relevant_failures.append(
                {
                    "line_number": idx + 1,              # 1-based
                    "error_type": primary_keyword,       # driven by keywords
                    "keywords": matched_keywords,        # all matched keywords
                    "bm25_score": float(score),          # keep BM25 score for debugging
                    "message": message,
                    "context_lines": context_window,
                }
            )

        return {"step_name": step_name, "relevant_failures": relevant_failures}


    # ------------------------------------------------------------------
    def _retrieve_relevant_files(
        self, query_text: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Use BM25 to retrieve the most relevant Python files in the repo
        given the combined error context text.
        """
        documents: List[str] = []
        file_paths: List[str] = []

        for root, _, files in os.walk(self.repo_path):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                            content = fp.read()
                            documents.append(content)
                            file_paths.append(path)
                    except Exception:
                        # Ignore unreadable files
                        continue

        if not documents:
            return []

        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        ranked_files = [
            {
                "file": file_paths[i],
                "score": float(scores[i]),
                "reason": f"Matched tokens from error context (score={scores[i]:.2f})",
            }
            for i in top_indices
            if scores[i] > 0.0
        ]

        return ranked_files

    # ------------------------------------------------------------------
    def ci_log_analysis(self) -> List[Dict[str, Any]]:
        """
        Analyze CI logs using BM25 (for both error context + file retrieval).
        """
        print("Running Tool: BM25 Log + File Retrieval Analyzer")
        results: List[Dict[str, Any]] = []
        THRESHOLD = 80_000

        for step in self.ci_log:
            step_name = step.get("step_name", "unknown_step")
            log = step.get("log", "")
            print(f"\nProcessing Step: {step_name}")

            try:
                log_lines = log if isinstance(log, list) else log.splitlines()
                log_text = "\n".join(log_lines)

                total_tokens = len(self._encoder.encode(log_text))

                if total_tokens > THRESHOLD:
                    chunks = chunk_log_by_tokens(
                        log_text, max_tokens=10_000, overlap=200
                    )
                    print(f"Chunking activated: {len(chunks)} chunks for '{step_name}'")
                else:
                    chunks = [log_text]

                all_failures: List[Dict[str, Any]] = []
                for chunk_text in chunks:
                    details = self._extract_relevant_context(chunk_text, step_name)
                    all_failures.extend(details["relevant_failures"])

                # -------------------------------
                # Build combined_query from message + context_lines
                # -------------------------------
                combined_query_parts: List[str] = []

                for f in all_failures:
                    # message
                    msg = f.get("message")
                    if msg:
                        combined_query_parts.append(str(msg))

                    # context_lines
                    ctx_lines = f.get("context_lines") or []
                    for line in ctx_lines:
                        if line:
                            combined_query_parts.append(str(line))

                combined_query = " ".join(combined_query_parts)

                # -------------------------------
                # Collect full_error_context (deduped) for output
                # -------------------------------
                full_error_context: List[str] = []
                seen_ctx = set()

                for f in all_failures:
                    for line in f.get("context_lines") or []:
                        if not line:
                            continue
                        if line not in seen_ctx:
                            seen_ctx.add(line)
                            full_error_context.append(line)

                # -------------------------------
                # BM25 file retrieval using enriched query
                # -------------------------------
                relevant_files = (
                    self._retrieve_relevant_files(combined_query)
                    if combined_query.strip()
                    else []
                )

                results.append(
                    {
                        "step_name": step_name,
                        "relevant_failures": all_failures,
                        "relevant_files": relevant_files,
                        "full_error_context": full_error_context,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "step_name": step_name,
                        "error": str(e),
                        "relevant_failures": [],
                        "relevant_files": [],
                        "full_error_context": [],
                    }
                )

        return results


    # ------------------------------------------------------------------
    def _generate_summary(self, log_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a structured final error summary from error details,
        workflow tools, and validation checks.
        """
        print(" Running Tool: _generate_summary")
        workflow_details = self.workflow

        log_json = json.dumps(log_details, indent=2, ensure_ascii=False)
        workflow_json = json.dumps(workflow_details, indent=2, ensure_ascii=False)

        total_tokens = self._estimate_tokens(log_json) + self._estimate_tokens(
            workflow_json
        )

        try:
            if total_tokens <= MAX_TOKENS_SUMMARY:
                summary = self.full_content_summary(log_details, workflow_details)
                return summary
            else:
                # FIX: call without argument, it uses self.workflow internally
                reduced_log_details = self._summarize_log_details_if_large(log_details, workflow_json)
                summary = self.full_content_summary(reduced_log_details, workflow_json)
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
    
    def _summarize_log_details_if_large(self, log_details, workflow_details) -> List[Dict[str, Any]]:

        CHUNK_SIZE = 50_000  # characters per chunk
        MAX_TOKENS = 50_000

        log_chunk_summaries: List[Dict[str, Any]] = []

        # Normalize workflow_details to text once
        try:
            workflow_text = json.dumps(workflow_details, indent=2, ensure_ascii=False)
        except TypeError:
            workflow_text = str(workflow_details)

        # Ensure log_details is a list of entries (each should be a dict)
        if isinstance(log_details, list):
            entries = log_details
        else:
            entries = [log_details]

        for entry_idx, entry in enumerate(entries):
            # entry is expected to be: {'step_name', 'relevant_failures', 'relevant_files', 'full_error_context'}
            if isinstance(entry, dict):
                step_name = entry.get("step_name", "unknown_step")
                relevant_failures = entry.get("relevant_failures", [])
                relevant_files = entry.get("relevant_files", [])
                full_error_context = entry.get("full_error_context", [])
            else:
                # fallback if something weird is passed
                step_name = "unknown_step"
                relevant_failures = []
                relevant_files = []
                full_error_context = []

            # Convert relevant_failures to text for token estimate + chunking
            failures_text = json.dumps(relevant_failures, indent=2, ensure_ascii=False)

            # Estimate tokens on the text
            try:
                token_count = self._estimate_tokens(failures_text)
            except Exception:
                token_count = len(failures_text)  # fallback

            # Decide whether to chunk or not
            if token_count > MAX_TOKENS or len(failures_text) > CHUNK_SIZE:
                log_chunks = [
                    failures_text[i: i + CHUNK_SIZE]
                    for i in range(0, len(failures_text), CHUNK_SIZE)
                ]
            else:
                log_chunks = [failures_text]

            # step_name must be valid JSON in the schema example
            step_name_json = json.dumps(step_name)

            for idx, chunk in enumerate(log_chunks):
                prompt = f"""
    You are a CI failure summarization agent for a single CI log CHUNK.

    You receive pre-processed CI log information for a FAILED CI RUN, which contains, for this step:
    - "step_name"
    - "relevant_failures": list of failures with fields such as
        - "line_number"
        - "error_type"
        - "message"
        - optionally "context_lines"
    - "relevant_files": list of candidate source files with fields such as
        - "file"
        - "score"
        - "reason"
    - "full_error_context": additional lines from the same CI step

    Your task for THIS CHUNK ONLY:
    Produce a **structured, evidence-based JSON summary** that explains what went wrong in this chunk
    and identifies the most likely failing Python file(s), if any.

    IMPORTANT:  
    There are **two sources of file path evidence**:

    ### (A) BM25 Candidate Files (under this entry's "relevant_files")
    These are *possible* matches only.  
    Do **NOT** treat them as correct unless the CI log text or error context for THIS STEP also supports them.
    Do **NOT** include them blindly.

    ### (B) Log-Derived File Paths (HIGHEST PRIORITY)
    You MUST scan both:
    - the chunked failure text (shown below as "LOG DETAILS: ..."), and
    - the "full_error_context"
    for any file paths in error messages, context lines, or stack traces.

    If a Python file path appears in CI log evidence (this chunk or full_error_context):
    - It is considered **strong proof** of involvement in this step's failure.
    - It SHOULD be included in "relevant_files" for this chunk, even if BM25 did not rank it.
    - Extract both the normalized file path and the exact line number when visible.

    If a file appears only in BM25 candidates but **not** in any log text (this chunk or full_error_context),
    you MUST NOT include it.

    ---

    ## HOW TO SELECT relevant_files (IMPORTANT LOGIC FOR THIS CHUNK)

    1. **Extract all `.py` paths mentioned in error messages**, context lines, or stack traces
    in THIS CHUNK and in full_error_context:
    - Example patterns:
        - "examples/foo/bar.py:15:1: error"
        - "path/to/file.py:42: ..."
        - "File \"path/to/file.py\", line 99"
    - Normalize the path:
        - remove trailing ":line:col" segments,
        - keep the ".py" path only (e.g., "examples/foo/bar.py").

    2. **Pick the file(s) that show the clearest evidence of causing the CI failure in THIS CHUNK.**  
    Evidence includes:
    - ruff errors (I001, F401, etc.),
    - flake8/mypy errors,
    - Python traceback paths,
    - test failure assertions,
    - "File ... line ..." patterns or similar.

    3. **If multiple error files appear**, include all of them in "relevant_files" for this chunk.

    4. **If no explicit file appears in logs for THIS CHUNK and full_error_context**:
    - Return an empty "relevant_files" list for this chunk.

    5. For each selected file:
        - "file" = normalized Python file path,
        - "line_number" = first line number extracted from logs for that file, or null if unknown,
        - "reason" = short explanation quoting or paraphrasing the specific log line that links THIS CHUNK'S failure to this file.

    ---

    For THIS CHUNK ONLY, extract a **partial summary** of the CI failure using
    the following STRICT JSON schema (do not add or remove keys):

    {{
    "step_name": {step_name_json},
    "chunk_index": {idx},
    "sha_fail": "{self.sha_fail}",
    "error_context": [
        "English explanation(s) of the root cause(s) visible in THIS CHUNK, supported by log evidence."
    ],
    "relevant_files": [
        {{
        "file": "path/to/file.py",
        "line_number": 123,
        "reason": "Short explanation of why this file is tied to the failure in THIS CHUNK."
        }}
    ],
    "error_types": [
        {{
        "category": "High-level category, e.g. 'Test Failure', 'Runtime Error', 'Dependency Error', 'Configuration Error', 'Code Formatting', 'Type Checking'",
        "subcategory": "More specific description, e.g. 'Code Formatting – ruff I001 unsorted imports', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
        "evidence": "Short quote or paraphrase from THIS CHUNK or full_error_context that justifies this classification."
        }}
    ]
    }}

    ### Rules (IMPORTANT)

    1. Use ONLY the information provided in this prompt:
    - the chunked relevant_failures text ("LOG DETAILS" below),
    - the BM25 candidate "relevant_files" for this step,
    - the "full_error_context" for this step,
    - and the workflow_text (if useful for context).

    2. **error_context**:
    - Summarize the root cause(s) found in THIS CHUNK in 1–3 short sentences.
    - If nothing meaningful appears, use an empty list [].

    3. **relevant_files** (STRICT RULES):
    - Extract file paths from logs first (this chunk + full_error_context).
    - Normalize paths (remove trailing ":line:col" segments).
    - Use "line_number": null if the line is not visible.
    - Only include files that have STRONG evidence linking them to the failure
        in THIS CHUNK or its full_error_context.
    - Do NOT include any file that appears only as a BM25 candidate without log evidence.
    - If no file clearly meets these conditions, set "relevant_files": [].

    4. **error_types**:
    - Provide entries that best describe the errors visible in THIS CHUNK.
    - Categories/subcategories must match actual errors (tests, runtime, dependencies, formatting, type checking, etc.).
    - Each "evidence" must be clearly supported by text from THIS CHUNK or full_error_context.

    5. Use null for any unknown scalar values.

    6. Do NOT add extra top-level keys, and do NOT change the schema.

    7. Return STRICT JSON ONLY — no markdown, no comments, no natural language outside JSON.

    --- LOG DETAILS: relevant failure info from Log CHUNK (JSON/TEXT) ---
    {chunk}

    --- BM25 Candidate Relevant Files for this Step (JSON/TEXT) ---
    {json.dumps(relevant_files, indent=2, ensure_ascii=False)}

    --- Full Error Context from Log for this Step (JSON/TEXT) ---
    {json.dumps(full_error_context, indent=2, ensure_ascii=False)}

    --- CI WORKFLOW / RUN CONTEXT (JSON/TEXT) ---
    {workflow_text}
    """

                try:
                    response = self.llm.invoke([HumanMessage(content=prompt)]).content
                    try:
                        summary = json.loads(response)
                    except json.JSONDecodeError:
                        summary = demjson3.decode(response)
                        
                    log_chunk_summaries.append(summary)
                except Exception as e:
                    # If one chunk fails, log and continue
                    self._log_error(
                        method="_summarize_log_details_if_large",
                        error=e,
                        step=f"entry_{entry_idx}_chunk_{idx}",
                    )
                    
                    log_chunk_summaries.append(
                        {
                            "step_name": step_name,
                            "chunk_index": idx,
                            "sha_fail": self.sha_fail,
                            "error_context": [],
                            "relevant_files": [],
                            "error_types": [],
                            "error": f"Failed to summarize chunk: {str(e)}",
                        }
                    )

        return log_chunk_summaries


            
    # ------------------------------------------------------------------
#     def _summarize_workflow_if_large(self) -> Any:
#         """
#         If the workflow content is > 500_000 characters, summarize it chunk by chunk
#         using the LLM. Otherwise, return the original workflow object.
#         """
#         workflow_details = self.workflow
#         try:
#             workflow_text = json.dumps(
#                 workflow_details, indent=2, ensure_ascii=False
#             )
#         except TypeError:
#             workflow_text = str(workflow_details)

#         MAX_CHARS = 500_000
#         if len(workflow_text) <= MAX_CHARS:
#             return workflow_details

#         print("Workflow content is large; summarizing chunk by chunk with LLM...")

#         CHUNK_SIZE = 300_000
#         workflow_chunks = [
#             workflow_text[i : i + CHUNK_SIZE]
#             for i in range(0, len(workflow_text), CHUNK_SIZE)
#         ]

#         chunk_summaries = []
#         for idx, chunk in enumerate(workflow_chunks):
#             prompt = f"""
# You are a CI workflow analyzer. Your goal is to understand which CI jobs are
# responsible for a FAILED CI RUN and what validation they perform.

# You will receive:
# - A fragment (chunk) of a CI workflow file (YAML/JSON/text).
# - CI log information for the failed run.

# Your tasks for THIS CHUNK ONLY:

# 1. Identify **validation jobs** in this chunk:
#    - Jobs that run tests, linting, static analysis, type checking, build steps,
#      packaging, or coverage.
#    - For each such job, record:
#      - Its name / id
#      - What kind of validation it does (e.g. "tests", "lint", "type-check", "build", "coverage")
#      - Which tools it uses (pytest, ruff, black, mypy, coverage, etc.)
#      - Key steps and their commands.

# 2. From those jobs, determine which ones actually **failed** or are clearly
#    responsible for the CI failure using the CI logs.

# Return STRICT JSON ONLY (no markdown, no comments) with this shape:

# {{
#   "chunk_index": {idx},
#   "jobs": [
#     {{
#       "job_id_or_name": "string or null",   // from jobs.<id> or 'name' if available
#       "is_validation_job": true,            // always true here; only list validation jobs
#       "validation_kind": [
#         "tests", "lint", "type-check", "build", "coverage"
#         // choose one or more that apply, or [] if unclear
#       ],
#       "validation_tools": [
#         // e.g. "pytest", "ruff", "black", "mypy", "coverage"
#       ],
#       "key_steps": [
#         {{
#           "step_name": "name if available, else null",
#           "command": "main command if present, else null"
#         }}
#       ]
#     }}
#   ],

#   "failed_jobs": [
#     {{
#       "job_id_or_name": "name or id of the job that failed or caused failure, or null",
#       "validation_kind": [
#         "tests", "lint", "type-check", "build", "coverage"
#       ],
#       "validation_tools": [
#         // tools used by this failed job, e.g. "pytest", "ruff"
#       ],
#       "failed_step": {{
#         "step_name": "step name that failed, if known, else null",
#         "command": "command that failed, if known, else null"
#       }},
#       "evidence_from_logs": "short quote or paraphrase from CI logs that proves this job failed",
#       "reason": "why this job is considered failed or directly related to the CI failure"
#     }}
#   ],

#   "notes": "any other important CI behavior in this chunk that explains validation or failure"
# }}

# Rules:

# - In "jobs":
#   - Include **only validation jobs** found in this chunk.
#   - They can be successful or failed; list all validation jobs you can identify.

# - In "failed_jobs":
#   - **Only** include jobs that actually failed or are clearly responsible for the CI failure.
#   - If there is **no failed job in this chunk**, set "failed_jobs" to an empty array: [].
#   - Do NOT list successful jobs here.

# - Use the CI logs to decide which jobs failed. If uncertain, leave "failed_jobs" as [].

# --- CI Workflow Chunk ---
# {chunk}

# --- CI Log information for failed jobs ---
# {self.ci_log}
# """
#             try:
#                 response = self.llm.invoke([HumanMessage(content=prompt)]).content
#                 try:
#                     summary = json.loads(response)
#                 except json.JSONDecodeError:
#                     summary = demjson3.decode(response)
#                 chunk_summaries.append(summary)
#             except Exception as e:
#                 # If one chunk fails, log and continue
#                 self._log_error(
#                     method="_summarize_workflow_if_large",
#                     error=e,
#                     step=f"chunk_{idx}",
#                 )
#                 chunk_summaries.append({
#                     "chunk_index": idx,
#                     "jobs": [],
#                     "failed_jobs": [],
#                     "notes": f"Failed to summarize chunk: {str(e)}"
#                 })

#         return {"chunk_summaries": chunk_summaries}

    # ------------------------------------------------------------------
    def full_content_summary(
        self, log_details: List[Dict[str, Any]], workflow_details: Any
    ) -> Dict[str, Any]:
        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read CI job logs and workflow details, then produce a **structured, evidence-based JSON summary** that classifies the errors clearly by category and subcategory.

IMPORTANT:  
There are **two sources of file path evidence**:

### (A) BM25 Candidate Files (under log_details → relevant_files)
These are *possible* matches only.  
Do **NOT** treat them as correct unless the CI log text or error context or CI failure reason also supports them. Do **NOT** include them blindly. 

### (B) Log-Derived File Paths (highest priority)
You MUST scan all `context_lines` and `full_error_context` to define any file paths mentioned which files caused the CI failure.

If a file appears in CI log evidence:
- It is considered **strong proof** of involvement.
- It MUST be included in `relevant_files` (even if BM25 did not rank it).
- Extract both the normalized file path and the exact line number.

If a file appears only in BM25 candidates but **not** in log text, you MUST NOT include it.

---

## HOW TO SELECT relevant_files (IMPORTANT LOGIC)

1. **Extract all `.py` paths mentioned in error messages**, context lines, or stack traces:
   - Example: `examples/foo/bar.py:15:1: error`
   - Normalize the path (remove line/column suffix, keep `.py` path).

2. **Pick the file(s) that show the clearest evidence of causing the CI failure.**  
   Evidence includes:
   - ruff errors (I001, F401, etc.)
   - flake8/mypy errors
   - Python traceback paths
   - test failure assertions
   - “File ... line ...” patterns

3. **If multiple error files appear**, include all of them.

4. **If no explicit file appears in logs**, return an empty `relevant_files` list.

5. For each file:
   - `"file"` = normalized Python file path
   - `"line_number"` = first line number extracted from logs, or `null`
   - `"reason"` = short explanation quoting the specific log line

---

## INPUTS

1. CI Log Details (from step analysis):
{json.dumps(log_details, indent=2, ensure_ascii=False)}

2. Workflow Details:
{json.dumps(workflow_details, indent=2, ensure_ascii=False)}

---

## OUTPUT FORMAT (strict JSON only)

{{
  "sha_fail": "{self.sha_fail}",
  "error_context": [
    "Plain-English explanation(s) of the root cause(s), supported by log evidence."
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

1. **Derive — do not assume.**
   - Infer both `category` and `subcategory` based only on CI log evidence.
   - Every classification must quote or paraphrase log lines.

2. **Error Context**
   - Use 1–3 short English sentences summarizing the true root cause(s).
   - Include file name and line numbers when available.

3. **Relevant Files — STRICT RULES**
   - Extract file paths from logs first (highest priority).
   - Normalize paths (remove trailing `:line:col` segments).
   - Use `"line_number": null` if missing.
   - For each file, explain the exact log message that proves it is involved.
   - BM25 candidates should be used only to support ranking—not as evidence.

4. **Failed Job**
   - Identify the exact CI job and step that failed.
   - Identify the command/tool that produced the error (e.g., ruff, pytest, mypy).

5. **Output Rules**
   - Return **only valid JSON** — no markdown, commentary, or code fences.
   - Do not hallucinate; use `null` for unknowns.
   - Merge duplicates; ensure items are concise, evidence-based, and traceable.

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
        generated_summary = self._generate_summary(log_details)
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

    def filter_chunks(raw_chunks):
        n_chunks = len(raw_chunks)

        # Always keep last 6 chunks (unconditionally)
        last_start = max(0, n_chunks - 6)
        last_chunks = raw_chunks[last_start:]

        filtered_chunks = []

        # 1. Check all chunks BEFORE the last 6
        for chunk in raw_chunks[:last_start]:
            text_lower = chunk.lower()

            # If any keyword found → keep this chunk
            if any(kw.lower() in text_lower for kw in ERROR_KEYWORDS):
                filtered_chunks.append(chunk)

        # 2. Add the last 6 chunks without keyword checking
        filtered_chunks.extend(last_chunks)

        chunks = filtered_chunks

        print(
            f"Filtered from {len(raw_chunks)} ➝ {len(chunks)} chunks "
            f"(kept {len(last_chunks)} last-chunks + keywords matches)"
        )