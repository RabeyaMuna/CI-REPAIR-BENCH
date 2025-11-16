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

from utilities.load_config import load_config
from utilities.chunking_logic import chunk_log_by_tokens

load_dotenv()

MAX_TOKENS_SUMMARY = 90_000


class CILogAnalyzer:
    def __init__(
        self,
        repo_path: str,
        ci_log: List[Dict[str, Any]],
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: ChatOpenAI
    ):
        self.config = load_config()
        self.repo_path = repo_path
        self.ci_log = ci_log
        self.sha_fail = sha_fail
        self.workflow = workflow
        self.workflow_path = workflow_path

        self.llm = llm
        self._encoder = self._get_encoder()

        self.error_details: List[Dict[str, Any]] = []

        self.error_keywords = [
            "Traceback",
            "Exception",
            "AssertionError",
            "ImportError",
            "ModuleNotFoundError",
            "TypeError",
            "ValueError",
            "AttributeError",
            "RuntimeError",
            "IndexError",
            "KeyError",
            "SyntaxError",
            "IndentationError",
            "NameError",
            "FileNotFoundError",
            "pytest",
            "E ",
            "FAILED",
            "FAIL",
            "error",
            "fail",
            "exit code 1",
            "black",
            "ruff",
            "flake8",
            "mypy",
            "coverage",
            "unittest",
        ]

    # ------------------------------------------------------------------
    def _extract_relevant_context(self, chunk_text: str, step_name: str) -> Dict[str, Any]:
        """
        Extract relevant error context lines using BM25 and group them.
        """
        lines = chunk_text.splitlines()
        if not lines:
            return {"step_name": step_name, "relevant_failures": []}

        tokenized_corpus = [line.split() for line in lines]
        bm25 = BM25Okapi(tokenized_corpus)
        query = " ".join(self.error_keywords)
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)

        BM25_TOP_N = 100
        CONTEXT_LINES = 10
        top_indices = [
            i
            for i, score in sorted(
                enumerate(scores), key=lambda x: x[1], reverse=True
            )
            if score > 0.0
        ][:BM25_TOP_N]

        relevant_failures = []
        seen = set()

        for idx in top_indices:
            line_text = lines[idx].strip()
            if not line_text or line_text in seen:
                continue
            seen.add(line_text)

            error_type = next(
                (kw for kw in self.error_keywords if kw.lower() in line_text.lower()),
                "UnknownError",
            )
            start = max(0, idx - CONTEXT_LINES)
            end = min(len(lines), idx + CONTEXT_LINES + 1)
            context_window = [lines[i] for i in range(start, end) if lines[i].strip()]

            if "Error" in line_text or "Exception" in line_text:
                parts = line_text.split(":", 1)
                message = parts[1].strip() if len(parts) > 1 else line_text
            else:
                message = line_text

            relevant_failures.append(
                {
                    "line_number": idx + 1,
                    "error_type": error_type,
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

                # Use cached encoder instead of re-creating one
                total_tokens = len(self._encoder.encode(log_text))

                if total_tokens > THRESHOLD:
                    chunks = chunk_log_by_tokens(
                        log_text, max_tokens=80_000, overlap=200
                    )
                    print(
                        f"Chunking activated: {len(chunks)} chunks for '{step_name}'"
                    )
                else:
                    chunks = [log_text]

                all_failures: List[Dict[str, Any]] = []
                for chunk_text in chunks:
                    details = self._extract_relevant_context(chunk_text, step_name)
                    all_failures.extend(details["relevant_failures"])

                combined_query = " ".join(
                    f["message"] for f in all_failures if f.get("message")
                )
                relevant_files = (
                    self._retrieve_relevant_files(combined_query)
                    if combined_query
                    else []
                )

                results.append(
                    {
                        "step_name": step_name,
                        "relevant_failures": all_failures,
                        "relevant_files": relevant_files,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "step_name": step_name,
                        "error": str(e),
                        "relevant_failures": [],
                        "relevant_files": [],
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
            # entry is expected to be: {'step_name', 'relevant_failures', 'relevant_files'}
            if isinstance(entry, dict):
                step_name = entry.get("step_name", "unknown_step")
                relevant_failures = entry.get("relevant_failures", [])
                relevant_files = entry.get("relevant_files", [])
            else:
                # fallback if something weird is passed
                step_name = "unknown_step"
                relevant_failures = []
                relevant_files = []

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
                    failures_text[i : i + CHUNK_SIZE]
                    for i in range(0, len(failures_text), CHUNK_SIZE)
                ]
            else:
                log_chunks = [failures_text]

            # step_name must be valid JSON in the schema example
            step_name_json = json.dumps(step_name)

            for idx, chunk in enumerate(log_chunks):
                prompt = f"""
You are a CI log analyzer.

You receive pre-processed CI log information for a FAILED CI RUN, which contains entries like:
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
      "subcategory": "More specific description, e.g. 'Runtime Error – ValueError in Optuna objective', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
      "evidence": 'Short quote or paraphrase from THIS CHUNK that justifies this classification.'
    }}
  ]
}}

### Rules (IMPORTANT)

1. Use ONLY the information from the chunk shown below.

2. **error_context**:
   - Summarize the root cause(s) found in THIS CHUNK.
   - If nothing meaningful appears, use an empty list [].

3. **relevant_files**:
   - You MUST start from the `relevant_files` list provided in this chunk. Do not invent any new file paths.
   - Additionally, you may use file paths that are explicitly mentioned in the failure text in THIS CHUNK (for example, in stack traces or error messages).
   - Return ONLY those files that are clearly tied to the CI failure in THIS CHUNK.
     A file is "relevant" ONLY if:
       - it is explicitly mentioned in the failure messages or context in THIS CHUNK, OR
       - the errors in THIS CHUNK clearly occur in, or are described as coming from, that file.
   - If you are not sure that a file is truly related to the failure in THIS CHUNK, DO NOT include it.
   - Only select files that have STRONG evidence linking them to the failure in THIS CHUNK.
   - If no file clearly meets these conditions, return "relevant_files": [].
   - For each returned file:
       - use the exact path from the input or from the failure text,
       - set "line_number" to the failing line if visible; otherwise null,
       - provide a short, evidence-based "reason" explaining how THIS CHUNK links the failure to that file.
    - Do not include all the files from the input list unless they are all clearly relevant to give the failure in the log chunk or mentioned in the log chunk and reason of Ci workflow failure.

4. **error_types**:
   - Provide entries that best describe the errors visible in THIS CHUNK.
   - Choose categories/subcategories that match the actual errors (exceptions, test failures, dependency errors, configuration errors, formatting, etc.).

5. Use null for any unknown scalar values.

6. Do NOT add extra top-level keys.

7. Return STRICT JSON ONLY — no markdown, no comments, no natural language outside JSON.

--- LOG DETAILS: relevant failure info from Log CHUNK (JSON/TEXT) ---
{chunk}

--- Relevant Files from Log (JSON/TEXT) ---
{json.dumps(relevant_files, indent=2, ensure_ascii=False)}

--- CI LOG INFORMATION FOR FAILED RUN ---
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
    def _summarize_workflow_if_large(self) -> Any:
        """
        If the workflow content is > 500_000 characters, summarize it chunk by chunk
        using the LLM. Otherwise, return the original workflow object.
        """
        workflow_details = self.workflow
        try:
            workflow_text = json.dumps(
                workflow_details, indent=2, ensure_ascii=False
            )
        except TypeError:
            workflow_text = str(workflow_details)

        MAX_CHARS = 500_000
        if len(workflow_text) <= MAX_CHARS:
            return workflow_details

        print("Workflow content is large; summarizing chunk by chunk with LLM...")

        CHUNK_SIZE = 300_000
        workflow_chunks = [
            workflow_text[i : i + CHUNK_SIZE]
            for i in range(0, len(workflow_text), CHUNK_SIZE)
        ]

        chunk_summaries = []
        for idx, chunk in enumerate(workflow_chunks):
            prompt = f"""
You are a CI workflow analyzer. Your goal is to understand which CI jobs are
responsible for a FAILED CI RUN and what validation they perform.

You will receive:
- A fragment (chunk) of a CI workflow file (YAML/JSON/text).
- CI log information for the failed run.

Your tasks for THIS CHUNK ONLY:

1. Identify **validation jobs** in this chunk:
   - Jobs that run tests, linting, static analysis, type checking, build steps,
     packaging, or coverage.
   - For each such job, record:
     - Its name / id
     - What kind of validation it does (e.g. "tests", "lint", "type-check", "build", "coverage")
     - Which tools it uses (pytest, ruff, black, mypy, coverage, etc.)
     - Key steps and their commands.

2. From those jobs, determine which ones actually **failed** or are clearly
   responsible for the CI failure using the CI logs.

Return STRICT JSON ONLY (no markdown, no comments) with this shape:

{{
  "chunk_index": {idx},
  "jobs": [
    {{
      "job_id_or_name": "string or null",   // from jobs.<id> or 'name' if available
      "is_validation_job": true,            // always true here; only list validation jobs
      "validation_kind": [
        "tests", "lint", "type-check", "build", "coverage"
        // choose one or more that apply, or [] if unclear
      ],
      "validation_tools": [
        // e.g. "pytest", "ruff", "black", "mypy", "coverage"
      ],
      "key_steps": [
        {{
          "step_name": "name if available, else null",
          "command": "main command if present, else null"
        }}
      ]
    }}
  ],

  "failed_jobs": [
    {{
      "job_id_or_name": "name or id of the job that failed or caused failure, or null",
      "validation_kind": [
        "tests", "lint", "type-check", "build", "coverage"
      ],
      "validation_tools": [
        // tools used by this failed job, e.g. "pytest", "ruff"
      ],
      "failed_step": {{
        "step_name": "step name that failed, if known, else null",
        "command": "command that failed, if known, else null"
      }},
      "evidence_from_logs": "short quote or paraphrase from CI logs that proves this job failed",
      "reason": "why this job is considered failed or directly related to the CI failure"
    }}
  ],

  "notes": "any other important CI behavior in this chunk that explains validation or failure"
}}

Rules:

- In "jobs":
  - Include **only validation jobs** found in this chunk.
  - They can be successful or failed; list all validation jobs you can identify.

- In "failed_jobs":
  - **Only** include jobs that actually failed or are clearly responsible for the CI failure.
  - If there is **no failed job in this chunk**, set "failed_jobs" to an empty array: [].
  - Do NOT list successful jobs here.

- Use the CI logs to decide which jobs failed. If uncertain, leave "failed_jobs" as [].

--- CI Workflow Chunk ---
{chunk}

--- CI Log information for failed jobs ---
{self.ci_log}
"""
            try:
                response = self.llm.invoke([HumanMessage(content=prompt)]).content
                try:
                    summary = json.loads(response)
                except json.JSONDecodeError:
                    summary = demjson3.decode(response)
                chunk_summaries.append(summary)
            except Exception as e:
                # If one chunk fails, log and continue
                self._log_error(
                    method="_summarize_workflow_if_large",
                    error=e,
                    step=f"chunk_{idx}",
                )
                chunk_summaries.append({
                    "chunk_index": idx,
                    "jobs": [],
                    "failed_jobs": [],
                    "notes": f"Failed to summarize chunk: {str(e)}"
                })

        return {"chunk_summaries": chunk_summaries}

    # ------------------------------------------------------------------
    def full_content_summary(
        self, log_details: List[Dict[str, Any]], workflow_details: Any
    ) -> Dict[str, Any]:
        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read CI job logs and workflow details, then produce a **structured, evidence-based JSON summary** that classifies the errors clearly by category and subcategory.

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
   - Infer both `category` and `subcategory` based on log and workflow evidence.
   - Each subcategory must be concrete and justifiable.

2. **Error Context**
   - Use 1–3 short English sentences summarizing root causes.
   - Include exception type, file, and line if available.

3. **Relevant Files**
   - Include only files explicitly tied to the failure.
   - Use `"line_number": null` if the line is not specified.
   - Each must have a `"reason"` summarizing the file’s relation to the failure.

4. **Failed Job**
   - Identify the CI job and step that failed, and its triggering command.

5. **Output Rules**
   - Return **only valid JSON** — no markdown, commentary, or code fences.
   - Do not hallucinate; use `null` for unknowns.
   - Merge duplicates and ensure each item is concise, evidence-based, and traceable to the logs.
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
