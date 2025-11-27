from typing import List, Dict, Any
import json
import os
import time
import re
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
                        "relevant_files": relevant_files
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
        fetch_log_details = self.process_log_details(log_details)
        log_json = json.dumps(fetch_log_details, indent=2, ensure_ascii=False)
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
    - "keywords"
    - "bm25_score"
    - "message"
    - "context_lines" (raw CI log lines for that failure)
- "relevant_files": list of BM25 candidate source files with fields such as
    - "file"
    - "score"
    - "reason"

Your task for THIS CHUNK ONLY:
Produce a **structured, evidence-based JSON summary** that:
1. Explains what went wrong in this chunk (overall error context).
2. Identifies only those Python file(s) that have **clear evidence** of being tied to the CI failure in this step.

---

## SOURCES OF FILE PATH EVIDENCE

You have two sources of file paths:

### (A) BM25 Candidate Files (this entry's "relevant_files")
These are **candidate** files only.  
They are NOT automatically correct.

You may only include a BM25 candidate in the final "relevant_files" if:

1. The same file path (or a very clear variant of it) also appears in the log content of THIS CHUNK
   (inside the `message` or `context_lines` fields of `relevant_failures`), **AND**
2. The surrounding log context clearly indicates that this file is involved in the failure
   (for example, it appears in a traceback, a failing test error, a lint/formatting error, or
   some message that directly reports an error in that file).

If a BM25 file never appears in any `message` or `context_lines` in THIS CHUNK,
or the context does not link it to an actual error, you MUST NOT include it.

### (B) Log-Derived File Paths (HIGHEST PRIORITY)
You must scan all `message` and `context_lines` entries inside `relevant_failures`
for file paths that appear in error messages, context lines, or stack traces.

Examples of log patterns:
- "examples/foo/bar.py:15:1: error"
- "path/to/file.py:42: ..."
- "File \\"path/to/file.py\\", line 99"

From these, extract the **normalized** Python file path by:
- stripping trailing ":line:col" segments,
- keeping only the ".py" path (e.g. "examples/foo/bar.py").

You may include a log-derived file in the final "relevant_files" even if it is NOT in the BM25 list,
but only when the log context clearly shows it is part of the failure
(e.g., the file appears in the last frames of a traceback, in a failing test error,
or in a lint/formatting error that caused the step to fail).

A file that appears in logs without any clear error context (for example, just printed as a path
or part of a search path) MUST NOT be included.

---

## HOW TO SELECT relevant_files (STRICT LOGIC FOR THIS CHUNK)

For THIS CHUNK ONLY:

1. From `relevant_failures` (messages + context_lines), extract all paths.

2. For each path, decide if the log text clearly ties it to the CI failure, such as:
   - it appears in a traceback or error message,
   - it is reported as failing a test,
   - it is identified in a lint/formatting error,
   - it is explicitly referenced as the source of the failure.

3. Cross-check with BM25 candidates:
   - If a file is a BM25 candidate **and** has strong log evidence of being involved in the CI failure directly and indirectly, you SHOULD include it.
   - If a BM25 candidate is **not** supported by log evidence in THIS CHUNK, you MUST NOT include it.
   - If a file is **not** in BM25 but the logs clearly show it is the error-causing file, you SHOULD include it (log evidence is more important than BM25 scores).

4. If multiple error-related files appear with clear evidence, include all of them.

5. If no file has clear evidence of being involved in THIS CHUNK's failure,
   set "relevant_files": [].

For each selected file:
- "file"        = normalized path,
- "line_number" = first line number extracted from logs for that file (or null if unknown),
- "reason"      = short explanation quoting or paraphrasing the relevant log lines that link
                  THIS CHUNK's failure to this file.

---

## WHAT TO OUTPUT FOR THIS CHUNK

For THIS CHUNK ONLY, extract a **partial summary** of the CI failure using
the following STRICT JSON schema (do not add or remove keys):

{{
  "step_name": {step_name_json},
  "chunk_index": {idx},
  "sha_fail": "{self.sha_fail}",
  "error_context": [
    "English explanation(s) of the root cause(s) visible in THIS CHUNK, supported by log evidence from relevant_failures. Describe what this step was trying to do and why it failed in this chunk. Mention key errors, exit codes, and any clearly error-related files."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "reason": "Short explanation of why this file is tied to the failure in THIS CHUNK, quoting or paraphrasing the relevant log lines."
    }}
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Test Failure', 'Runtime Error', 'Dependency Error', 'Configuration Error', 'Code Formatting', 'Type Checking'",
      "subcategory": "More specific description, e.g. 'Code Formatting – ruff I001 unsorted imports', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
      "evidence": "Short quote or paraphrase from THIS CHUNK (from message/context_lines) that justifies this classification."
    }}
  ]
}}

---

## RULES (VERY IMPORTANT)

1. Use ONLY the information provided in this prompt:
   - the chunked relevant_failures text ("LOG DETAILS" below),
   - the BM25 candidate "relevant_files" for this step,
   - the workflow_text (if present in the prompt) only for high-level context, not for file selection.

2. **error_context**:
   - Summarize the root cause(s) visible in THIS CHUNK only, in 1–3 short sentences.
   - If there is no meaningful failure information in this chunk, use "error_context": [].

3. **relevant_files**:
   - MUST be derived from file paths that are clearly tied to error messages in THIS CHUNK.
   - A BM25 candidate that never appears in `message` or `context_lines`, or appears without any error/traceback context, MUST be excluded.
   - A file that appears in logs without any error/traceback context MUST be excluded.
   - Only include files that have strong evidence of being involved in the failure in this chunk.

4. **error_types**:
   - Provide entries that best describe the errors visible in THIS CHUNK.
   - Categories/subcategories must be consistent with the actual errors (tests, runtime, dependencies, formatting, type checking, etc.).
   - Each "evidence" must be clearly supported by text taken from `message` or `context_lines`.

5. Use null for any unknown scalar values.

6. Do NOT add extra top-level keys, and do NOT change the schema.

7. Return STRICT JSON ONLY — no markdown, no comments, no natural language outside JSON.

---

--- LOG DETAILS: relevant_failures for this CHUNK (JSON/TEXT) ---
{chunk}

--- BM25 Candidate Relevant Files for this Step (JSON/TEXT) ---
{json.dumps(relevant_files, indent=2, ensure_ascii=False)}

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

    def process_log_details(self, log_details: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_steps: List[Dict[str, Any]] = []

        for entry_idx, entry in enumerate(log_details):
            step_name = entry.get("step_name", "unknown_step")
            relevant_failures = entry.get("relevant_failures", [])
            relevant_files = entry.get("relevant_files", [])

            if not relevant_failures:
                continue

            # 1) Check total token size for all failures
            failures_json_str = json.dumps(relevant_failures, indent=2)
            total_tokens = self._estimate_tokens(failures_json_str)

            if total_tokens > 90_000:
                # Chunk the LIST, not the token count
                failure_chunks = self._split_failures_by_tokens(
                    relevant_failures,
                    max_tokens=90_000,
                )
            else:
                failure_chunks = [relevant_failures]

            combined_failures: List[Dict[str, Any]] = []

            # 2) Process each chunk with the LLM
            for chunk_idx, failures_chunk in enumerate(failure_chunks):
                prompt = f"""
You are a CI log explanation assistant.

You will receive **one CI step** and a subset of its detected failures. Your job is to produce:
1. A high-level natural-language explanation (`error_context`) summarizing what this subset of failures means for the step.
2. One **combined** natural-language explanation (`relevant_failures`) that walks through **all failures in this subset**, preserving every piece of information.

---

## INPUT

step_name:
{step_name}

subset_of_relevant_failures (list of dictionaries):
{json.dumps(failures_chunk, indent=2)}

---

## TASK

For this subset of failures:

1. Carefully read every failure in `subset_of_relevant_failures`. For each failure you get:
   - line_number
   - error_type
   - keywords
   - bm25_score
   - message
   - context_lines (raw CI log lines)

2. Use these fields to build a **single continuous narrative** in natural language that:
   - Mentions every failure, one by one or grouped logically.
   - For each failure, includes:
     - its line_number and error_type,
     - the message (rewritten or quoted where useful),
     - the keywords and the exact bm25_score value (for example: “for keywords ['error', 'traceback'] with bm25_score=34.91”),
     - a natural-language description of what the original context_lines show.
   - From `context_lines`, extract:
     - all file paths and explain their roles (e.g., “this is the test file that failed”, “this is part of the Python standard library in the traceback”, “this is a project source file”, “this is a configuration file”),
     - all error messages, exception types, exit codes, failing commands, tests, modules, etc.
   - You may remove superficial noise (timestamps, repeated prefixes), but you must **not drop any distinct factual information**.

3. `error_context`:
   - Summarize, at a higher level, what these failures collectively indicate:
     - what this step is trying to do,
     - what kinds of errors are occurring,
     - which main files or commands are implicated,
     - how the keywords and bm25_score values show that these logs are strongly related to errors.

4. `relevant_failures`:
   - Produce **one long string** that describes all failures in detail.
   - The text should be coherent and readable but exhaustive:
     - Every failure from the input must be represented.
     - All file paths must be mentioned and explained.
     - All error codes, exception names, test names, or command names must be preserved.
     - All keywords and their bm25_score values must be explicitly included somewhere in the narrative.

5. Do not return raw log lines; instead, always paraphrase them into clear natural language while keeping their meaning.

---

## OUTPUT FORMAT (STRICT JSON)

Return a single JSON object with exactly these keys:

{{
  "step_name": "{step_name}",
  "error_context": "High-level explanation for THIS CHUNK ONLY (subset of failures), summarizing what went wrong, which files and commands were involved, and how the keywords and bm25_score values support this.",
  "relevant_failures": "One combined natural-language explanation for ALL failures in this subset. This string must include, for every failure, its line_number, error_type, message, keywords, bm25_score, and a natural-language reconstruction of its context_lines, mentioning all files and important details without losing any information."
}}

Rules:
- The actual JSON must not contain comments.
- step_name in the output must match the input step_name.
- error_context and relevant_failures must be strings.
- You must not omit any failure or any important detail from the input.
- The response MUST be valid JSON with double quotes and no trailing commas.
""".strip()


                try:
                    raw_response = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()

                    try:
                        chunk_summary = json.loads(raw_response)
                    except json.JSONDecodeError:
                        chunk_summary = demjson3.decode(raw_response)
                        
                    combined_failures.append(chunk_summary)

                except Exception as e:
                    self._log_error(
                        method="process_log_details",
                        error=e,
                        step=f"entry_{entry_idx}_chunk_{chunk_idx}",
                    )

            processed_steps.append(
                {
                    "step_name": step_name,
                    "relevant_failures": combined_failures,
                    "relevant_files": relevant_files
                }
            )

        return processed_steps

    # ------------------------------------------------------------------
    def full_content_summary(self, log_details: List[Dict[str, Any]], workflow_details: Any) -> Dict[str, Any]:
        log_details_text = json.dumps(log_details, indent=2, ensure_ascii=False)
        workflow_text = json.dumps(workflow_details, indent=2, ensure_ascii=False)
        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read CI job logs and workflow details, then produce a **structured, evidence-based JSON summary** that clearly explains:
- what failed,
- which files are actually involved in the failure,
- how to classify the error(s) by category and subcategory,
- which CI job/step/command failed.

IMPORTANT:  
There are **two sources of file path evidence** in the input:

### (A) BM25 Candidate Files (under log_details[*] → relevant_files)
These are *possible* matches only.  
They are NOT guaranteed to be correct.
You must **NOT** include a BM25 candidate file as relevant unless:
1) The same file path (or a clear variant of it) appears in the CI log text
   (inside `message` or `context_lines` of some `relevant_failures` entry), AND
2) The surrounding log context clearly ties that file to an error, test failure, traceback,
   lint/formatting issue, or other failure reason.

If a file appears only as a BM25 candidate but **never appears in any log text** related to errors,
or if there is no clear evidence that the file is involved in the failure,
you MUST NOT include it in the final `relevant_files` output.

### (B) Log-Derived File Paths (highest priority)
Your highest-quality evidence comes from **log text itself**.

You MUST scan all `message` and `context_lines` fields inside `log_details[*]["relevant_failures"]`
to find file paths that appear in error messages, context lines, or stack traces.

Examples of log patterns:
- "examples/foo/bar.py:15:1: error"
- "path/to/file.py:42: ..."
- "File \\"path/to/file.py\\", line 99"

From these, extract the **normalized** Python file path by:
- stripping trailing ":line:col" segments,
- keeping only the file path in repo-style form with forward slashes, e.g. "examples/foo/bar.py".

If a file appears in CI log evidence in a way that clearly indicates it is involved in the failure
(e.g., in a traceback, a failing test, a ruff/flake8/mypy error, etc.):
- It is considered **strong proof** of involvement.
- It SHOULD be included in the final `relevant_files` output (even if BM25 did not rank it).
- Extract both the normalized file path and the first visible line number when available.

However, a file that appears in logs **without** any error/traceback/test-failure context
(for example, just printed as part of a path or environment variable) MUST NOT be treated
as an error-causing file.

---

## HOW TO SELECT relevant_files (GLOBAL LOGIC)

Working over **all steps** in `log_details`:

1. From every step's `relevant_failures`:
   - Look into `message` and `context_lines`.
   - Extract all file paths that appear in error messages, tracebacks, or test failures.
   - Normalize the paths as described above.

2. For each file file path, decide if the surrounding log context clearly ties it to the CI failure:
   - It appears in a traceback or exception message.
   - It is reported as a failing test file.
   - It is the target of a lint/formatting error (e.g., ruff, flake8, black, isort).
   - It is mentioned in a mypy/type-checking failure.
   - It appears in a "File ... line ..." frame that is part of the root cause, NOT just a framework/helper.

3. Cross-check with BM25 candidates for each step:
   - If a file is a BM25 candidate **and** logs clearly show error involvement, you SHOULD include it.
   - If a BM25 candidate is **not** supported by log evidence, you MUST NOT include it.
   - If a file is **not** in BM25 but logs clearly show it as an error-causing file, you SHOULD include it.

4. If multiple files have strong, log-based evidence of being involved in the failure, include all of them.

5. If no file has clear evidence of being involved in the failure across all steps, set `relevant_files`: [].

For each selected file in the final summary:
- `"file"`        = normalized file path
- `"line_number"` = first line number extracted from logs for that file, or `null` if unknown
- `"reason"`      = short explanation quoting or paraphrasing the specific log lines that link
                    this file to the failure (e.g., traceback frame, lint error, failing test, etc.)

---

## INPUTS
1. Failed Commit(sha_fail): {self.sha_fail}
2. CI Log Details (from step analysis, ALL STEPS):

{log_details_text}

Each entry typically has:
- "step_name": name of the CI step,
- "relevant_failures": list of failure dicts with:
    - "line_number"
    - "error_type"
    - "keywords"
    - "bm25_score"
    - "message"
    - "context_lines" (raw or summarized CI log lines),
- "relevant_files": BM25 candidate files (file, score, reason).

3. Workflow Details (jobs, steps, and commands):

{workflow_text}

Use these details to identify which job/step failed and which command/tool (e.g., ruff, pytest, mypy)
was actually running when the failure occurred.

---

## OUTPUT FORMAT (strict JSON only)

Return a single JSON object in exactly this shape:

{{
  "sha_fail": "{self.sha_fail}",
  "error_context": [
    "Plain-English explanation(s) of the root cause(s), supported by log evidence. Summarize what failed, why it failed, and which step(s)/tool(s) are responsible."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "reason": "Short evidence-based explanation of why this file is tied to the failure, quoting or paraphrasing the relevant log lines."
    }}
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Code Formatting', 'Dependency Error', 'Test Failure', 'Runtime Error', 'Type Checking', 'Configuration Error'",
      "subcategory": "More specific type under that category, e.g. 'Unused Import', 'Line Length Exceeded', 'ImportError: No module named X', 'AssertionError in unit test', 'Mypy type mismatch', 'Missing dependency x'",
      "evidence": "Brief quote or paraphrase from CI logs that proves this classification. This must reference specific messages, context_lines, or error text."
    }}
  ],
  "failed_job": [
    {{
      "job": "Job name or ID taken from workflow_details or logs",
      "step": "Specific step name that failed, as seen in log_details or workflow_details",
      "command": "Exact command or action that caused the failure, such as 'ruff check', 'pytest', 'mypy', 'python -m tox', etc."
    }}
  ]
}}

---

## INSTRUCTIONS

1. **Derive — do not assume.**
   - Infer both `category` and `subcategory` based only on CI log evidence.
   - Every classification in `error_types` must be justified by `evidence` taken from log text.

2. **Error Context**
   - Use 1–3 short English sentences in `error_context` that summarize the true root cause(s),
     combining information across all steps and all chunks.
   - Mention file names and line numbers when available. Mentioned file path in a normailized form as well.
   - If multiple distinct error categories exist (e.g. both formatting and tests), mention them.

3. **Relevant Files — STRICT RULES**
   - Extract file paths from log text first (highest priority).
   - Normalize paths (remove trailing `:line:col` segments).
   - Use `"line_number": null` if the line is not visible.
   - For each file, explain the exact log message(s) that prove it is involved.
   - BM25 candidates should only support ranking; they are *never* sufficient evidence on their own.
   - If there is no strong evidence that any file is involved, return `"relevant_files": []`.

4. **Failed Job**
   - Use workflow_details and log_details to identify:
     - the CI job that failed,
     - the specific step (by name) where the failure occurred,
     - the command or tool that produced the error (e.g., `ruff check`, `pytest`, `mypy`, `tox -e py312`).
   - If some field is unknown, set its value to `null`.

5. **Output Rules**
   - Must provide `sha_fail` exactly as given.
   - Return **only valid JSON** — no markdown, no comments, no code fences.
   - Do **not** invent file paths, commands, or job names; use `null` where something is truly unknown.
   - Merge duplicates and keep lists concise, but **do not drop distinct error categories or distinct relevant files**.
   - All items must be evidence-based and traceable to the input logs.

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
        if generated_summary["sha_fail"] in (None, "", "null", "unknown"):
            generated_summary["sha_fail"] = self.sha_fail
        return generated_summary

    # ------------------------------------------------------------------
    def _get_encoder(self):
        """Safely get a tiktoken encoder for the model."""
        try:
            return tiktoken.encoding_for_model(self.model_name)
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