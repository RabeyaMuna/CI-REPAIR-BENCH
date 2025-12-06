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
CHUNK_SIZE_LOG = 80_000
CONTEXT_LINES = 5          # reduced to shrink windows
BM25_TOP_N = 30           # reduced to avoid huge lists


class CILogAnalyzerBM25:
    def __init__(
        self,
        repo_path: str,
        ci_log: List[Dict[str, Any]],
        sha_fail: str,
        workflow: Any,
        workflow_path: str,
        llm: ChatOpenAI,
        model_name: str,
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

    # ------------------ BM25 Helpers ------------------
    def _tokenize_lines(self, text: str) -> List[List[str]]:
        return [line.lower().split() for line in text.splitlines() if line.strip()]

    def _rank_lines_with_bm25(self, lines: List[str], keywords: List[str]):
        """
        Rank log lines by BM25 relevance to the error keywords.
        Returns (ranked_indices, scores).
        """
        tokenized_corpus = self._tokenize_lines("\n".join(lines))
        if not tokenized_corpus:
            return [], []
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = [tok for kw in keywords for tok in kw.split()]
        if not query_tokens:
            return list(range(len(lines))), [0.0] * len(lines)
        scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return ranked_indices, scores

    # ------------------ Chunk filtering helper ------------------
    @staticmethod
    def filter_chunks(raw_chunks: List[str]) -> List[str]:
        """
        Keep:
        - all last 6 chunks (tail of the log),
        - plus any earlier chunks that contain ERROR_KEYWORDS.
        """
        n_chunks = len(raw_chunks)
        if n_chunks == 0:
            return []

        last_start = max(0, n_chunks - 6)
        last_chunks = raw_chunks[last_start:]

        filtered_chunks: List[str] = []

        for chunk in raw_chunks[:last_start]:
            text_lower = chunk.lower()
            if any(kw.lower() in text_lower for kw in ERROR_KEYWORDS):
                filtered_chunks.append(chunk)

        filtered_chunks.extend(last_chunks)

        print(
            f"Filtered from {len(raw_chunks)} ➝ {len(filtered_chunks)} chunks "
            f"(kept {len(last_chunks)} last-chunks + keyword matches)"
        )
        return filtered_chunks

    # ------------------ Log flatten ------------------
    def flatten_log(self, log):
        if isinstance(log, str):
            return log
        elif isinstance(log, list):
            flat_lines = []
            for item in log:
                if isinstance(item, str):
                    flat_lines.append(item)
                elif isinstance(item, list):
                    flat_lines.extend(str(i) for i in item)
                else:
                    flat_lines.append(str(item))
            return "\n".join(flat_lines)
        else:
            return str(log)

    # ------------------ CI Log Analysis ------------------
    def _extract_relevant_context(self, chunk_text: str, step_name: str) -> Dict[str, Any]:
        """
        Use BM25 over lines in this chunk to pick top error-context windows.
        """
        lines = chunk_text.splitlines()
        if not lines:
            return {"step_name": step_name, "relevant_failures": []}

        normalized_keywords = list({kw.lower().strip() for kw in ERROR_KEYWORDS if kw.strip()})
        ranked_indices, scores = self._rank_lines_with_bm25(lines, normalized_keywords)

        relevant_failures: List[Dict[str, Any]] = []
        covered_indices = set()  # to avoid overlapping windows

        for idx in ranked_indices:
            # Stop if we already collected enough
            if len(relevant_failures) >= BM25_TOP_N:
                break

            # Guard against empty scores & non-positive scores
            if scores is not None and len(scores) > idx and scores[idx] <= 0:
                continue

            line = lines[idx].strip()
            if not line:
                continue

            matched_keywords = [kw for kw in normalized_keywords if kw in line.lower()]
            if not matched_keywords:
                continue

            start = max(0, idx - CONTEXT_LINES)
            end = min(len(lines), idx + CONTEXT_LINES + 1)
            span_range = range(start, end)

            # Skip if region already covered
            if any(i in covered_indices for i in span_range):
                continue

            covered_indices.update(span_range)

            context_window = [l for l in lines[start:end] if l.strip()]
            message = line.split(":", 1)[1].strip() if ":" in line else line

            relevant_failures.append(
                {
                    "line_number": idx + 1,
                    "error_type": matched_keywords[0],
                    "keywords": matched_keywords,
                    "bm25_score": float(scores[idx]) if scores is not None and len(scores) > idx else 0.0,
                    "message": message,
                    "context_lines": context_window,
                }
            )

        return {"step_name": step_name, "relevant_failures": relevant_failures}

    def _retrieve_relevant_files(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        documents, file_paths = [], []
        for root, _, files in os.walk(self.repo_path):
            for f in files:
                if f.endswith(".py"):
                    path = os.path.join(root, f)
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                            documents.append(fp.read())
                            file_paths.append(path)
                    except Exception:
                        continue

        if not documents or not query_text.strip():
            return []

        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(query_text.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [
            {
                "file": file_paths[i],
                "score": float(scores[i]),
                "reason": f"Matched tokens from error context (score={scores[i]:.2f})",
            }
            for i in top_indices
            if scores[i] > 0.0
        ]

    def ci_log_analysis(self) -> List[Dict[str, Any]]:
        print("Running Tool: BM25 Log + File Retrieval Analyzer")
        results: List[Dict[str, Any]] = []

        for step in self.ci_log:
            step_name = step.get("step_name", "unknown_step")
            log = self.flatten_log(step.get("log", ""))
            total_tokens = self._estimate_tokens(log)

            if total_tokens > CHUNK_SIZE_LOG:
                raw_chunks = chunk_log_by_tokens(
                    log, max_tokens=CHUNK_SIZE_LOG, overlap=200, model=self.model_name
                )
                print(f"Chunking activated: {len(raw_chunks)} raw chunks for '{step_name}'")
                chunks = self.filter_chunks(raw_chunks)
                print(f"Processing Step: {step_name}, {len(chunks)} filtered chunk(s)")
            else:
                chunks = [log]
                print(f"Processing Step: {step_name}, 1 chunk (no chunking)")

            all_failures: List[Dict[str, Any]] = []
            for chunk_text in chunks:
                ctx = self._extract_relevant_context(chunk_text, step_name)
                all_failures.extend(ctx["relevant_failures"])

            combined_query = " ".join(
                f.get("message", "") + " " + " ".join(f.get("context_lines", []))
                for f in all_failures
            ).strip()

            relevant_files = (
                self._retrieve_relevant_files(combined_query) if combined_query else []
            )

            results.append(
                {
                    "step_name": step_name,
                    "relevant_failures": all_failures,
                    "relevant_files": relevant_files,
                }
            )

        return results

    # ------------------------------------------------------------------
    def _generate_summary(self, log_details: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a structured final error summary from error details
        and workflow definition.
        """
        print(" Running Tool: _generate_summary")
        workflow_details = self.workflow

        # FIRST PASS: per-step LLM explanations on BM25 hits
        processed_log_details = self.process_log_details(log_details)

        # Estimate size using PROCESSED details
        log_json = json.dumps(processed_log_details, indent=2, ensure_ascii=False)
        workflow_json = json.dumps(workflow_details, indent=2, ensure_ascii=False)
        total_tokens = self._estimate_tokens(log_json) + self._estimate_tokens(workflow_json)

        try:
            if total_tokens <= MAX_TOKENS_SUMMARY:
                summary_input = processed_log_details
            else:
                # SECOND PASS: summarize processed details
                summary_input = self._summarize_log_details_if_large(
                    processed_log_details, workflow_details
                )

            summary = self.full_content_summary(summary_input, workflow_details)
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

    def _summarize_log_details_if_large(
        self, log_details: List[Dict[str, Any]], workflow_details: Any
    ) -> List[Dict[str, Any]]:
        """
        Second-level summarization when processed_log_details is still too large.
        Works on processed entries (from process_log_details).
        """
        CHUNK_SIZE = 50_000  # characters per chunk
        MAX_TOKENS = 50_000

        log_chunk_summaries: List[Dict[str, Any]] = []

        try:
            workflow_text = json.dumps(workflow_details, indent=2, ensure_ascii=False)
        except TypeError:
            workflow_text = str(workflow_details)

        entries = log_details if isinstance(log_details, list) else [log_details]

        for entry_idx, entry in enumerate(entries):
            if isinstance(entry, dict):
                step_name = entry.get("step_name", "unknown_step")
                relevant_failures = entry.get("relevant_failures", [])
                relevant_files = entry.get("relevant_files", [])
            else:
                step_name = "unknown_step"
                relevant_failures = []
                relevant_files = []

            failures_text = json.dumps(relevant_failures, indent=2, ensure_ascii=False)

            try:
                token_count = self._estimate_tokens(failures_text)
            except Exception:
                token_count = len(failures_text)

            if token_count > MAX_TOKENS or len(failures_text) > CHUNK_SIZE:
                log_chunks = [
                    failures_text[i : i + CHUNK_SIZE]
                    for i in range(0, len(failures_text), CHUNK_SIZE)
                ]
            else:
                log_chunks = [failures_text]

            print(
                f"[CHUNK-SUMMARY] Step '{step_name}' entry_index={entry_idx} "
                f"→ {len(log_chunks)} chunk(s) for LLM summarization."
            )

            step_name_json = json.dumps(step_name)

            for idx, chunk in enumerate(log_chunks):
                prompt = f"""
You are a CI failure summarization agent for a single CI log CHUNK.

INPUT:
- sha_fail: {self.sha_fail}
- step_name: {step_name}
- pre-processed chunk of this step's explanations (JSON/text) in `LOG_DETAILS` below
- BM25 candidate files for this step in `CANDIDATE_FILES` below
- CI workflow context in `WORKFLOW` below

Your job for THIS CHUNK ONLY:
1. Summarize, in English, what this chunk of explanations says about the failure.
2. Identify only those Python files that have clear evidence of being tied to the CI failure
   in this chunk, based on file paths mentioned in the explanations.

You must output STRICT JSON with this schema (do not add/remove keys):

{{
  "step_name": {step_name_json},
  "chunk_index": {idx},
  "sha_fail": "{self.sha_fail}",
  "error_context": [
    "English explanation(s) of the root cause(s) visible in THIS CHUNK, supported by evidence from LOG_DETAILS."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "reason": "Short explanation of why this file is tied to the failure in THIS CHUNK, quoting or paraphrasing the explanations or log snippets."
    }}
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Test Failure', 'Runtime Error', 'Dependency Error', 'Configuration Error', 'Code Formatting', 'Type Checking'",
      "subcategory": "More specific description, e.g. 'Code Formatting – ruff I001 unsorted imports', 'Test Failure – AssertionError in unit test', 'Dependency Error – missing package x'",
      "evidence": "Short quote or paraphrase from THIS CHUNK that justifies this classification."
    }}
  ]
}}

RULES:
- Use only information present in LOG_DETAILS / CANDIDATE_FILES / WORKFLOW.
- Only include a file in `relevant_files` if this chunk clearly ties it to the failure (tracebacks, failing tests, lint errors, etc.).
- If no file is clearly tied in this chunk, use an empty list for `relevant_files`.
- Use null for unknown numeric/line values.
- Return ONLY valid JSON, no markdown, no extra text.
- All string values must be single-line (no embedded line breaks).
- Do not paste large raw log blocks; summarize them in your own words including all information.
- If you need to mention that logs span multiple lines, just say e.g. "the log shows a stack trace mentioning file X and function Y" rather than copying it.

--- LOG_DETAILS (pre-processed explanations for this chunk) ---
{chunk}

--- CANDIDATE_FILES (BM25 candidates for this step) ---
{json.dumps(relevant_files, indent=2, ensure_ascii=False)}

--- WORKFLOW (CI workflow context) ---
{workflow_text}
""".strip()

                try:
                    t0 = time.time()
                    response = self.llm.invoke([HumanMessage(content=prompt)]).content
                    elapsed = time.time() - t0
                    print(
                        f"[LLM] _summarize_log_details_if_large: step='{step_name}' "
                        f"chunk {idx+1}/{len(log_chunks)} took {elapsed:.2f}s"
                    )
                    try:
                        summary = json.loads(response)
                    except json.JSONDecodeError:
                        summary = demjson3.decode(response)

                    log_chunk_summaries.append(summary)
                except Exception as e:
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
        """
        First-level summarization: per-step explanation of BM25-selected failures.
        Turns raw BM25 hits into more compact natural-language explanations.
        """
        processed_steps: List[Dict[str, Any]] = []

        for entry_idx, entry in enumerate(log_details):
            step_name = entry.get("step_name", "unknown_step")
            relevant_failures = entry.get("relevant_failures", [])
            relevant_files = entry.get("relevant_files", [])
            if not relevant_failures:
                continue

            failures_json_str = json.dumps(relevant_failures, indent=2)
            total_tokens = self._estimate_tokens(failures_json_str)

            if total_tokens > 80_000:
                failure_chunks = chunk_log_by_tokens(
                    failures_json_str,
                    max_tokens=80_000,
                    model=self.model_name,
                )
            else:
                failure_chunks = [failures_json_str]

            print(
                f"[PROCESS-LOG-DETAILS] Step '{step_name}' has "
                f"{len(failure_chunks)} failure chunk(s) for LLM explanation."
            )

            combined_failures: List[Dict[str, Any]] = []

            for chunk_idx, failures_chunk in enumerate(failure_chunks):
                prompt = f"""
You are a CI log explanation assistant.

You receive **one CI step** and a subset of its detected failures (as JSON text in `subset_of_relevant_failures`).

Your job for THIS SUBSET ONLY:
1. Produce a short high-level explanation `error_context` of what these failures mean for this step.
2. Produce ONE combined, detailed narrative `relevant_failures` that walks through ALL failures in this subset.

INPUT:
- step_name: {step_name}
- subset_of_relevant_failures (JSON list of dicts):
{failures_chunk}

Each failure dict may contain:
- line_number
- error_type
- keywords
- bm25_score
- message
- context_lines (raw CI log lines)

- Read every failure in this subset.
- Build a continuous narrative that:
  - Mentions each failure (line_number, error_type, message, keywords, bm25_score).
  - Summarizes what the context_lines show (files involved, error messages, tests, commands, etc.) IN YOUR OWN WORDS.
  - DO NOT copy raw context_lines or large log blocks. Only paraphrase them in natural language.
  - Keep the text reasonably short and compact.

OUTPUT (STRICT JSON):

{
  "step_name": "<same as input>",
  "error_context": "Single-line, high-level explanation for THIS CHUNK ONLY (subset of failures), summarizing what went wrong, which files and commands were involved, and how the keywords and bm25_score values support this. DO NOT include line breaks inside this string.",
  "relevant_failures": "Single-line narrative for ALL failures in this subset. This string must include, for every failure, its line_number, error_type, message, keywords, bm25_score, and a natural-language summary of its context_lines. DO NOT paste raw logs or multi-line snippets. DO NOT include unescaped line breaks; keep this as one line of text."
}


RULES:
- Return ONLY valid JSON (no markdown, no comments, no extra text).
- error_context and relevant_failures must each be a SINGLE-LINE STRING:
  - Do not put line breaks inside the quotes.
  - Use sentences separated by period or semicolon instead of newlines.
- Never copy raw JSON or raw CI log blocks into the string values. Always paraphrase them in your own words.
- Do not omit any distinct failure, but you may summarize log details concisely
""".strip()

                try:
                    t0 = time.time()
                    raw_response = self.llm.invoke(
                        [HumanMessage(content=prompt)]
                    ).content.strip()
                    elapsed = time.time() - t0

                    print(
                        f"[LLM] process_log_details: step='{step_name}' "
                        f"chunk {chunk_idx+1}/{len(failure_chunks)} took {elapsed:.2f}s"
                    )
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
                    "relevant_files": relevant_files,
                }
            )

        return processed_steps

    # ------------------------------------------------------------------
    def full_content_summary(
        self, log_details: List[Dict[str, Any]], workflow_details: Any
    ) -> Dict[str, Any]:
        """
        Final aggregation: combine per-step summaries + workflow into one
        global CI failure summary.
        """
        log_details_text = json.dumps(log_details, indent=2, ensure_ascii=False)
        workflow_text = json.dumps(workflow_details, indent=2, ensure_ascii=False)

        prompt = f"""
You are a CI failure summarization agent.

Your task:
Read CI job logs (already pre-processed into summaries) and workflow details,
then produce a structured, evidence-based JSON summary that explains:
- what failed,
- which files are actually involved in the failure,
- how to classify the error(s) by category and subcategory,
- which CI job/step/command failed.

## INPUTS:
1. sha_fail: {self.sha_fail}
2. log_details (JSON):
{log_details_text}

3. workflow_details (JSON):
{workflow_text}

## OUTPUT FORMAT (STRICT JSON):

{{
  "sha_fail": "{self.sha_fail}",
  "error_context": [
    "Plain-English explanation(s) of the root cause(s), supported by evidence from log_details. Summarize what failed, why it failed, and which step(s)/tool(s) are responsible."
  ],
  "relevant_files": [
    {{
      "file": "path/to/file.py",
      "line_number": 123,
      "reason": "Short evidence-based explanation of why this file is tied to the failure, quoting or paraphrasing the relevant summaries/log text."
    }}
  ],
  "error_types": [
    {{
      "category": "High-level category, e.g. 'Code Formatting', 'Dependency Error', 'Test Failure', 'Runtime Error', 'Type Checking', 'Configuration Error'",
      "subcategory": "More specific type under that category, e.g. 'Unused Import', 'Line Length Exceeded', 'ImportError: No module named X', 'AssertionError in unit test', 'Mypy type mismatch', 'Missing dependency x'",
      "evidence": "Brief quote or paraphrase from the summaries/log text that proves this classification."
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

## GUIDANCE:
- Derive everything from log_details + workflow_details; do not invent data.
- Extract file paths and line numbers from the text where possible; use null if unknown.
- Do NOT copy long raw CI logs or workflow files directly into any string.
- All string values must be single-line (no embedded line breaks). Use ". " or "; " to separate sentences, not newline characters.
- If job/step/command cannot be determined, use null for those fields.
- Return ONLY valid JSON, no markdown or extra text.
""".strip()

        try:
            t0 = time.time()
            response = self.llm.invoke([HumanMessage(content=prompt)]).content
            elapsed = time.time() - t0
            print(
                f"[LLM] full_content_summary: sha_fail={self.sha_fail} "
                f"took {elapsed:.2f}s"
            )
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
        if generated_summary.get("sha_fail") in (None, "", "null", "unknown"):
            generated_summary["sha_fail"] = self.sha_fail
        return generated_summary

    # ------------------------------------------------------------------
    def _get_encoder(self):
        """Safely get a tiktoken encoder for the model."""
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def _estimate_tokens(self, text: str) -> int:
        """Estimate tokens for a given text using the cached encoder."""
        if text is None:
            return 0
        try:
            return len(self._encoder.encode(text))
        except Exception:
            return len(text)

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
