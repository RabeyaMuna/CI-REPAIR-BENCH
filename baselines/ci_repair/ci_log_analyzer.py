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


class CILogAnalyzer:
    def __init__(self, repo_path: str, ci_log: List[Dict[str, Any]], sha_fail: str, workflow: str, workflow_path: str):
        self.config = load_config()
        self.repo_path = repo_path
        self.ci_log = ci_log
        self.sha_fail = sha_fail
        self.workflow = workflow
        self.workflow_path = workflow_path
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.error_details = []

        self.error_keywords = [
            "Traceback", "Exception", "AssertionError", "ImportError", "ModuleNotFoundError",
            "TypeError", "ValueError", "AttributeError", "RuntimeError", "IndexError", "KeyError",
            "SyntaxError", "IndentationError", "NameError", "FileNotFoundError",
            "pytest", "E ", "FAILED", "FAIL", "error", "fail", "exit code 1",
            "black", "ruff", "flake8", "mypy", "coverage", "unittest"
        ]

    # ----------------------------------------------------------------------
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
            i for i, score in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            if score > 0.0
        ][:BM25_TOP_N]

        relevant_failures = []
        seen = set()
        for idx in top_indices:
            line_text = lines[idx].strip()
            if not line_text or line_text in seen:
                continue
            seen.add(line_text)

            error_type = next((kw for kw in self.error_keywords if kw.lower() in line_text.lower()), "UnknownError")
            start = max(0, idx - CONTEXT_LINES)
            end = min(len(lines), idx + CONTEXT_LINES + 1)
            context_window = [lines[i] for i in range(start, end) if lines[i].strip()]

            if "Error" in line_text or "Exception" in line_text:
                parts = line_text.split(":", 1)
                message = parts[1].strip() if len(parts) > 1 else line_text
            else:
                message = line_text

            relevant_failures.append({
                "line_number": idx + 1,
                "error_type": error_type,
                "message": message,
                "context_lines": context_window
            })

        return {"step_name": step_name, "relevant_failures": relevant_failures}

    # ----------------------------------------------------------------------
    def _retrieve_relevant_files(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Use BM25 to retrieve the most relevant Python files in the repo
        given the combined error context text.
        """
        documents = []
        file_paths = []

        # Walk through repo and collect all Python source files
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
                        continue

        if not documents:
            return []

        tokenized_docs = [doc.split() for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = query_text.split()
        scores = bm25.get_scores(tokenized_query)

        # Sort and select top files
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        ranked_files = [
            {
                "file": file_paths[i],
                "score": float(scores[i]),
                "reason": f"Matched tokens from error context (score={scores[i]:.2f})"
            }
            for i in top_indices if scores[i] > 0.0
        ]

        return ranked_files

    # ----------------------------------------------------------------------
    def ci_log_analysis(self) -> List[Dict[str, Any]]:
        """
        Analyze CI logs using BM25 (for both error context + file retrieval).
        """
        print("Running Tool: BM25 Log + File Retrieval Analyzer")
        results = []
        THRESHOLD = 100_000

        for step in self.ci_log:
            step_name = step.get("step_name", "unknown_step")
            log = step.get("log", "")
            print(f"\nProcessing Step: {step_name}")

            try:
                log_lines = log if isinstance(log, list) else log.splitlines()
                log_text = "\n".join(log_lines)
                enc = tiktoken.encoding_for_model("gpt-4o-mini")
                total_tokens = len(enc.encode(log_text))

                if total_tokens > THRESHOLD:
                    chunks = chunk_log_by_tokens(log_text, max_tokens=80000, overlap=200)
                    print(f"Chunking activated: {len(chunks)} chunks for '{step_name}'")
                else:
                    chunks = [log_text]

                all_failures = []
                for chunk_text in chunks:
                    details = self._extract_relevant_context(chunk_text, step_name)
                    all_failures.extend(details["relevant_failures"])

                # Combine all error lines to create the file retrieval query
                combined_query = " ".join(f["message"] for f in all_failures if f["message"])

                relevant_files = self._retrieve_relevant_files(combined_query) if combined_query else []

                results.append({
                    "step_name": step_name,
                    "relevant_failures": all_failures,
                    "relevant_files": relevant_files
                })

            except Exception as e:
                results.append({
                    "step_name": step_name,
                    "error": str(e),
                    "relevant_failures": [],
                    "relevant_files": []
                })

        return results

    def _generate_summary(self, log_details) -> Dict:
        """
        Generate structured CI error summary with adaptive chunking, cumulative context,
        and a final LLM synthesis step to merge all chunk summaries (no truncation, no ```json fences).
        """
        print(" Running Tool: _generate_summary")

        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        MAX_CONTEXT = 120_000   # safe upper bound for GPT-4o-mini
        CHUNK_LIMIT = 80_000    # when to start chunking
        CHUNK_SIZE = 60_000     # approximate per-chunk size

        workflow_details = self.workflow
        log_json = json.dumps(log_details, indent=2, ensure_ascii=False)
        workflow_json = json.dumps(workflow_details, indent=2, ensure_ascii=False)

        log_tokens = len(enc.encode(log_json))
        workflow_tokens = len(enc.encode(workflow_json))
        combined_tokens = log_tokens + workflow_tokens
        print(f"  [INFO] log_tokens={log_tokens}, workflow_tokens={workflow_tokens}, combined={combined_tokens}")

        # =====================
        # Step 1: Chunk logs and workflow if needed
        # =====================
        def make_chunks(text, limit=CHUNK_LIMIT, chunk_size=CHUNK_SIZE):
            """Split long JSON text by line into smaller token-safe chunks."""
            tokens = len(enc.encode(text))
            if tokens <= limit:
                return [text]

            print(f"  [WARN] Splitting content ({tokens} tokens) into chunks of ~{chunk_size} tokens.")
            lines = text.splitlines()
            chunks, current, counter = [], [], 0
            for line in lines:
                current.append(line)
                counter += len(enc.encode(line))
                if counter >= chunk_size:
                    chunks.append("\n".join(current))
                    current, counter = [], 0
            if current:
                chunks.append("\n".join(current))
            return chunks

        log_chunks = make_chunks(log_json)
        workflow_chunks = make_chunks(workflow_json)
        total_parts = max(len(workflow_chunks), len(log_chunks))

        # =====================
        # Step 2: Analyze each chunk with cumulative context
        # =====================
        summaries = []
        exception_dir = os.path.join(self.config["exception_dir"], "chunk_summaries")
        os.makedirs(exception_dir, exist_ok=True)

        OUTPUT_RULES = """
    STRICT OUTPUT RULES:
    - Return **raw JSON only**.
    - Do NOT include any backticks or code fences (no ```json or ```).
    - Do NOT include markdown or explanations.
    - The first character must be '{' and the last must be '}'.
    """

        def _safe_json_parse(text: str):
            """Remove stray markdown and parse JSON safely."""
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                return demjson3.decode(cleaned)

        for i in range(total_parts):
            workflow_part = workflow_chunks[i] if i < len(workflow_chunks) else workflow_chunks[-1]
            log_part = log_chunks[i] if i < len(log_chunks) else log_chunks[-1]

            print(f"  [INFO] Processing chunk {i + 1}/{total_parts}")

            previous_results_text = (
                json.dumps(summaries, indent=2, ensure_ascii=False) if summaries else "None"
            )

            prompt = f"""
    You are a CI failure summarization agent analyzing part {i + 1} of {total_parts}.

    ## CONTEXT
    Earlier chunk summaries are provided below to ensure continuity:
    {previous_results_text}

    ---

    ### CI Log Details (Part {i + 1}/{total_parts})
    {log_part}

    ### Workflow Details (Part {i + 1}/{total_parts})
    {workflow_part}

    ---

    ### TASK
    Summarize this part’s errors into a structured JSON object, combining prior insights if applicable.

    {OUTPUT_RULES}

    ### OUTPUT FORMAT (strict JSON)
    {{
    "sha_fail": "{self.sha_fail}",
    "part": {i + 1},
    "error_context": ["Concise root cause(s)."],
    "relevant_files": [
        {{
        "file": "path/to/file.py",
        "line_number": 123,
        "reason": "Why it's related to the failure."
        }}
    ],
    "error_types": [
        {{
        "category": "High-level (e.g., Test Failure, Dependency Error, Runtime Error)",
        "subcategory": "Specific subtype (e.g., AssertionError, ImportError)",
        "evidence": "Short quote or paraphrase from logs."
        }}
    ],
    "failed_job": [
        {{
        "job": "Job name",
        "step": "Step name that failed",
        "command": "Command/tool that caused failure"
        }}
    ]
    }}
    """

            try:
                response = self.llm.invoke([HumanMessage(content=prompt)]).content
                summary = _safe_json_parse(response)
                summaries.append(summary)

                # Save each chunk’s result
                part_path = os.path.join(exception_dir, f"{self.sha_fail}_part{i + 1}.json")
                with open(part_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[ERROR] Failure in chunk {i + 1}: {e}")
                err_obj = {"part": i + 1, "error": str(e)}
                summaries.append(err_obj)
                with open(
                    os.path.join(exception_dir, f"{self.sha_fail}_part{i + 1}_error.json"),
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(err_obj, f, indent=2)

        # =====================
        # Step 3: Final synthesis LLM pass (no truncation, no chunking)
        # =====================
        print("  [INFO] Running final LLM synthesis across all chunk summaries.")
        summaries_json = json.dumps(summaries, indent=2, ensure_ascii=False)

        synthesis_prompt = f"""
    You are an advanced CI repair analyst.

    Below are structured JSON summaries of all analyzed log/workflow chunks for commit `{self.sha_fail}`:

    {summaries_json}

    {OUTPUT_RULES}

    ### TASK
    Integrate all partial summaries into **one coherent, deduplicated final summary**.

    ### REQUIREMENTS
    - Merge overlapping errors and evidence.
    - Combine related error_context items.
    - Deduplicate relevant_files and error_types.
    - Produce ONE concise, clean JSON.

    ### OUTPUT FORMAT (strict JSON)
    {{
    "sha_fail": "{self.sha_fail}",
    "error_context": ["Merged plain-English root causes."],
    "relevant_files": [
        {{
        "file": "path/to/file.py",
        "line_number": 123,
        "reason": "Why this file is tied to the error."
        }}
    ],
    "error_types": [
        {{
        "category": "Error category",
        "subcategory": "Error subtype",
        "evidence": "Representative log snippet."
        }}
    ],
    "failed_job": [
        {{
        "job": "Job name",
        "step": "Step name",
        "command": "Command that failed"
        }}
    ]
    }}
    """

        try:
            synthesis_response = self.llm.invoke([HumanMessage(content=synthesis_prompt)]).content
            final_summary = _safe_json_parse(synthesis_response)
        except Exception as e:
            print(f"[ERROR] Final synthesis step failed: {e}")
            final_summary = {
                "error": str(e),
                "sha_fail": self.sha_fail,
                "note": "Failed during final synthesis"
            }

        # Save final synthesis output
        final_path = os.path.join(exception_dir, f"{self.sha_fail}_final_summary.json")
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2, ensure_ascii=False)

        print(f" Completed: _generate_summary (multi-step LLM synthesis). Final summary saved to: {final_path}")

        return final_summary


       

    def run(self) -> Dict[str, Any]:
        print(f"Fully Autonomous Execution for Commit: {self.sha_fail}")
        log_details = self.ci_log_analysis()
        generated_summary = self._generate_summary(log_details)
        
        return generated_summary
        
        
        
    def _log_error(self, method: str, error: Exception, step: str = ""):
        base_dir = os.path.join(self.config["exception_dir"], "interrupted_error_log")
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"{self.sha_fail}_{method}_{int(time.time())}_error.json"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "commit": self.sha_fail,
                "method": method,
                "step": step,
                "error": str(error)
            }, f, indent=2)
        print(f"[ERROR LOGGED] {file_path}")
