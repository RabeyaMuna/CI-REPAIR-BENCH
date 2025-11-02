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
        """Generate a structured final error summary from error details, workflow tools, and validation checks."""
        print(" Running Tool: _generate_summary")
        
        log_details = log_details
        workflow_details = self.workflow

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
   - Each subcategory must be concrete and justifiable (e.g. “ImportError: No module named taipy.gui.servers.fastapi” → category: “Dependency Error”, subcategory: “Missing Module”).

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
            error_dir = os.path.join(self.config["exception_dir"], "interrupted_error_log")

            os.makedirs(error_dir, exist_ok=True)

            error_data = {
                "sha_fail": self.sha_fail,
                "error": str(e),
                "tool": "ErrorContextExtractionAgent.run"
            }

            error_file = os.path.join(error_dir, f"{self.sha_fail}_error.json")
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_data, f, indent=4)

            return {"error": f"Failed to generate summary: {e}"}
       

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
