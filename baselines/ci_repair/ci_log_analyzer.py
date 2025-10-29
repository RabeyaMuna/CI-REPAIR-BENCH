from typing import List, Dict, Any, TypedDict
import json
import os
import sys
import time
import demjson3
import tiktoken
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END

from utilities.load_config import load_config
from utilities.chunking_logic import chunk_log_by_tokens

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure it's in your .env file.")

class CILogAnalyzer:
    def __init__(self, repo_path: str, ci_log: List[Dict[str, Any]], sha_fail: str, workflow: str, workflow_path: str):
        self.config = load_config()
        self.ci_log = ci_log
        self.sha_fail = sha_fail
        self.workflow = workflow
        self.workflow_path = workflow_path
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.all_step_outputs = []
        self.error_details = []
        self.summary = {}
        self.repo_path = repo_path
        self.workflow_tools_detail = None
        self.workflow_validation_checks = None
    

    def ci_log_analysis(self) -> List[Dict[str, Any]]:
        """Extract all CI step, error contexts, chunk logs if necessary, and capture summaries, relevant files, and error blocks."""
        
        print("Running Tool: process_ci_steps_to_extract_error_contexts")
        results = []
        THRESHOLD = 100_000

        for step in self.ci_log:
            step_name = step.get("step_name")
            log = step.get("log")
            print(f"\nProcessing Step: {step_name}")

            try:
                log_text = log if isinstance(log, str) else "\n".join(log)
                enc = tiktoken.encoding_for_model("gpt-4o-mini")
                total_tokens = len(enc.encode(log_text))

                print(f"Token count for '{step_name}': {total_tokens}")

                if total_tokens > THRESHOLD:
                    chunks = chunk_log_by_tokens(log_text, max_tokens=60000, overlap=200)
                    print(f"Chunking activated: {len(chunks)} chunks created for step '{step_name}'")
                else:
                    chunks = [log_text]
                    print(f"No chunking needed for '{step_name}'")

                step_chunks = []

                for i, chunk in enumerate(chunks):
                    print(f"Processing chunk {i + 1}/{len(chunks)}...")

                    prompt = f"""
Analyze the following CI log chunk and extract comprehensive information with a focus on capturing all error context.

## INSTRUCTIONS:
1. Create an extremely detailed natural language summary that includes EVERY piece of information from the log chunk
   - Mention ALL commands, operations, and outcomes
   - Mention ALL test results
   - Mention ALL files (with repo-relative paths) and why they appear
   - Mention ALL warnings and errors
2. Extract ALL file paths mentioned in the log (normalized to repo-relative format)
3. Identify ALL failures, errors, and issues with their complete context
4. Extract the exact error blocks with their complete surrounding context

## OUTPUT FORMAT:
Return ONLY valid JSON with this exact structure:

{{
  "step_name": "{step_name}",
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

    

    def _generate_summary(self, log_details) -> Dict:
        """Generate a structured final error summary from error details, workflow tools, and validation checks."""
        print(" Running Tool: _generate_summary")
        
        log_details = log_details
        workflow_details = self.workflow

        prompt = f"""
You are a CI failure summarizer agent.

Your goal is to read CI job logs and workflow context, then return a clean, structured summary with:
- concise error_context (plain English reasons with evidence),
- relevant_files tied to the failure,
- error_types that YOU derive from the evidence (no pre-given taxonomy),
- failed_job mapping (job/step/command) linked to the failure.

---
## Inputs

1. CI Log Details from step analysis:
{json.dumps(log_details, indent=2, ensure_ascii=False)}

2. Workflow Details:
{json.dumps(workflow_details, indent=2, ensure_ascii=False)}
---

## Category Instructions (derive, don't assume)
- Do NOT use any pre-defined category list.
- Infer clear, data-driven categories directly from the logs and failed commit evidence.
- Each item in error_types must be concise and specific, e.g.:
  - "ImportError: No module named 'X'"
  - "AssertionError: expected 2 == 3 in tests/test_api.py::test_y"
  - "Black formatting check failed (line length > 88)"
  - "Type checking (mypy): Incompatible return type in foo.py:123"
  - "Pytest collection error: syntax error in bar.py:45"
  - "GitHub Actions: step 'Run tests' exited with code 2"
- If you cannot justify a category with log evidence, do not include it.
- Deduplicate near-duplicates (merge similar items).

### Instructions:

1. RELEVANT FILES (unique):
   - List only files directly tied to the failure.
   - For each: include "file", "line_number" (integer or null), and "reason" (quote or paraphrase log evidence).

2. ERROR CONTEXT (plain English):
   - Summarize root cause(s) in short sentences.
   - Include exception type, file + line (if available), exact error message, and a brief explanation.

3. JOBS RELEVANT TO ERRORS:
   - For each failed step: return
     {{
       "job": "<job name or ID>",
       "step": "<step name>",
       "command": "<command or action run>"
     }}

4. ERROR TYPES (derived):
   - Produce a list of concise, evidence-backed labels as described above.
   - No generic buckets; be concrete and tied to the log.
   - Include both primary failure(s) and any clearly validated checks that passed/failed if the evidence is explicit.

### Final output JSON format (strict):
{{
  "sha_fail": "{self.sha_fail}",
  "error_context": ["..."],
  "relevant_files": [
    {{
      "file": "...",
      "line_number": null,
      "reason": "Provided error or workflow failure reason from the CI log"
    }}
  ],
  "error_types": ["..."],
  "failed_job": [
    {{
      "job": "...",
      "step": "...",
      "command": "..."
    }}
  ]
}}

## CRITICAL RULES:
- Return ONLY raw JSON in the above format—no text before or after.
- Do NOT include code fences.
- Do NOT add any extra keys, commentary, or filler text.
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
