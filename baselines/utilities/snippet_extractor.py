import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

def extract_snippet_from_line_range(
    original_file_content: str,
    line_range: Optional[Tuple[int, int]],
) -> str:
    """
    Extracts the code snippet from the given line range and attaches it to the fault.

    Args:
        original_file_content (str): Full text of the source file.
        line_range (tuple[int, int]): The start and end line numbers (1-based).
        fault (dict): Fault dictionary containing at least 'line_range'.
        fault_locations (list[dict]): List to append processed fault results to.

    Returns:
        None. (The function updates fault_locations in-place.)
    """
    if not line_range:
        logger.debug("No line_range provided, skipping fault extraction.")
        return

    start, end = line_range
    lines = original_file_content.splitlines()

    # Validate range
    if not (1 <= start <= end <= len(lines)):
        logger.warning(f"Invalid line range {line_range} for file with {len(lines)} lines.")
        return

    # Extract snippet
    snippet = "\n".join(lines[start - 1:end])
    
    return snippet

def _process_fault_and_extract_snippet(
    self,
    fault: dict,
    original_file_content: str,
    valid_start: int,
    valid_end: int,
    chunk_idx: int,
    fault_locations: list,
) -> dict:
    """
    Handles snippet extraction and line range detection for a single fault entry.
    Returns the processed fault (with code_snippet and line_range updated if found).
    """

    snippet = fault.get("code_snippet")
    if not snippet:
        return fault  # No snippet provided, return as-is

    # Handle list-based snippets
    if isinstance(snippet, list):
        snippet = "\n".join(snippet)

    # Try to find line range from snippet
    line_range = self.find_line_range(original_file_content, snippet)

    if line_range is not None:
        start, end = line_range

        if valid_start <= start and end <= valid_end:
            # Normal case: attach found range
            fault["line_range"] = list(line_range)
            fault["code_snippet"] = snippet
            fault_locations.append(fault)

            print("\n--- Fault Detected ---")
            print("Code Snippet:\n", snippet)
            print("Line Range:", line_range)
            print("---------------------\n")

        else:
            print(f"[Chunk {chunk_idx+1}] Skipping fault outside chunk range {valid_start}-{valid_end}: {line_range}")

    else:
        # Fallback if the fault already includes a range
        fallback_range = fault.get("line_range")
        if fallback_range:
            start, end = int(fallback_range[0]), int(fallback_range[1])
            lines = original_file_content.splitlines()

            if 1 <= start <= end <= len(lines) and valid_start <= start and end <= valid_end:
                extracted_snippet = "\n".join(lines[start - 1:end])
                fault["code_snippet"] = extracted_snippet
                fault["line_range"] = [start, end]
                fault_locations.append(fault)

                print("\n--- Fault Detected (by range) ---")
                print("Code Snippet:\n", extracted_snippet)
                print("Line Range:", (start, end))
                print("---------------------\n")

    return fault


def find_line_range(file_content: str, snippet: str) -> tuple[int, int] | None:
        """
        Finds the line range of a given code snippet in the full file content.
        
        Args:
            file_content (str): Full file content as a string.
            snippet (str): Multiline code snippet to locate.

        Returns:
            (start_line, end_line) if found, else None.
        """
        file_lines = file_content.splitlines()
        snippet_lines = snippet.strip("\n").splitlines()
        snippet_length = len(snippet_lines)

        for i in range(len(file_lines) - snippet_length + 1):
            window = file_lines[i:i + snippet_length]
            if window == snippet_lines:
                # Line numbers are 1-indexed (not 0-indexed)
                print("DEBUG: match found for snippet",[i + 1, i + snippet_length])
                return  (i + 1, i + snippet_length)     
        
        print("DEBUG: No match found for snippet")
        return None
    