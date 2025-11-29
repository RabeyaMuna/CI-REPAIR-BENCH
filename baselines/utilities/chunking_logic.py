from typing import List, Dict, Any
import json
import os
import time
import tiktoken
from tiktoken import encoding_for_model

def _get_encoder_from_model(model: str):
    name = (model or "").lower()

    # OpenAI GPT-family models: use the exact encoding if possible
    if name.startswith("gpt") or "gpt-" in name:
        try:
            return encoding_for_model(model)
        except Exception:
            # If tiktoken doesn't recognize the exact model, fall back
            pass

    # For "cl100k_base" explicitly, or for Claude/DeepSeek/etc. → generic BPE
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        # This should basically never happen, but just in case:
        raise RuntimeError(f"Failed to load cl100k_base encoding: {e}")

def count_tokens(text: str, model: str = "cl100k_base"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_log_by_tokens(log_text: str, max_tokens: int = 60000, overlap: int = 100, model: str = "cl100k_base") -> List[str]:
    # Safe encoding that accepts all special tokens as plain text
    enc = _get_encoder_from_model(model)

    lines = log_text.splitlines()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(enc.encode(line))

        if current_tokens + line_tokens > max_tokens:
            chunks.append("\n".join(current_chunk))

            # Start new chunk with overlap
            if overlap > 0:
                overlap_lines = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk
                current_chunk = list(overlap_lines)
                current_tokens = sum(len(enc.encode(l)) for l in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(line)
        current_tokens += line_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks



   
def chunk_lines_with_overlap(content: str, lines_per_chunk=200, overlap=20):
    lines = content.splitlines()
    total = len(lines)
    chunks = []
    i = 0

    while i < total:
        start = max(i - overlap, 0)
        end = min(i + lines_per_chunk, total)
        chunk = "\n".join(lines[start:end])
        chunks.append((start + 1, end, chunk))  # 1-based line numbers
        i += lines_per_chunk

    return chunks

def _estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))