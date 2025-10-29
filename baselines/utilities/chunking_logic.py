from typing import List, Dict, Any
import json
import os
import time
import tiktoken
from tiktoken import encoding_for_model

def get_chunk_params(token_count):
    if token_count > 1_000_000:
        return 6000, 100
    elif token_count > 500_000:
        return 5000, 100
    elif token_count > 100_000:
        return 3000, 100
    else:
        return token_count, 0  # no chunking needed

def count_tokens(text: str, model: str = "gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_log_by_tokens(log_text: str, max_tokens: int = 60000, overlap: int = 100) -> List[str]:
    # Safe encoding that accepts all special tokens as plain text
    enc = encoding_for_model("gpt-4o-mini")

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


def chunk_log_safely(
    log_text: str,
    max_tokens: int = 90_000,
    max_chars: int = 100_000,
    overlap_lines: int = 50
) -> List[str]:
    enc = encoding_for_model("gpt-4o-mini")  # no special args here
    lines = log_text.splitlines()
    chunks = []
    current_chunk = []
    current_chars = 0

    for line in lines:
        line_length = len(line) + 1
        current_chars += line_length
        current_chunk.append(line)

        approx_token_count = len(enc.encode("\n".join(current_chunk), disallowed_special=()))  # FIXED
        
        if current_chars > max_chars or approx_token_count > max_tokens:
            chunks.append("\n".join(current_chunk[:-overlap_lines] if len(current_chunk) > overlap_lines else current_chunk))
            current_chunk = current_chunk[-overlap_lines:]
            current_chars = sum(len(l) + 1 for l in current_chunk)

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
