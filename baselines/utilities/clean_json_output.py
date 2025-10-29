import re

def clean_llm_json_output(raw: str) -> str:
    # Removes wrapping triple backticks and optional language hints
    cleaned = re.sub(r"^```(?:json)?", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
    return cleaned.strip()
