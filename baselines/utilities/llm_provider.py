# utilities/llm_provider.py

import os
from dataclasses import dataclass
from typing import Dict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # you must set this in .env
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "dummy-local-key")  # fallback


@dataclass
class LLMInfo:
    """Configuration for one logical LLM."""
    provider: str
    model_name: str
    temperature: float = 0.0
    base_url: str | None = None
    api_key_env: str | None = None  # which env var holds the key (if not OPENAI_API_KEY)
    api_key: str | None = None      # direct key value (optional)


# Here you list *all* models you want to use in your system
LLM_REGISTRY: Dict[str, LLMInfo] = {
    # OpenAI models
    "gpt-4o-mini": LLMInfo(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    "gpt4o": LLMInfo(
        provider="openai",
        model_name="gpt-4o",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    "gpt-4.1": LLMInfo(
        provider="openai",
        model_name="gpt-4.1",  # change to the exact model name you use
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    
    "gpt5.1": LLMInfo(
        provider="openai",
        model_name="gpt-5.1",   # exact API name
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),
    
    "gpt-5-mini": LLMInfo(
        provider="openai",
        model_name="gpt-5-mini",
        temperature=0.0,
        api_key=OPENAI_API_KEY,
    ),

    # DeepSeek via OpenAI-compatible API
    "DeepSeek-V3.2": LLMInfo(
        provider="deepseek",
        model_name="DeepSeek-V3.2", # Fixed missing comma
        temperature=0.0,
        base_url="https://api.deepseek.com/v1",
        api_key=DEEPSEEK_API_KEY,     # comes from DEEPSEEK_API_KEY in .env
    ),

    # Local Llama (OpenAI-compatible server)
    "local-llama": LLMInfo(
        provider="local",
        model_name="llama-3-8b-instruct",
        temperature=0.0,
        base_url="http://localhost:8000/v1",
        api_key=LOCAL_LLM_API_KEY,    # usually just some dummy/token for your local server
    ),
}


def get_llm(model_key: str) -> ChatOpenAI:
    """
    Return a ChatOpenAI-like instance for the given logical model_key.
    Same function is used everywhere in the system.
    """
    if model_key not in LLM_REGISTRY:
        raise ValueError(f"Unknown model_key: {model_key}")

    info = LLM_REGISTRY[model_key]

    # Decide which API key to use
    api_key = None
    if info.api_key_env:
        api_key = os.getenv(info.api_key_env)
    elif info.api_key:
        api_key = info.api_key
    elif info.provider == "openai":
        api_key = OPENAI_API_KEY

    kwargs = {
        "model": info.model_name,
        "temperature": info.temperature,
    }

    if api_key:
        kwargs["api_key"] = api_key

    if info.base_url:
        kwargs["base_url"] = info.base_url

    return ChatOpenAI(**kwargs)
