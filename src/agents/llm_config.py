"""Central LLM configuration for the Intelligent Hospital Assistant.

All hosted LLM calls use Groq through its OpenAI-compatible API.
The API key is loaded from GROQ_API_KEY and must never be committed.
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def get_llm(temperature: float = 0, max_tokens: int = 1024) -> ChatOpenAI:
    """Return the single shared Groq-backed LangChain chat model."""
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set it in your environment or .env file."
        )

    return ChatOpenAI(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# Fail early with a clear message when the application starts without a key.
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not configured. Set it in your environment or .env file."
    )
