"""Shared LLM used by the orchestrator."""
from agents.llm_config import get_llm

router_llm = get_llm(temperature=0, max_tokens=1024)
