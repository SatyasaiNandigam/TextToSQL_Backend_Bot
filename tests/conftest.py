"""
Pytest configuration and shared fixtures.

The sys.modules mocks at the top MUST come before any project imports.
validator.py has module-level LLM initialization (`llm = OpenAIClient().get_llm()`
and `prompt_registry.load_prompts()`). Mocking these modules here prevents
those calls from failing during unit tests that have no API keys or DB access.
"""
import sys
from unittest.mock import MagicMock

# --- Mock LLM-dependent modules before any test file triggers their import ---
sys.modules["llm"] = MagicMock()
sys.modules["llm.openai_client"] = MagicMock()
sys.modules["llm.groq_client"] = MagicMock()
sys.modules["core.prompt_registry"] = MagicMock(prompt_registry=MagicMock())
