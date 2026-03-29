"""
Pytest configuration and shared fixtures.

The sys.modules mocks at the top MUST come before any project imports.
validator.py has module-level LLM initialization (`llm = OpenAIClient().get_llm()`
and `prompt_registry.load_prompts()`). Mocking these modules here prevents
those calls from failing during unit tests that have no API keys or DB access.
"""
import sys
import pytest
from unittest.mock import MagicMock

# --- Mock LLM-dependent modules before any test file triggers their import ---
sys.modules["llm"] = MagicMock()
sys.modules["llm.openai_client"] = MagicMock()
sys.modules["llm.groq_client"] = MagicMock()
sys.modules["core.prompt_registry"] = MagicMock(prompt_registry=MagicMock())


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "behavioral: LLM-backed integration tests (require API keys, excluded from fast CI)"
    )
    config.addinivalue_line(
        "markers", "llm_eval: LLM judge evaluation tests (require OPENAI_API_KEY, non-deterministic scores)"
    )
