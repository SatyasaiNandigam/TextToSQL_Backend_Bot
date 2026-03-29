"""
Evaluation conftest.

Removes the LLM/registry sys.modules mocks added by tests/conftest.py
so that evaluation tests can import and invoke the real OpenAI client
and prompt registry. Follows the same pattern as tests/behavioral/conftest.py.

This file runs after tests/conftest.py but before test files are imported.
"""
import sys

_MOCKED_MODULES = [
    "llm",
    "llm.openai_client",
    "llm.groq_client",
    "core.prompt_registry",
]

for _mod in _MOCKED_MODULES:
    sys.modules.pop(_mod, None)
