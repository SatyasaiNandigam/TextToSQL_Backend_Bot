"""
Behavioral test conftest.

Removes the LLM/registry mocks added by the parent conftest so that
behavioral tests can import and use the real Groq client and prompt registry.
This runs after tests/conftest.py but before test files are imported.
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
