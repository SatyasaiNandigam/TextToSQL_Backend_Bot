"""
Integration test conftest.

Removes the LLM/registry mocks added by the parent conftest so that
integration tests can import schema_rag_pipeline and real DB modules.
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
