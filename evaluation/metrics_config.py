"""
Ragas metric configuration for Level 2 context relevancy evaluation.

Uses gpt-4o-mini as the LLM judge (same model as the rest of the pipeline).
Metrics operate on retrieved doc.page_content chunks — NOT the DDL output.

Ragas version: 0.4.x
- Uses llm_factory + AsyncOpenAI (not LangchainLLMWrapper)
- Uses metric.ascore() per sample (not evaluate())
- EvaluationDataset / evaluate() are deprecated in this version
"""

import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference, ContextRecall


# Score thresholds from EVALUATION_PLAN.md
THRESHOLDS: dict[str, float] = {
    "context_precision": 0.80,
    "context_recall": 0.85,
}


def get_metrics() -> tuple:
    """Return (precision_metric, recall_metric) initialized with gpt-4o-mini.

    ContextPrecisionWithoutReference:
        Measures whether each retrieved context chunk is useful for answering
        the user question (noise detection). No reference answer required.

    ContextRecall:
        Measures whether all information needed to produce the reference answer
        is present in the retrieved contexts (miss detection).

    Returns a tuple so callers can use both metrics in one asyncio.gather() call.
    """
    judge_llm = llm_factory(
        "gpt-4o-mini",
        client=AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    )
    return (
        ContextPrecisionWithoutReference(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    )
