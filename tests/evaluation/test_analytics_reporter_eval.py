"""
DeepEval GEval tests for the analytics_reporter (analytics_agent) node.

Run:
    pytest tests/evaluation/ -v -m llm_eval -k "analytics_reporter"
"""
import json
import pathlib
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage
from core.state import AgentState
from pipeline.nodes.analytics_reporter import analytics_agent

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"
_JUDGE_MODEL = "gpt-4o-mini"


def _load_dataset() -> list[dict]:
    with open(DATASETS_DIR / "analytics_reporter_eval.json", encoding="utf-8") as f:
        return json.load(f)


_ALL_CASES = _load_dataset()


def _make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["question"])],
        current_user_query=case["question"],
        resolved_query=case["question"],
        completed_steps=case["completed_steps"],
        raw_data=case["raw_data"],
    )


# ---------------------------------------------------------------------------
# GEval metric factories
# ---------------------------------------------------------------------------

def _metric_factual_accuracy() -> GEval:
    return GEval(
        name="factual_accuracy",
        criteria=(
            "The retrieval context contains raw query results as a list of records with numeric values. "
            "The output is a natural language summary. "
            "Evaluate ONLY whether each numeric value that appears in the output can be verified "
            "against the retrieval context. Acceptable forms: exact match, rounded to fewer decimals, "
            "abbreviated (e.g., '₹1.2M' for 1,234,567 or '₹85K' for 85,000). "
            "Do NOT penalise for summarising a subset of rows — it is valid to highlight only the "
            "top or most notable data points. "
            "Do NOT penalise for the time range or scope of the data; only check the numbers stated. "
            "A stated value that cannot be found in any row of the retrieval context is a failure. "
            "Stating a value from the wrong row or column is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.80,
        model=_JUDGE_MODEL,
    )


def _metric_question_relevance() -> GEval:
    return GEval(
        name="question_relevance",
        criteria=(
            "The input is the user's question. The output is a natural language analytics summary. "
            "Evaluate whether the summary directly answers what the user asked. "
            "A response discussing tangential metrics is a partial failure. "
            "A response answering a different question entirely is a complete failure."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.80,
        model=_JUDGE_MODEL,
    )


def _metric_data_coverage() -> GEval:
    return GEval(
        name="data_coverage",
        criteria=(
            "The retrieval context contains the raw query results. "
            "The output is a natural language summary. "
            "Evaluate whether the response covers the key insights: "
            "the top-ranked item (if ranked), the primary metric value for the leading entity, "
            "and any significant contrast between top and bottom items. "
            "Mentioning only one data point when many are available is a partial failure. "
            "Omitting the leading entity entirely is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.75,
        model=_JUDGE_MODEL,
    )


def _metric_executive_clarity() -> GEval:
    return GEval(
        name="executive_clarity",
        criteria=(
            "The output is a natural language summary for a non-technical executive audience. "
            "Check ONLY for the following violations — penalise proportionally to severity: "
            "(1) Presence of snake_case SQL column identifiers such as total_price, order_id, "
            "grand_total, created_at, user_id, variant_id, product_id, category_id — "
            "these are complete failures. "
            "(2) Presence of database table names in a technical context: order_items, "
            "product_variants — complete failures. "
            "(3) Unexplained technical abbreviations such as AOV, MoM, YoY — partial failure. "
            "The following are ACCEPTABLE and must NOT be penalised: "
            "- Formatted numbers of any size (e.g., ₹1.2M, ₹85K, 21 orders, 65 units). "
            "- Real-world entity names: city names, product names, customer names, "
            "  category names, brand names, payment method names. "
            "- Currency symbols (₹, $) and percentage signs. "
            "- Plain English business terms: 'average order value', 'total revenue', "
            "  'order count', 'repeat buyers'. "
            "A response free of snake_case identifiers and unexplained abbreviations "
            "should score at or near 1.0 regardless of number formatting style."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.70,
        model=_JUDGE_MODEL,
    )


def _metric_no_hallucination() -> GEval:
    return GEval(
        name="no_hallucination",
        criteria=(
            "The retrieval context contains the raw query results (as JSON) and the SQL used. "
            "The output is a natural language summary. "
            "Evaluate whether every specific factual claim in the output — entity names, "
            "numeric values, rankings, category names — can be traced back to a row or value "
            "in the retrieval context. "
            "Special case — empty results: if the retrieval context shows an empty list "
            "('step_1': []) and the output states that no data was found or revenue was zero, "
            "this is NOT a hallucination — award a near-perfect score. "
            "Special case — number abbreviations: ₹2.6M for 2,620,268 or ₹850K for 850,218 "
            "are mathematically correct rounded forms and are NOT hallucinations. "
            "Only flag a number as hallucinated if its rounded form cannot map to any value "
            "in the retrieval context (e.g., ₹850M for 850,218 IS a hallucination). "
            "Fabricating a product name, city, or ranking that does not appear "
            "anywhere in the retrieval context is a complete failure. "
            "Summarising or paraphrasing data that is present in the context is acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.80,
        model=_JUDGE_MODEL,
    )


_METRIC_FACTORIES = {
    "factual_accuracy": _metric_factual_accuracy,
    "question_relevance": _metric_question_relevance,
    "data_coverage": _metric_data_coverage,
    "executive_clarity": _metric_executive_clarity,
    "no_hallucination": _metric_no_hallucination,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case", _ALL_CASES,
    ids=[f"{c['id']:02d}-{c['category']}" for c in _ALL_CASES],
)
def test_analytics_reporter(case):
    state = _make_state(case)
    result = analytics_agent(state)
    assert result.analysis, f"analytics_agent returned empty analysis for case {case['id']}"
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result.analysis,
        retrieval_context=[json.dumps(case["raw_data"]), case["sql"]],
    )
    assert_test(test_case, [_METRIC_FACTORIES[m]() for m in case["geval_metrics"]])
