"""
DeepEval GEval tests for the schema_summarizer node.

Run:
    pytest tests/evaluation/ -v -m llm_eval -k "schema_summarizer"
"""
import pathlib
import json
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage
from core.state import AgentState
from pipeline.nodes.schema_summarizer import schema_summarizer

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"


def _load_dataset() -> list[dict]:
    with open(DATASETS_DIR / "schema_summarizer_eval.json", encoding="utf-8") as f:
        return json.load(f)


_ALL_CASES = _load_dataset()


def _make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["question"])],
        current_user_query=case["question"],
        resolved_query=case["question"],
        table_schema=case["raw_schema"],
    )


def _metric_fk_path_preservation() -> GEval:
    return GEval(
        name="fk_path_preservation",
        criteria=(
            "The input contains raw schema context for multiple tables plus a user question. "
            "The output is a compressed schema summary. "
            "Evaluate whether the output correctly identifies the FK join paths needed to answer "
            "the question, including all intermediate bridge tables. "
            "For product/category queries the full chain is: "
            "orders -> order_items -> product_variants -> products -> categories, with join keys. "
            "Missing a critical intermediate table is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.80,
    )


def _metric_column_completeness() -> GEval:
    return GEval(
        name="column_completeness",
        criteria=(
            "The input contains raw schema for tables relevant to the user question. "
            "The output is a compressed schema summary. "
            "Evaluate whether the output includes the specific columns needed: "
            "For revenue queries: total_price (order_items) and grand_total (orders) must appear. "
            "For count queries: the relevant PK must appear. "
            "For date-filtered queries: created_at with the correct table must appear. "
            "Missing the primary metric column is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.75,
    )


def _metric_noise_suppression() -> GEval:
    return GEval(
        name="noise_suppression",
        criteria=(
            "The input contains raw schema for potentially many tables. "
            "The output is a compressed schema summary for a focused user question. "
            "Evaluate whether the output includes only tables relevant to the query "
            "and excludes irrelevant ones. "
            "A summary with excessive irrelevant schema is a partial failure. "
            "A focused, minimal summary receives a perfect score."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.70,
    )


_METRIC_FACTORIES = {
    "fk_path_preservation": _metric_fk_path_preservation,
    "column_completeness": _metric_column_completeness,
    "noise_suppression": _metric_noise_suppression,
}


@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case", _ALL_CASES,
    ids=[f"{c['id']:02d}-{c['category']}" for c in _ALL_CASES],
)
def test_schema_summarizer(case):
    state = _make_state(case)
    result = schema_summarizer(state)
    assert result.table_schema, (
        f"schema_summarizer returned empty table_schema for case {case['id']}"
    )
    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result.table_schema,
        retrieval_context=[case["raw_schema"]],
    )
    assert_test(test_case, [_METRIC_FACTORIES[m]() for m in case["geval_metrics"]])
