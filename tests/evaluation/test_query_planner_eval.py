"""
DeepEval GEval tests for the query_planner node.

Run:
    pytest tests/evaluation/ -v -m llm_eval -k "query_planner"
"""
import json
import pathlib
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage
from core.state import AgentState
from schema.intent_classification_schema import IntentClassifierSchema
from pipeline.nodes.query_planner import query_planner

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"
_JUDGE_MODEL = "gpt-4o-mini"


def _load_dataset() -> list[dict]:
    with open(DATASETS_DIR / "query_planner_eval.json", encoding="utf-8") as f:
        return json.load(f)


_ALL_CASES = _load_dataset()


def _make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["question"])],
        current_user_query=case["question"],
        resolved_query=case["question"],
        intent_result=IntentClassifierSchema(**case["intent_result"]),
        table_schema=case["table_schema"],
    )


# ---------------------------------------------------------------------------
# GEval metric factories
# ---------------------------------------------------------------------------

def _metric_objective_completeness() -> GEval:
    return GEval(
        name="objective_completeness",
        criteria=(
            "The input is a user question and intent metadata. "
            "The output is a plan objective written in plain English. "
            "Evaluate whether the objective includes: "
            "(1) Which tables to join and in what sequence. "
            "(2) What filter conditions to apply (date, category, status). "
            "(3) What aggregation function and column to aggregate (SUM/COUNT/AVG on which column). "
            "(4) What to return in the result (columns, LIMIT if top-N). "
            "Missing the primary aggregation or the primary filter is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


def _metric_aggregation_rule_adherence() -> GEval:
    return GEval(
        name="aggregation_rule_adherence",
        criteria=(
            "The input is a question asking for revenue or spend when products or categories are involved. "
            "The output is a plan objective in plain English. "
            "Evaluate whether the objective correctly specifies SUM(oi.total_price) as the revenue column "
            "when order_items must be joined. "
            "If the objective specifies SUM(o.grand_total) in a context where order_items must be joined, "
            "this is a complete failure — grand_total repeats per line item causing row multiplication. "
            "If the question only needs the orders table, SUM(o.grand_total) is acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.85,
    )


def _metric_join_path_specification() -> GEval:
    return GEval(
        name="join_path_specification",
        criteria=(
            "The input is a question requiring multiple table joins. "
            "The output is a plan objective describing which tables to join. "
            "For product/category/brand queries the chain must be: "
            "orders -> order_items -> product_variants -> products -> categories/brands. "
            "An objective that says 'join orders to products' without mentioning order_items "
            "and product_variants describes an invalid join path and is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


def _metric_date_filter_precision() -> GEval:
    return GEval(
        name="date_filter_precision",
        criteria=(
            "The input is a question with an explicit time constraint. "
            "The output is a plan objective in plain English. "
            "Evaluate whether the date filter is expressed using a precise SQL-ready pattern: "
            "- 'last 30 days' -> o.created_at >= CURRENT_DATE - INTERVAL '30 days' "
            "- 'this month' -> DATE_TRUNC('month', o.created_at) = DATE_TRUNC('month', CURRENT_DATE) "
            "- 'in 2024' -> o.created_at >= '2024-01-01' AND o.created_at < '2025-01-01' "
            "An objective with no date filter when one was required is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.70,
    )


_METRIC_FACTORIES = {
    "objective_completeness": _metric_objective_completeness,
    "aggregation_rule_adherence": _metric_aggregation_rule_adherence,
    "join_path_specification": _metric_join_path_specification,
    "date_filter_precision": _metric_date_filter_precision,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case", _ALL_CASES,
    ids=[f"{c['id']:02d}-{c['category']}" for c in _ALL_CASES],
)
def test_query_planner(case):
    state = _make_state(case)
    result = query_planner(state)
    assert result.plan and len(result.plan) > 0, (
        f"query_planner returned empty plan for case {case['id']}"
    )
    test_case = LLMTestCase(
        input=f"{case['question']}\nIntent: {json.dumps(case['intent_result'])}",
        actual_output=result.plan[0].objective,
        retrieval_context=[case["table_schema"]],
    )
    assert_test(test_case, [_METRIC_FACTORIES[m]() for m in case["geval_metrics"]])
