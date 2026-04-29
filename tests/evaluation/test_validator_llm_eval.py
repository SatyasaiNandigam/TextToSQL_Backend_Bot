"""
DeepEval GEval tests for the validator node (LLM validation path only).
Hard-rule blocking is already tested in tests/unit/test_validator_hard_rules.py.

Run:
    pytest tests/evaluation/ -v -m llm_eval -k "validator"
    pytest tests/evaluation/ -v -k "validator and false_positive"   # free, no API
"""
import json
import pathlib
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage
from core.state import AgentState
from schema.planner_schema import PlanStep
from pipeline.nodes.validator import validator

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"
_JUDGE_MODEL = "gpt-4o-mini"


def _load_dataset() -> list[dict]:
    with open(DATASETS_DIR / "validator_eval.json", encoding="utf-8") as f:
        return json.load(f)


_ALL_CASES = _load_dataset()
_TRUE_POSITIVE_CASES = [c for c in _ALL_CASES if c["expected_status"] == "invalid"]
_FALSE_POSITIVE_CASES = [c for c in _ALL_CASES if c["expected_status"] == "valid"]


def _make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["question"])],
        current_user_query=case["question"],
        resolved_query=case["question"],
        plan=[PlanStep(step_number=1, objective=case["objective"])],
        plan_index=0,
        sql=case["sql"],
        table_schema=case["table_schema"],
        retry_count=0,
    )


# ---------------------------------------------------------------------------
# GEval metric factories
# ---------------------------------------------------------------------------

def _metric_catch_schema_error() -> GEval:
    return GEval(
        name="catch_schema_error",
        criteria=(
            "The input contains an SQL query that uses a column or table name that does not exist "
            "in the provided schema. The output is the validator's JSON response. "
            "Evaluate whether the validator correctly identified that the SQL references an "
            "invalid column or table. The output status must be 'invalid' and the reason must "
            "specifically mention the invalid column or table name. "
            "A response of 'valid' when the SQL contains a schema error is a complete failure. "
            "A vague reason that does not identify the specific offending column or table is a partial failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


def _metric_catch_join_path_error() -> GEval:
    return GEval(
        name="catch_join_path_error",
        criteria=(
            "The input contains an SQL query that skips required intermediate tables in the join "
            "chain (e.g., order_items joined directly to products, or orders joined directly to "
            "categories without going through order_items, product_variants, products). "
            "Evaluate whether the validator correctly identified the invalid join path. "
            "The status must be 'invalid' and the reason must identify which tables were skipped. "
            "A 'valid' response is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


def _metric_catch_aggregation_error() -> GEval:
    return GEval(
        name="catch_aggregation_error",
        criteria=(
            "The input contains an SQL query that aggregates on order-level columns (e.g. SUM or AVG "
            "of a column from the orders table, or COUNT of an orders column without DISTINCT) while "
            "the orders table is also joined to order_items, causing row multiplication (fan-out). "
            "Evaluate whether the validator correctly identified this aggregation error. "
            "The status must be 'invalid'. The reason must identify the problematic aggregation "
            "or explain that the join causes row duplication inflating the result. "
            "A 'valid' response is a complete failure. "
            "A reason that correctly identifies the inflation risk — regardless of whether the "
            "aggregation uses SUM, AVG, or COUNT — is a passing response."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.70,
    )


def _metric_correct_valid_pass() -> GEval:
    return GEval(
        name="correct_valid_pass",
        criteria=(
            "The input contains a syntactically and semantically correct PostgreSQL SQL query "
            "that follows the schema exactly, uses correct join paths, correct aggregations, "
            "correct date filters, and produces the correct result shape for the stated objective. "
            "Evaluate whether the validator correctly passed this query as valid. "
            "The status must be 'valid' and failed_check must be an empty string. "
            "Any 'invalid' response for a correct query is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.80,
    )


def _metric_repair_hint_actionability() -> GEval:
    return GEval(
        name="repair_hint_actionability",
        criteria=(
            "The input contains an SQL query with a known error. "
            "The output is the validator's JSON response flagging it as invalid. "
            "Evaluate whether the repair_hint field provides a specific, actionable instruction "
            "that a developer could follow to fix the SQL without ambiguity. "
            "Vague hints like 'fix the SQL' or 'check the schema' are complete failures. "
            "A hint that correctly identifies the issue but lacks specificity is a partial failure."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=_JUDGE_MODEL,
        threshold=0.65,
    )


_METRIC_FACTORIES = {
    "catch_schema_error": _metric_catch_schema_error,
    "catch_join_path_error": _metric_catch_join_path_error,
    "catch_aggregation_error": _metric_catch_aggregation_error,
    "correct_valid_pass": _metric_correct_valid_pass,
    "repair_hint_actionability": _metric_repair_hint_actionability,
}


# ---------------------------------------------------------------------------
# Session-scoped result collector
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def validator_result_collector():
    results = []
    yield results
    _print_summary(results)


def _print_summary(results: list[dict]):
    print("\n")
    print("=" * 70)
    print("  VALIDATOR LLM EVAL — SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        marker = "+" if r["passed"] else "-"
        print(
            f"  [{marker}] {status}  cat={r['category']:<28}  "
            f"id={r['id']:>2}  {r['description'][:35]}"
        )
        if not r["passed"]:
            for metric_name, score, threshold in r["metric_scores"]:
                flag = "OK  " if score >= threshold else "FAIL"
                print(
                    f"           {flag}  {metric_name:<28}  "
                    f"score={score:.3f}  threshold={threshold:.2f}"
                )
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    if total:
        print(f"\n  OVERALL  {passed}/{total} ({passed / total * 100:.0f}%)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# True-positive tests — validator must flag invalid SQL
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case", _TRUE_POSITIVE_CASES,
    ids=[f"TP-{c['id']:02d}-{c['category']}" for c in _TRUE_POSITIVE_CASES],
)
def test_validator_true_positive(case, validator_result_collector):
    state = _make_state(case)
    result = validator(state)

    # Deterministic pre-check (free — no API cost)
    assert result.validation_feedback["status"] == "invalid", (
        f"Expected status='invalid' for case {case['id']}: {case['description']}\n"
        f"Got: {result.validation_feedback}"
    )
    assert result.validation_feedback["failed_check"] != "", (
        f"Expected non-empty failed_check for case {case['id']}"
    )

    metrics = [_METRIC_FACTORIES[m]() for m in case["geval_metrics"]]

    test_case = LLMTestCase(
        input=f"{case['question']}\nObjective: {case['objective']}\nSQL: {case['sql']}",
        actual_output=json.dumps(result.validation_feedback),
        retrieval_context=[case["table_schema"]],
    )

    for metric in metrics:
        metric.measure(test_case)

    metric_scores = [(m.name, m.score, m.threshold) for m in metrics]
    passed = all(m.score >= m.threshold for m in metrics)

    validator_result_collector.append({
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "passed": passed,
        "metric_scores": metric_scores,
    })

    assert_test(test_case, metrics)


# ---------------------------------------------------------------------------
# False-positive tests — valid SQL must not be rejected (deterministic, free)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case", _FALSE_POSITIVE_CASES,
    ids=[f"FP-{c['id']:02d}-{c['description'][:40]}" for c in _FALSE_POSITIVE_CASES],
)
def test_validator_false_positive(case):
    """Valid SQL must not be rejected. No API cost."""
    state = _make_state(case)
    result = validator(state)

    assert result.validation_feedback["status"] == "valid", (
        f"Valid SQL incorrectly rejected for case {case['id']}: {case['description']}\n"
        f"Got: {result.validation_feedback}"
    )
    assert result.validation_feedback["failed_check"] == "", (
        f"failed_check should be empty for valid SQL, case {case['id']}"
    )
