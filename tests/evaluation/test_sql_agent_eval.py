"""
DeepEval GEval tests for the sql_agent node.
Tests both fresh SQL generation and repair/regeneration with validation feedback.

Run:
    pytest tests/evaluation/ -v -m llm_eval -k "sql_agent"
    pytest tests/evaluation/ -v -m llm_eval -k "sql_agent and fresh"
    pytest tests/evaluation/ -v -m llm_eval -k "sql_agent and repair"
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
from pipeline.nodes.sql_agent import sql_agent

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"
_JUDGE_MODEL = "gpt-4o-mini"


def _load_dataset() -> list[dict]:
    with open(DATASETS_DIR / "sql_agent_eval.json", encoding="utf-8") as f:
        return json.load(f)


_ALL_CASES = _load_dataset()
_FRESH_CASES = [c for c in _ALL_CASES if c["scenario_type"] == "fresh_generation"]
_REPAIR_CASES = [c for c in _ALL_CASES if c["scenario_type"] == "repair"]


def _make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["question"])],
        current_user_query=case["question"],
        resolved_query=case["question"],
        plan=[PlanStep(step_number=1, objective=case["plan_objective"])],
        plan_index=0,
        table_schema=case["table_schema"],
        validation_feedback=case.get("validation_feedback"),
        error=case.get("error"),
        retry_count=1 if case.get("validation_feedback") else 0,
    )


# ---------------------------------------------------------------------------
# GEval metric factories
# ---------------------------------------------------------------------------

def _metric_schema_adherence() -> GEval:
    return GEval(
        name="schema_adherence",
        criteria=(
            "The input is a user question plus an SQL plan objective plus a table schema. "
            "The output is a PostgreSQL SQL query. "
            "Evaluate whether every table name and every column name referenced in the SQL "
            "exists in the provided table schema. "
            "A query that references any table or column not present in the schema is a complete failure. "
            "Do not penalize for alias usage — o=orders, oi=order_items, pv=product_variants, "
            "p=products, c=categories, b=brands, u=users are expected aliases."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.80,
    )


def _metric_join_path_correctness() -> GEval:
    return GEval(
        name="join_path_correctness",
        criteria=(
            "The output is a PostgreSQL SQL query. "
            "Evaluate whether all joins follow valid FK paths defined in the schema. "
            "Critical rule: order_items must not join directly to products — "
            "product_variants is always required as an intermediate table. "
            "Orders must not join directly to categories, brands, or products. "
            "The full chain is: orders -> order_items -> product_variants -> products -> categories/brands. "
            "Any illegal direct join that skips a required intermediate table is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


def _metric_aggregation_correctness() -> GEval:
    return GEval(
        name="aggregation_correctness",
        criteria=(
            "The output is a PostgreSQL SQL query for a revenue or spend metric. "
            "When order_items appears in FROM/JOIN clauses, revenue must be SUM(oi.total_price), "
            "NOT SUM(o.grand_total) — grand_total is an order-level value that repeats once per "
            "line item when order_items is joined, causing row multiplication. "
            "Using SUM(o.grand_total) when order_items is joined is a complete failure. "
            "When counting orders alongside order_items, COUNT(DISTINCT o.order_id) must be used. "
            "If only the orders table is used (no order_items), SUM(o.grand_total) is acceptable."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


def _metric_plan_fulfillment() -> GEval:
    return GEval(
        name="plan_fulfillment",
        criteria=(
            "The input contains a plan objective in plain English describing tables to join, "
            "filter conditions, aggregation function and column, output columns, and optional LIMIT. "
            "The output is a PostgreSQL SQL query. "
            "Evaluate whether the SQL implements all elements described in the objective: "
            "correct tables joined in the correct order, all filter conditions present, "
            "correct aggregation function on the correct column, correct output columns, "
            "and LIMIT applied if specified. "
            "Missing the primary filter condition or primary aggregation is a complete failure."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_JUDGE_MODEL,
        threshold=0.70,
    )


def _metric_repair_incorporates_hint() -> GEval:
    return GEval(
        name="repair_incorporates_hint",
        criteria=(
            "The input contains a prior SQL with a known error and a repair_hint describing the fix. "
            "The output is the regenerated SQL query. "
            "Evaluate whether the output SQL has corrected the specific issue identified in the repair_hint. "
            "Fixing the wrong issue entirely is a complete failure. "
            "Partially fixing the issue (e.g., adding the missing table but using the wrong join condition) "
            "is a partial failure. "
            "Introducing a new error while fixing the original is also a partial failure."
        ),
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=_JUDGE_MODEL,
        threshold=0.75,
    )


_METRIC_FACTORIES = {
    "schema_adherence": _metric_schema_adherence,
    "join_path_correctness": _metric_join_path_correctness,
    "aggregation_correctness": _metric_aggregation_correctness,
    "plan_fulfillment": _metric_plan_fulfillment,
    "repair_incorporates_hint": _metric_repair_incorporates_hint,
}


# ---------------------------------------------------------------------------
# Session-scoped result collector
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sql_agent_result_collector():
    results = {"fresh": [], "repair": []}
    yield results
    _print_summary(results)


def _print_summary(results: dict):
    print("\n")
    print("=" * 70)
    print("  SQL AGENT EVAL — SUMMARY")
    print("=" * 70)
    total_pass = total_fail = 0
    for scenario, cases in results.items():
        if not cases:
            continue
        passed = sum(1 for r in cases if r["passed"])
        failed = len(cases) - passed
        total_pass += passed
        total_fail += failed
        pct = passed / len(cases) * 100
        print(f"\n  {scenario.upper()}  {passed}/{len(cases)} ({pct:.0f}%)")
        print(f"  {'─' * 66}")
        for r in cases:
            status = "PASS" if r["passed"] else "FAIL"
            marker = "+" if r["passed"] else "-"
            print(
                f"  [{marker}] {status}  id={r['id']:>2}  cat={r['category']:<22}  "
                f"{r['description'][:30]}"
            )
            if not r["passed"]:
                for metric_name, score, threshold in r["metric_scores"]:
                    flag = "OK  " if score >= threshold else "FAIL"
                    print(
                        f"           {flag}  {metric_name:<28}  "
                        f"score={score:.3f}  threshold={threshold:.2f}"
                    )
    overall = total_pass + total_fail
    if overall:
        pct = total_pass / overall * 100
        print(f"\n  {'─' * 66}")
        print(f"  OVERALL  {total_pass}/{overall} ({pct:.0f}%)")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Fresh generation tests
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case", _FRESH_CASES,
    ids=[f"fresh-{c['id']:02d}-{c['category']}" for c in _FRESH_CASES],
)
def test_sql_agent_fresh_generation(case, sql_agent_result_collector):
    state = _make_state(case)
    result = sql_agent(state)

    assert result.sql, f"sql_agent returned empty SQL for case {case['id']}: {case['description']}"

    metrics = [_METRIC_FACTORIES[m]() for m in case["geval_metrics"]]

    test_case = LLMTestCase(
        input=f"{case['question']}\nObjective: {case['plan_objective']}",
        actual_output=result.sql,
        retrieval_context=[case["table_schema"]],
    )

    for metric in metrics:
        metric.measure(test_case)

    metric_scores = [(m.name, m.score, m.threshold) for m in metrics]
    passed = all(m.score >= m.threshold for m in metrics)

    sql_agent_result_collector["fresh"].append({
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "passed": passed,
        "metric_scores": metric_scores,
    })

    assert_test(test_case, metrics)


# ---------------------------------------------------------------------------
# Repair tests
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case", _REPAIR_CASES,
    ids=[f"repair-{c['id']:02d}-{c['category']}" for c in _REPAIR_CASES],
)
def test_sql_agent_repair(case, sql_agent_result_collector):
    state = _make_state(case)
    result = sql_agent(state)

    assert result.sql, f"sql_agent returned empty SQL for repair case {case['id']}: {case['description']}"

    repair_hint = case["validation_feedback"]["repair_hint"]
    metrics = [_METRIC_FACTORIES[m]() for m in case["geval_metrics"]]

    test_case = LLMTestCase(
        input=(
            f"{case['question']}\nObjective: {case['plan_objective']}\n"
            f"Repair hint: {repair_hint}"
        ),
        actual_output=result.sql,
        retrieval_context=[case["table_schema"]],
    )

    for metric in metrics:
        metric.measure(test_case)

    metric_scores = [(m.name, m.score, m.threshold) for m in metrics]
    passed = all(m.score >= m.threshold for m in metrics)

    sql_agent_result_collector["repair"].append({
        "id": case["id"],
        "category": case["category"],
        "description": case["description"],
        "passed": passed,
        "metric_scores": metric_scores,
    })

    assert_test(test_case, metrics)
