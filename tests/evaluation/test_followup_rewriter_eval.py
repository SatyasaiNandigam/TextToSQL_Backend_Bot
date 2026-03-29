"""
deepeval evaluation for the followup_rewriter node.

Measures rewrite quality using LLM-judge metrics:
  - standalone_completeness: no unresolved pronouns/references
  - intent_preservation: same analytical intent as the terse original query
  - context_integration: correctly pulls entity names from prior result summary
  - no_hallucination: no invented terms not present in input or context
  - query_specificity: rewritten query is specific enough to generate correct SQL

Passthrough cases (is_followup='false') use a deterministic assertion only.

Run:
    pytest tests/evaluation/ -v -m llm_eval          # LLM-judge cases
    pytest tests/evaluation/ -v -k "passthrough"     # deterministic cases
    pytest tests/evaluation/ -v                      # all cases
    
    pytest tests/evaluation/ -v -k "passthrough" -s           # deterministic only (free)       
    pytest tests/evaluation/ -v -m llm_eval -k "REFINE" -s   # smoke test one type
    pytest tests/evaluation/ -v -m llm_eval -s               # full LLM eval run
    pytest tests/unit/ -v                                     # unit tests unaffected

Skip in fast CI:
    pytest tests/unit/ -v
"""
import json
import pathlib

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage

from core.state import AgentState
from pipeline.nodes.followup_rewriter import followup_rewriter
from schema.follow_up_schema import FollowUpSchema

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _load_eval_dataset() -> list[dict]:
    path = DATASETS_DIR / "followup_rewriter_eval.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_ALL_CASES = _load_eval_dataset()
_REWRITE_CASES = [c for c in _ALL_CASES if c["is_followup"] == "true"]
_PASSTHROUGH_CASES = [c for c in _ALL_CASES if c["is_followup"] == "false"]


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def _make_rewriter_state(case: dict) -> AgentState:
    # FollowUpSchema.type Literal does not accept "passthrough"; map it to "new"
    followup_type = case["followup_type"] if case["followup_type"] != "passthrough" else "new"
    return AgentState(
        messages=[HumanMessage(content=case["current_query"])],
        current_user_query=case["current_query"],
        last_result_summary=case.get("last_result_summary"),
        followup=FollowUpSchema(
            is_followup=case["is_followup"],
            type=followup_type,
            instruction="",
        ),
    )


# ---------------------------------------------------------------------------
# deepeval metric factories — one fresh instance per test to avoid stale state
# ---------------------------------------------------------------------------

def _geval_standalone() -> GEval:
    return GEval(
        name="standalone_completeness",
        criteria=(
            "The input is a terse follow-up query. The retrieval context is the prior result "
            "summary that provides the missing context. The output is the rewritten query. "
            "Evaluate whether the output query is fully self-contained: a reader who sees only "
            "the output (without the input or the retrieval context) can understand exactly what "
            "data is being requested. Entity names, time ranges, filters, and dimensions should "
            "be explicit. No pronouns or implicit references (like 'those', 'that', 'them', 'it') "
            "should remain unresolved. It is correct and expected for the output to include "
            "specifics from the retrieval context."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
    )


def _geval_intent() -> GEval:
    return GEval(
        name="intent_preservation",
        criteria=(
            "The input is a terse follow-up query. The retrieval context is the prior result "
            "summary. The output is the rewritten standalone query. "
            "Evaluate whether the output preserves the core analytical operation expressed in "
            "the input (e.g., changing the limit to 5, filtering by a status, switching "
            "a category, shifting a time window). The output may and should add specifics from "
            "the retrieval context to make the query standalone — this is expected. "
            "Penalize only if the core operation from the input is dropped or changed."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
    )


def _geval_context() -> GEval:
    return GEval(
        name="context_integration",
        criteria=(
            "The output query correctly incorporates relevant information from the prior "
            "result context (provided in the retrieval context field) to resolve ambiguities. "
            "Entity names, metrics, time windows, or grouping dimensions from the prior "
            "summary appear in the rewritten query where needed to make it self-contained."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.65,
    )


def _geval_no_hallucination() -> GEval:
    return GEval(
        name="no_hallucination",
        criteria=(
            "The output query does not introduce business entities, time periods, "
            "product names, category names, or metrics that appear in neither the "
            "input query nor the retrieval context (prior result summary). "
            "Every specific term in the output must be traceable to either the input or the context."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.8,
    )


def _geval_query_specificity() -> GEval:
    return GEval(
        name="query_specificity",
        criteria=(
            "The input is a terse follow-up query. The retrieval context is the prior result "
            "summary. The output is the rewritten standalone query. "
            "Evaluate whether the output query is specific enough to generate a correct SQL "
            "query without further clarification. It should name concrete entities (product "
            "names, categories, time periods, cities, metrics) drawn from the input and the "
            "retrieval context. A database analyst reading only the output query should know "
            "exactly which tables, columns, filters, and aggregations to use. "
            "Adding specifics from the retrieval context is correct and expected behavior."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        threshold=0.7,
    )


def _select_metrics(followup_type: str) -> list:
    if followup_type == "refine":
        return [
            _geval_standalone(),
            _geval_intent(),
            _geval_no_hallucination(),
            _geval_query_specificity(),
        ]
    if followup_type == "regroup":
        return [
            _geval_standalone(),
            _geval_intent(),
            _geval_context(),
            _geval_no_hallucination(),
        ]
    if followup_type == "drilldown":
        return [
            _geval_standalone(),
            _geval_context(),
            _geval_no_hallucination(),
            _geval_query_specificity(),
        ]
    return [_geval_standalone(), _geval_intent(), _geval_no_hallucination()]


# ---------------------------------------------------------------------------
# Session-scoped result collector for summary report
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def eval_result_collector():
    results = []
    yield results
    _print_eval_summary(results)


def _print_eval_summary(results: list[dict]):
    print("\n")
    print("=" * 70)
    print("  FOLLOWUP REWRITER — LLM EVAL SUMMARY")
    print("=" * 70)

    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        marker = "+" if r["passed"] else "-"
        print(
            f"  [{marker}] {status}  type={r['followup_type']:<10}  "
            f"id={r['id']:>2}  {r['description'][:40]}"
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
# LLM eval tests — parametrized over rewrite cases
# ---------------------------------------------------------------------------

@pytest.mark.llm_eval
@pytest.mark.parametrize(
    "case",
    _REWRITE_CASES,
    ids=[
        f"{c['followup_type'].upper()}-{c['id']:02d}-{c['description'][:35]}"
        for c in _REWRITE_CASES
    ],
)
def test_followup_rewriter_quality(case, eval_result_collector):
    state = _make_rewriter_state(case)
    result = followup_rewriter(state)
    actual = result.resolved_query

    test_case = LLMTestCase(
        input=case["current_query"],
        actual_output=actual,
        expected_output=case["expected_rewritten_query"],
        retrieval_context=(
            [case["last_result_summary"]] if case.get("last_result_summary") else []
        ),
    )

    metrics = _select_metrics(case["followup_type"])

    # Measure all metrics before calling assert_test so we can capture scores
    for metric in metrics:
        metric.measure(test_case)

    metric_scores = [
        (m.name, m.score, m.threshold) for m in metrics
    ]
    passed = all(m.score >= m.threshold for m in metrics)

    eval_result_collector.append({
        "id": case["id"],
        "description": case["description"],
        "followup_type": case["followup_type"],
        "passed": passed,
        "metric_scores": metric_scores,
    })

    assert_test(test_case, metrics)


# ---------------------------------------------------------------------------
# Passthrough tests — deterministic, no LLM judge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "case",
    _PASSTHROUGH_CASES,
    ids=[
        f"PASSTHROUGH-{c['id']:02d}-{c['description'][:35]}"
        for c in _PASSTHROUGH_CASES
    ],
)
def test_followup_rewriter_passthrough(case):
    """When is_followup='false', the node must return current_query unchanged."""
    state = _make_rewriter_state(case)
    result = followup_rewriter(state)

    assert result.resolved_query == case["current_query"], (
        f"Passthrough failed: expected {case['current_query']!r}, "
        f"got {result.resolved_query!r}"
    )
