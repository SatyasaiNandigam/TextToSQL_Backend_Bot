"""
Behavioral tests for the intent_classifier node.

Measures classification accuracy (intent + metric + needs_chart) against ground
truth using the real Groq LLM.

Run:
    pytest tests/behavioral/ -v -m behavioral -k "intent_classifier"
    pytest tests/behavioral/ -v -m behavioral -k "intent_classifier and level1"
"""
import json
import pathlib
import pytest
from langchain_core.messages import HumanMessage
from core.state import AgentState
from pipeline.nodes.intent_classifier import intent_classifier

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"


def load_dataset(level: str) -> list[dict]:
    path = DATASETS_DIR / f"intent_classifier_{level}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["query"])],
        current_user_query=case["query"],
    )


@pytest.fixture(scope="session")
def result_collector():
    results = {"level1": [], "level2": []}
    yield results
    _print_summary(results)


def _print_summary(results: dict):
    print("\n")
    print("=" * 60)
    print("  INTENT CLASSIFIER — BEHAVIORAL TEST SUMMARY")
    print("=" * 60)
    total_pass = total_fail = 0
    for level, cases in results.items():
        if not cases:
            continue
        passed = sum(1 for r in cases if r["passed"])
        failed = len(cases) - passed
        total_pass += passed
        total_fail += failed
        pct = passed / len(cases) * 100
        print(f"\n  {level.upper()}  {passed}/{len(cases)} ({pct:.0f}%)")
        print(f"  {'─' * 56}")
        for r in cases:
            status = "PASS" if r["passed"] else "FAIL"
            marker = "+" if r["passed"] else "-"
            print(f"  [{marker}] {status}  id={r['id']:>2}  {r['scenario']}")
            if not r["passed"]:
                print(f"           intent:      expected={r['expected_intent']!r}  got={r['got_intent']!r}")
                print(f"           metric:      expected={r['expected_metric']!r}  got={r['got_metric']!r}")
                print(f"           needs_chart: expected={r['expected_needs_chart']!r}  got={r['got_needs_chart']!r}")
    overall = total_pass + total_fail
    if overall:
        pct = total_pass / overall * 100
        print(f"\n  {'─' * 56}")
        print(f"  OVERALL  {total_pass}/{overall} ({pct:.0f}%)")
    print("=" * 60)


@pytest.mark.behavioral
@pytest.mark.parametrize(
    "case",
    load_dataset("level1"),
    ids=[f"L1-{c['id']:02d}-{c['scenario'][:40]}" for c in load_dataset("level1")],
)
def test_level1_intent_classification(case, result_collector):
    state = make_state(case)
    result = intent_classifier(state)

    got_intent = result.intent_result.intent
    got_metric = result.intent_result.metric
    got_needs_chart = result.intent_result.needs_chart

    intent_correct = got_intent == case["expected_intent"]
    # Only assert metric when expected is non-null (null means "don't care")
    metric_correct = (case["expected_metric"] is None) or (got_metric == case["expected_metric"])
    chart_correct = got_needs_chart == case["expected_needs_chart"]
    passed = intent_correct and metric_correct and chart_correct

    result_collector["level1"].append({
        "id": case["id"],
        "scenario": case["scenario"],
        "passed": passed,
        "expected_intent": case["expected_intent"],
        "got_intent": got_intent,
        "expected_metric": case["expected_metric"],
        "got_metric": got_metric,
        "expected_needs_chart": case["expected_needs_chart"],
        "got_needs_chart": got_needs_chart,
    })

    assert intent_correct, (
        f"intent mismatch: expected {case['expected_intent']!r}, got {got_intent!r}\n"
        f"  query: {case['query']!r}"
    )
    assert metric_correct, (
        f"metric mismatch: expected {case['expected_metric']!r}, got {got_metric!r}\n"
        f"  query: {case['query']!r}"
    )
    assert chart_correct, (
        f"needs_chart mismatch: expected {case['expected_needs_chart']!r}, got {got_needs_chart!r}\n"
        f"  query: {case['query']!r}"
    )


@pytest.mark.behavioral
@pytest.mark.parametrize(
    "case",
    load_dataset("level2"),
    ids=[f"L2-{c['id']:02d}-{c['scenario'][:40]}" for c in load_dataset("level2")],
)
def test_level2_intent_classification(case, result_collector):
    state = make_state(case)
    result = intent_classifier(state)

    got_intent = result.intent_result.intent
    got_metric = result.intent_result.metric
    got_needs_chart = result.intent_result.needs_chart

    intent_correct = got_intent == case["expected_intent"]
    metric_correct = (case["expected_metric"] is None) or (got_metric == case["expected_metric"])
    chart_correct = got_needs_chart == case["expected_needs_chart"]
    passed = intent_correct and metric_correct and chart_correct

    result_collector["level2"].append({
        "id": case["id"],
        "scenario": case["scenario"],
        "passed": passed,
        "expected_intent": case["expected_intent"],
        "got_intent": got_intent,
        "expected_metric": case["expected_metric"],
        "got_metric": got_metric,
        "expected_needs_chart": case["expected_needs_chart"],
        "got_needs_chart": got_needs_chart,
    })

    assert intent_correct, (
        f"intent mismatch: expected {case['expected_intent']!r}, got {got_intent!r}\n"
        f"  query: {case['query']!r}"
    )
    assert metric_correct, (
        f"metric mismatch: expected {case['expected_metric']!r}, got {got_metric!r}\n"
        f"  query: {case['query']!r}"
    )
    assert chart_correct, (
        f"needs_chart mismatch: expected {case['expected_needs_chart']!r}, got {got_needs_chart!r}\n"
        f"  query: {case['query']!r}"
    )
