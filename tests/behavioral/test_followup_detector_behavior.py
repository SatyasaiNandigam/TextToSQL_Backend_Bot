"""
Behavioral tests for the followup_detector node.

These tests call the real Groq LLM with the actual prompt and measure
classification accuracy against a ground-truth dataset.

Run:
    pytest tests/behavioral/ -v -m behavioral
    pytest tests/behavioral/ -v -m behavioral -k "level1"
    pytest tests/behavioral/ -v -m behavioral -k "level2"

Skip in fast CI:
    pytest tests/unit/ -v
"""
import json
import pathlib
import pytest
from langchain_core.messages import HumanMessage
from core.state import AgentState
from pipeline.nodes.followup_detector import followup_detector

DATASETS_DIR = pathlib.Path(__file__).parents[2] / "datasets"


def load_dataset(level: str) -> list[dict]:
    path = DATASETS_DIR / f"followup_detector_{level}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_state(case: dict) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content=case["current_query"])],
        current_user_query=case["current_query"],
        last_objective=None if case["prior_objective"] == "N/A" else case["prior_objective"],
        last_result_summary=None if case["prior_summary"] == "N/A" else case["prior_summary"],
    )


# ---------------------------------------------------------------------------
# Session-scoped result collector for the summary report
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def result_collector():
    results = {"level1": [], "level2": []}
    yield results
    _print_summary(results)


def _print_summary(results: dict):
    print("\n")
    print("=" * 60)
    print("  FOLLOWUP DETECTOR — BEHAVIORAL TEST SUMMARY")
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
                print(f"           is_followup: expected={r['expected_is_followup']!r}  got={r['got_is_followup']!r}")
                print(f"           type:        expected={r['expected_type']!r}  got={r['got_type']!r}")

    overall = total_pass + total_fail
    if overall:
        pct = total_pass / overall * 100
        print(f"\n  {'─' * 56}")
        print(f"  OVERALL  {total_pass}/{overall} ({pct:.0f}%)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Level 1 — Straightforward cases
# ---------------------------------------------------------------------------

@pytest.mark.behavioral
@pytest.mark.parametrize(
    "case",
    load_dataset("level1"),
    ids=[f"L1-{c['id']:02d}-{c['scenario'][:40]}" for c in load_dataset("level1")],
)
def test_level1_followup_detection(case, result_collector):
    state = make_state(case)
    result = followup_detector(state)

    got_is_followup = result.followup.is_followup.lower()
    got_type = result.followup.type

    is_followup_correct = got_is_followup == case["expected_is_followup"]
    type_correct = got_type == case["expected_type"]
    passed = is_followup_correct and type_correct

    result_collector["level1"].append({
        "id": case["id"],
        "scenario": case["scenario"],
        "passed": passed,
        "expected_is_followup": case["expected_is_followup"],
        "got_is_followup": got_is_followup,
        "expected_type": case["expected_type"],
        "got_type": got_type,
    })

    assert is_followup_correct, (
        f"is_followup mismatch: expected {case['expected_is_followup']!r}, got {got_is_followup!r}\n"
        f"  query: {case['current_query']!r}"
    )
    assert type_correct, (
        f"type mismatch: expected {case['expected_type']!r}, got {got_type!r}\n"
        f"  query: {case['current_query']!r}"
    )


# ---------------------------------------------------------------------------
# Level 2 — Complex / ambiguous cases
# ---------------------------------------------------------------------------

@pytest.mark.behavioral
@pytest.mark.parametrize(
    "case",
    load_dataset("level2"),
    ids=[f"L2-{c['id']:02d}-{c['scenario'][:40]}" for c in load_dataset("level2")],
)
def test_level2_followup_detection(case, result_collector):
    state = make_state(case)
    result = followup_detector(state)

    got_is_followup = result.followup.is_followup.lower()
    got_type = result.followup.type

    is_followup_correct = got_is_followup == case["expected_is_followup"]
    type_correct = got_type == case["expected_type"]
    passed = is_followup_correct and type_correct

    result_collector["level2"].append({
        "id": case["id"],
        "scenario": case["scenario"],
        "passed": passed,
        "expected_is_followup": case["expected_is_followup"],
        "got_is_followup": got_is_followup,
        "expected_type": case["expected_type"],
        "got_type": got_type,
    })

    assert is_followup_correct, (
        f"is_followup mismatch: expected {case['expected_is_followup']!r}, got {got_is_followup!r}\n"
        f"  query: {case['current_query']!r}"
    )
    assert type_correct, (
        f"type mismatch: expected {case['expected_type']!r}, got {got_type!r}\n"
        f"  query: {case['current_query']!r}"
    )
