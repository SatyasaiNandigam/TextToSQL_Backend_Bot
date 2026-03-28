"""
Unit tests for all pipeline edge routing functions.
These are pure conditionals on AgentState fields — no LLM or DB required.
"""
import pytest
from pipeline.edges.rewriter_router import route_after_rewriter
from pipeline.edges.validation_router import route_after_validation
from pipeline.edges.execution_router import route_after_execution
from pipeline.edges.intent_classification_router import route_after_intent_classification
from tests.fixtures.agent_states import make_followup_state, make_post_execution_state, make_fresh_query_state, make_plan


# ---------------------------------------------------------------------------
# route_after_rewriter
# ---------------------------------------------------------------------------

class TestRewriterRouter:
    def test_fresh_query_routes_to_fresh(self):
        state = make_followup_state("How many orders?", is_followup="False", followup_type="new")
        assert route_after_rewriter(state) == "fresh"

    def test_no_followup_object_routes_to_fresh(self):
        state = make_fresh_query_state()
        assert route_after_rewriter(state) == "fresh"

    @pytest.mark.parametrize("followup_type", ["explain", "visualize"])
    def test_reuse_types_route_to_reuse(self, followup_type):
        state = make_followup_state("show me a chart", is_followup="True", followup_type=followup_type)
        assert route_after_rewriter(state) == "reuse"

    @pytest.mark.parametrize("followup_type", ["refine", "regroup", "drilldown"])
    def test_rewrite_types_route_to_rewrite(self, followup_type):
        state = make_followup_state("filter by city", is_followup="True", followup_type=followup_type)
        assert route_after_rewriter(state) == "rewrite"

    def test_is_followup_false_string_routes_to_fresh(self):
        state = make_followup_state("How many?", is_followup="False", followup_type="refine")
        assert route_after_rewriter(state) == "fresh"


# ---------------------------------------------------------------------------
# route_after_validation
# ---------------------------------------------------------------------------

class TestValidationRouter:
    def _state_with_feedback(self, status, retry_count=0, error=None):
        state = make_fresh_query_state()
        state.validation_feedback = {
            "status": status,
            "failed_check": "",
            "reason": "some reason",
            "repair_hint": "some hint",
        }
        state.retry_count = retry_count
        state.error = error
        return state

    def test_valid_feedback_routes_to_execute(self):
        state = self._state_with_feedback("valid")
        assert route_after_validation(state) == "execute"

    def test_hard_rule_violation_routes_to_max_retries(self):
        state = self._state_with_feedback("invalid", error="hard_rule_violation")
        assert route_after_validation(state) == "max_retries"

    def test_invalid_with_retries_remaining_routes_to_regenerate(self):
        state = self._state_with_feedback("invalid", retry_count=1)
        assert route_after_validation(state) == "regenerate"

    def test_invalid_at_retry_limit_routes_to_max_retries(self):
        state = self._state_with_feedback("invalid", retry_count=3)
        assert route_after_validation(state) == "max_retries"

    def test_restricted_status_routes_to_restricted(self):
        state = self._state_with_feedback("restricted")
        assert route_after_validation(state) == "restricted"

    def test_no_feedback_with_no_error_routes_to_regenerate(self):
        state = make_fresh_query_state()
        state.validation_feedback = None
        state.retry_count = 0
        assert route_after_validation(state) == "regenerate"

    def test_retry_count_2_still_routes_to_regenerate(self):
        state = self._state_with_feedback("invalid", retry_count=2)
        assert route_after_validation(state) == "regenerate"


# ---------------------------------------------------------------------------
# route_after_execution
# ---------------------------------------------------------------------------

class TestExecutionRouter:
    def test_all_steps_done_routes_to_done(self):
        plan = make_plan("Step 1")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1",
            raw_data={"step_1": [{"n": 1}]},
        )
        state.plan_index = 1  # plan has 1 step, index is now 1 → done
        assert route_after_execution(state) == "done"

    def test_more_steps_remaining_routes_to_next_step(self):
        plan = make_plan("Step 1", "Step 2")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1",
            raw_data={"step_1": [{"n": 1}]},
        )
        state.plan_index = 1  # step 1 done, step 2 still pending
        assert route_after_execution(state) == "next_step"

    def test_error_with_retries_remaining_routes_to_next_step(self):
        plan = make_plan("Step 1")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1",
            raw_data=None, error="relation does not exist", retry_count=1,
        )
        assert route_after_execution(state) == "next_step"

    def test_error_at_retry_limit_routes_to_failed(self):
        plan = make_plan("Step 1")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1",
            raw_data=None, error="relation does not exist", retry_count=3,
        )
        assert route_after_execution(state) == "failed"

    def test_error_at_exactly_3_retries_routes_to_failed(self):
        plan = make_plan("Step 1")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1",
            raw_data=None, error="db error", retry_count=3,
        )
        assert route_after_execution(state) == "failed"

    def test_error_at_2_retries_still_routes_to_next_step(self):
        plan = make_plan("Step 1")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1",
            raw_data=None, error="db error", retry_count=2,
        )
        assert route_after_execution(state) == "next_step"


# ---------------------------------------------------------------------------
# route_after_intent_classification
# ---------------------------------------------------------------------------

class TestIntentClassificationRouter:
    def test_no_error_routes_to_query_planner(self):
        state = make_fresh_query_state()
        assert route_after_intent_classification(state) == "query_planner"

    def test_restricted_intent_error_routes_to_hard_rule_violation(self):
        state = make_fresh_query_state()
        state.error = "restricted_operations_intent"
        assert route_after_intent_classification(state) == "hard_rule_violation"

    def test_other_error_still_routes_to_query_planner(self):
        state = make_fresh_query_state()
        state.error = "some_other_error"
        assert route_after_intent_classification(state) == "query_planner"
