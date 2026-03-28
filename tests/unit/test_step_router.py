import pytest
from pipeline.nodes.step_router import step_router
from tests.fixtures.agent_states import make_post_execution_state, make_plan


@pytest.fixture
def single_step_success():
    plan = make_plan("Count orders by city")
    return make_post_execution_state(
        query="Orders per city",
        plan=plan,
        sql="SELECT city, COUNT(*) FROM orders GROUP BY city",
        raw_data={"step_1": [{"city": "Mumbai", "count": 5}, {"city": "Delhi", "count": 3}]},
    )


@pytest.fixture
def two_step_mid_plan():
    plan = make_plan("Count orders by city", "Compute avg order value")
    return make_post_execution_state(
        query="Orders per city and AOV",
        plan=plan,
        sql="SELECT city, COUNT(*) FROM orders GROUP BY city",
        raw_data={"step_1": [{"city": "Mumbai", "count": 5}]},
    )


@pytest.fixture
def execution_error_state():
    plan = make_plan("Count orders")
    return make_post_execution_state(
        query="Count orders",
        plan=plan,
        sql="SELECT COUNT(*) FROM nonexistent",
        raw_data=None,
        error='relation "nonexistent" does not exist',
        retry_count=1,
    )


class TestStepRouterSuccess:
    def test_advances_plan_index(self, single_step_success):
        result = step_router(single_step_success)
        assert result.plan_index == 1

    def test_appends_completed_step(self, single_step_success):
        result = step_router(single_step_success)
        assert len(result.completed_steps) == 1

    def test_completed_step_has_correct_step_number(self, single_step_success):
        result = step_router(single_step_success)
        assert result.completed_steps[0]["step_number"] == 1

    def test_completed_step_has_correct_objective(self, single_step_success):
        result = step_router(single_step_success)
        assert result.completed_steps[0]["objective"] == "Count orders by city"

    def test_completed_step_has_correct_sql(self, single_step_success):
        result = step_router(single_step_success)
        assert "COUNT" in result.completed_steps[0]["sql"]

    def test_completed_step_sample_rows_capped_at_3(self):
        plan = make_plan("Fetch rows")
        many_rows = [{"id": i} for i in range(10)]
        state = make_post_execution_state(
            query="Fetch rows", plan=plan,
            sql="SELECT id FROM orders",
            raw_data={"step_1": many_rows},
        )
        result = step_router(state)
        assert len(result.completed_steps[0]["sample_rows"]) <= 3

    def test_completed_step_columns_extracted_from_first_row(self, single_step_success):
        result = step_router(single_step_success)
        assert result.completed_steps[0]["columns"] == ["city", "count"]

    def test_resets_sql_after_step(self, single_step_success):
        result = step_router(single_step_success)
        assert result.sql is None

    def test_resets_sql_executed(self, single_step_success):
        result = step_router(single_step_success)
        assert result.sql_executed is False

    def test_resets_validation_feedback(self):
        plan = make_plan("Step")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1", raw_data={"step_1": [{"n": 1}]}
        )
        state.validation_feedback = {"status": "valid", "failed_check": "", "reason": "", "repair_hint": ""}
        result = step_router(state)
        assert result.validation_feedback is None

    def test_resets_retry_count_to_zero(self):
        plan = make_plan("Step")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1", raw_data={"step_1": [{"n": 1}]},
            retry_count=2,
        )
        result = step_router(state)
        assert result.retry_count == 0

    def test_empty_raw_data_step_produces_empty_columns_and_rows(self):
        plan = make_plan("Empty result step")
        state = make_post_execution_state(
            query="q", plan=plan, sql="SELECT 1 WHERE false",
            raw_data={"step_1": []},
        )
        result = step_router(state)
        entry = result.completed_steps[0]
        assert entry["columns"] == []
        assert entry["sample_rows"] == []


class TestStepRouterMultiStep:
    def test_index_advances_to_next_step(self, two_step_mid_plan):
        result = step_router(two_step_mid_plan)
        assert result.plan_index == 1

    def test_only_one_step_in_completed_after_first(self, two_step_mid_plan):
        result = step_router(two_step_mid_plan)
        assert len(result.completed_steps) == 1


class TestStepRouterError:
    def test_does_not_advance_plan_index_on_error(self, execution_error_state):
        result = step_router(execution_error_state)
        assert result.plan_index == 0

    def test_increments_retry_count_on_error(self, execution_error_state):
        result = step_router(execution_error_state)
        assert result.retry_count == 2

    def test_does_not_append_completed_step_on_error(self, execution_error_state):
        result = step_router(execution_error_state)
        assert len(result.completed_steps) == 0
