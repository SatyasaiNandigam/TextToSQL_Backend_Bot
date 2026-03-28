from pipeline.nodes.orchestrator import orchestrator
from tests.fixtures.agent_states import make_dirty_state, make_fresh_query_state


class TestOrchestratorResets:
    """Orchestrator must wipe all per-turn pipeline fields on every invocation."""

    def test_resets_plan(self):
        result = orchestrator(make_dirty_state())
        assert result.plan == []

    def test_resets_plan_index(self):
        result = orchestrator(make_dirty_state())
        assert result.plan_index == 0

    def test_resets_completed_steps(self):
        result = orchestrator(make_dirty_state())
        assert result.completed_steps == []

    def test_resets_sql(self):
        result = orchestrator(make_dirty_state())
        assert result.sql is None

    def test_resets_sql_executed(self):
        result = orchestrator(make_dirty_state())
        assert result.sql_executed is False

    def test_resets_raw_data(self):
        result = orchestrator(make_dirty_state())
        assert result.raw_data is None

    def test_resets_validation_feedback(self):
        result = orchestrator(make_dirty_state())
        assert result.validation_feedback is None

    def test_resets_error(self):
        state = make_dirty_state()
        state.error = "some previous error"
        result = orchestrator(state)
        assert result.error is None

    def test_resets_retry_count(self):
        result = orchestrator(make_dirty_state())
        assert result.retry_count == 0

    def test_resets_table_schema(self):
        result = orchestrator(make_dirty_state())
        assert result.table_schema is None

    def test_resets_analysis(self):
        result = orchestrator(make_dirty_state())
        assert result.analysis is None

    def test_resets_narrative(self):
        result = orchestrator(make_dirty_state())
        assert result.narrative is None

    def test_resets_chart_base64(self):
        result = orchestrator(make_dirty_state())
        assert result.chart_base64 is None


class TestOrchestratorPreserves:
    """Orchestrator must not overwrite cross-turn context fields."""

    def test_preserves_current_user_query(self):
        state = make_dirty_state("What is the revenue?")
        result = orchestrator(state)
        assert result.current_user_query == "What is the revenue?"

    def test_preserves_resolved_query(self):
        state = make_dirty_state()
        state.resolved_query = "standalone version of query"
        result = orchestrator(state)
        assert result.resolved_query == "standalone version of query"

    def test_idempotent_on_already_clean_state(self):
        state = make_fresh_query_state()
        result = orchestrator(state)
        assert result.plan == []
        assert result.plan_index == 0
        assert result.retry_count == 0
