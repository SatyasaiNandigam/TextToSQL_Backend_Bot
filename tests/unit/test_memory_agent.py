import pytest
from langchain_core.messages import AIMessage
from pipeline.nodes.memory_agent import memory_agent
from tests.fixtures.agent_states import make_memory_agent_state, make_plan


class TestMemoryAgentNormalFlow:
    def test_returns_dict(self):
        result = memory_agent(make_memory_agent_state())
        assert isinstance(result, dict)

    def test_sets_last_user_query(self):
        result = memory_agent(make_memory_agent_state(query="Orders per city?"))
        assert result["last_user_query"] == "Orders per city?"

    def test_uses_resolved_query_when_present(self):
        state = make_memory_agent_state(query="these ones")
        state.resolved_query = "standalone resolved query"
        result = memory_agent(state)
        assert result["last_user_query"] == "standalone resolved query"

    def test_sets_last_objective_from_first_completed_step(self):
        result = memory_agent(make_memory_agent_state())
        assert result["last_objective"] == "Count orders by city"

    def test_last_result_summary_contains_question(self):
        result = memory_agent(make_memory_agent_state(query="Orders per city?"))
        assert "Orders per city?" in result["last_result_summary"]

    def test_last_result_summary_contains_objective(self):
        result = memory_agent(make_memory_agent_state())
        assert "Count orders by city" in result["last_result_summary"]

    def test_last_result_summary_contains_analysis(self):
        result = memory_agent(make_memory_agent_state(analysis="Mumbai had the most orders."))
        assert "Mumbai had the most orders." in result["last_result_summary"]

    def test_last_row_count_matches_raw_data(self):
        raw_data = {"step_1": [{"city": "Mumbai", "count": 5}, {"city": "Delhi", "count": 3}]}
        result = memory_agent(make_memory_agent_state(raw_data=raw_data))
        assert result["last_row_count"] == 2

    def test_messages_contains_ai_message(self):
        result = memory_agent(make_memory_agent_state())
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)

    def test_ai_message_content_is_analysis(self):
        result = memory_agent(make_memory_agent_state(analysis="Revenue is 50K"))
        assert result["messages"][0].content == "Revenue is 50K"

    def test_ai_message_fallback_when_no_analysis(self):
        result = memory_agent(make_memory_agent_state(analysis=None))
        assert result["messages"][0].content == "Analysis complete."


class TestMemoryAgentNoRawData:
    def test_handles_none_raw_data(self):
        state = make_memory_agent_state(raw_data=None, completed_steps=[])
        result = memory_agent(state)
        assert isinstance(result, dict)

    def test_last_row_count_zero_when_no_data(self):
        state = make_memory_agent_state(raw_data=None, completed_steps=[])
        result = memory_agent(state)
        assert result["last_row_count"] == 0

    def test_last_result_summary_no_results_when_empty(self):
        state = make_memory_agent_state(raw_data=None, completed_steps=[])
        result = memory_agent(state)
        assert "No Results" in result["last_result_summary"]


class TestMemoryAgentMaxRetries:
    def test_returns_error_message_at_max_retries(self):
        state = make_memory_agent_state(retry_count=3)
        result = memory_agent(state)
        assert "Maximum try limit reached" in result["messages"][0].content

    def test_last_result_summary_is_none_at_max_retries(self):
        state = make_memory_agent_state(retry_count=3)
        result = memory_agent(state)
        assert result["last_result_summary"] is None

    def test_last_row_count_is_none_at_max_retries(self):
        state = make_memory_agent_state(retry_count=3)
        result = memory_agent(state)
        assert result["last_row_count"] is None

    def test_still_sets_last_user_query_at_max_retries(self):
        state = make_memory_agent_state(query="Failing query", retry_count=3)
        result = memory_agent(state)
        assert result["last_user_query"] == "Failing query"

    def test_max_retries_triggers_at_exactly_3(self):
        state_ok = make_memory_agent_state(retry_count=2)
        state_fail = make_memory_agent_state(retry_count=3)
        result_ok = memory_agent(state_ok)
        result_fail = memory_agent(state_fail)
        assert "Maximum try limit reached" not in result_ok["messages"][0].content
        assert "Maximum try limit reached" in result_fail["messages"][0].content
