"""
Integration tests for the sql_executor (executor_agent) node.
Requires a live PostgreSQL connection via READ_DATABASE_URL.

Run:
    pytest tests/integration/ -v -m integration -k "sql_executor"
"""
import pytest
from langchain_core.messages import HumanMessage
from core.state import AgentState
from schema.planner_schema import PlanStep
from pipeline.nodes.sql_executor import executor_agent


def _make_state(sql: str, plan_index: int = 0) -> AgentState:
    return AgentState(
        messages=[HumanMessage(content="test query")],
        current_user_query="test query",
        plan=[PlanStep(step_number=plan_index + 1, objective="test objective")],
        plan_index=plan_index,
        sql=sql,
        raw_data=None,
    )


@pytest.mark.integration
class TestSQLExecutorSuccess:

    def test_simple_count_returns_rows(self):
        result = executor_agent(_make_state("SELECT COUNT(*) AS total_orders FROM orders"))
        assert result.error is None
        assert result.sql_executed is True
        rows = result.raw_data["step_1"]
        assert len(rows) == 1
        assert rows[0]["total_orders"] > 0

    def test_group_by_returns_multiple_rows(self):
        sql = (
            "SELECT u.city, COUNT(o.order_id) AS order_count "
            "FROM orders o JOIN users u ON o.user_id = u.user_id "
            "GROUP BY u.city ORDER BY order_count DESC LIMIT 5"
        )
        result = executor_agent(_make_state(sql))
        assert result.error is None
        rows = result.raw_data["step_1"]
        assert len(rows) >= 1
        assert "city" in rows[0]
        assert "order_count" in rows[0]

    def test_revenue_aggregation_columns_present(self):
        sql = (
            "SELECT c.name AS category, SUM(oi.total_price) AS total_revenue "
            "FROM orders o "
            "JOIN order_items oi ON o.order_id = oi.order_id "
            "JOIN product_variants pv ON oi.variant_id = pv.variant_id "
            "JOIN products p ON pv.product_id = p.product_id "
            "JOIN categories c ON p.category_id = c.category_id "
            "GROUP BY c.name ORDER BY total_revenue DESC LIMIT 5"
        )
        result = executor_agent(_make_state(sql))
        assert result.error is None
        rows = result.raw_data["step_1"]
        assert "category" in rows[0]
        assert "total_revenue" in rows[0]

    def test_sql_executed_true_on_success(self):
        result = executor_agent(_make_state("SELECT 1 AS ping"))
        assert result.sql_executed is True
        assert result.error is None

    def test_raw_data_step_key_matches_plan_index(self):
        result = executor_agent(_make_state("SELECT COUNT(*) AS c FROM orders", plan_index=0))
        assert "step_1" in result.raw_data

    def test_limit_respected(self):
        result = executor_agent(_make_state("SELECT order_id FROM orders LIMIT 5"))
        assert result.error is None
        assert len(result.raw_data["step_1"]) == 5


@pytest.mark.integration
class TestSQLExecutorErrorHandling:

    def test_syntax_error_sets_error_field(self):
        result = executor_agent(_make_state("SELECT * FORM orders"))
        assert result.error is not None

    def test_syntax_error_sets_sql_executed_false(self):
        result = executor_agent(_make_state("SELECT * FORM orders"))
        assert not result.sql_executed

    def test_nonexistent_table_sets_error(self):
        result = executor_agent(_make_state("SELECT * FROM nonexistent_table_xyz"))
        assert result.error is not None

    def test_nonexistent_column_sets_error(self):
        result = executor_agent(_make_state("SELECT orderDate FROM orders"))
        assert result.error is not None

    def test_error_does_not_corrupt_raw_data(self):
        result = executor_agent(_make_state("SELECT * FORM orders"))
        assert result.raw_data is None or "step_1" not in (result.raw_data or {})


@pytest.mark.integration
class TestSQLExecutorReadOnly:

    def test_insert_rejected_by_db(self):
        result = executor_agent(_make_state("INSERT INTO orders (order_id) VALUES (999999)"))
        assert result.error is not None

    def test_delete_rejected_by_db(self):
        result = executor_agent(_make_state("DELETE FROM orders WHERE 1=0"))
        assert result.error is not None
