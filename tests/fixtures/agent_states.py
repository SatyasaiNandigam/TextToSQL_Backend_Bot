"""
Factory functions that build minimal AgentState instances for each node under test.
Only the fields each node actually reads are populated; everything else uses defaults.
"""
from langchain_core.messages import HumanMessage
from core.state import AgentState
from schema.intent_classification_schema import IntentClassifierSchema, IntentType
from schema.planner_schema import PlanStep
from schema.follow_up_schema import FollowUpSchema


def make_plan(*objectives: str) -> list[PlanStep]:
    return [PlanStep(step_number=i + 1, objective=obj) for i, obj in enumerate(objectives)]


def make_fresh_query_state(query: str = "How many orders were placed in each city?") -> AgentState:
    """Minimal state for orchestrator and intent_classifier."""
    return AgentState(
        messages=[HumanMessage(content=query)],
        current_user_query=query,
    )


def make_dirty_state(query: str = "How many orders per city?") -> AgentState:
    """State with all pipeline fields populated — used to verify orchestrator resets them."""
    plan = make_plan("Count orders by city")
    return AgentState(
        messages=[HumanMessage(content=query)],
        current_user_query=query,
        resolved_query=query,
        plan=plan,
        plan_index=1,
        completed_steps=[{"step_number": 1, "objective": "Count orders by city", "sql": "SELECT 1", "columns": [], "sample_rows": []}],
        sql="SELECT city, COUNT(*) FROM orders GROUP BY city",
        sql_executed=True,
        raw_data={"step_1": [{"city": "Mumbai", "count": 5}]},
        validation_feedback={"status": "valid", "failed_check": "", "reason": "", "repair_hint": ""},
        error=None,
        retry_count=2,
        table_schema="Table: orders ...",
        analysis="Mumbai had the most orders.",
        narrative="Some narrative.",
        chart_base64="base64string",
    )


def make_post_execution_state(
    query: str,
    plan: list[PlanStep],
    sql: str,
    raw_data: dict | None,
    error: str | None = None,
    retry_count: int = 0,
) -> AgentState:
    """State after executor_agent — used by step_router and analytics_agent tests."""
    return AgentState(
        messages=[HumanMessage(content=query)],
        current_user_query=query,
        resolved_query=query,
        plan=plan,
        plan_index=0,
        completed_steps=[],
        sql=sql,
        sql_executed=True if not error else False,
        raw_data=raw_data,
        error=error,
        retry_count=retry_count,
    )


def make_memory_agent_state(
    query: str = "Orders per city?",
    plan: list[PlanStep] | None = None,
    completed_steps: list[dict] | None = None,
    raw_data: dict | None = None,
    analysis: str | None = "Mumbai had 5 orders.",
    retry_count: int = 0,
) -> AgentState:
    """Minimal state for memory_agent tests."""
    if plan is None:
        plan = make_plan("Count orders by city")
    if completed_steps is None:
        completed_steps = [
            {
                "step_number": 1,
                "objective": "Count orders by city",
                "sql": "SELECT city, COUNT(*) FROM orders GROUP BY city",
                "columns": ["city", "count"],
                "sample_rows": [{"city": "Mumbai", "count": 5}],
            }
        ]
    if raw_data is None:
        raw_data = {"step_1": [{"city": "Mumbai", "count": 5}]}
    return AgentState(
        messages=[HumanMessage(content=query)],
        current_user_query=query,
        resolved_query=query,
        plan=plan,
        plan_index=len(completed_steps),
        completed_steps=completed_steps,
        raw_data=raw_data,
        analysis=analysis,
        retry_count=retry_count,
    )


def make_followup_state(
    current_query: str,
    is_followup: str = "True",
    followup_type: str = "refine",
    instruction: str = "filter by Electronics",
) -> AgentState:
    """State for rewriter_router and followup_router tests."""
    return AgentState(
        messages=[HumanMessage(content=current_query)],
        current_user_query=current_query,
        followup=FollowUpSchema(
            is_followup=is_followup,
            type=followup_type,
            instruction=instruction,
        ),
    )
