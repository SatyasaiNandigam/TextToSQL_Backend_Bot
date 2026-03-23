from core.state import AgentState
from langgraph.graph import StateGraph, END, START

# nodes
from pipeline.nodes.followup_detector import followup_detector
from pipeline.nodes.orchestrator import orchestrator
from pipeline.nodes.intent_classifier import intent_classifier
from pipeline.nodes.schema_retriver import schema_retriever
from pipeline.nodes.schema_summarizer import schema_summarizer
from pipeline.nodes.query_planner import query_planner
from pipeline.nodes.sql_agent import sql_agent
from pipeline.nodes.validator import validator
from pipeline.nodes.sql_executor import executor_agent
from pipeline.nodes.step_router import step_router
from pipeline.nodes.analytics_reporter import analytics_agent
from pipeline.nodes.memory_agent import memory_agent

# edges
from pipeline.edges.followup_router import route_after_followup
from pipeline.edges.intent_classification_router import route_after_intent_classification
from pipeline.edges.validation_router import route_after_validation
from pipeline.edges.execution_router import route_after_execution

# memory

from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()






graph = StateGraph(AgentState)

graph.add_node("FOLLOW_UP DETECTOR", followup_detector)
graph.add_node("ORCHESTRATOR", orchestrator, description="The state where the agent is orchestrating the conversation and planning its next steps.")
graph.add_node("INTENT_CLASSIFIER", intent_classifier, description="The state where the agent is classifying the user's intent based on the conversation history.")
graph.add_node("SCHEMA RETRIEVER", schema_retriever, description="The state where the agent is retrieving the relevant schema information based on the current plan step objective.")
graph.add_node("SCHEMA SUMMARIZER", schema_summarizer, description="summarize the table schema context from the above retriever.")
graph.add_node("QUERY PLANNER", query_planner, description="The state where the agent is planning the SQL query based on the classified intent and conversation history.")
graph.add_node("SQL AGENT", sql_agent, description="The state where the agent is generating the SQL query.")
graph.add_node("VALIDATOR", validator, description="The state where the agent is validating the SQL query and ensuring they meet the user's needs.")
graph.add_node("EXECUTOR AGENT", executor_agent, description="The state where the agent executes the validated SQL query.")
graph.add_node("STEP ROUTER", step_router, description="The state where the agent routes to the next step in the plan or handles errors after execution.")
graph.add_node("ANALYTICS AGENT", analytics_agent, description="The state where the agent is analyzing the results of the SQL query and generating insights or visualizations.")
# graph.add_node("VISUALIZATION AGENT", visualization_agent, description="The state where the agent is creating visualizations based on the analysis of the SQL query results.")
graph.add_node("MEMORY AGENT", memory_agent, description="The state where the agent is managing the conversation history and any relevant information for future interactions.")
# graph.add_node("INSIGHT AGENT", insight_agent, description="The state where the agent is generating insights and narratives based on the analysis of the SQL query results.")


graph.add_edge(START, "FOLLOW_UP DETECTOR")

graph.add_conditional_edges(
    "FOLLOW_UP DETECTOR",
    route_after_followup,
    {
        "continue": "ORCHESTRATOR",
        "insight": "ANALYTICS AGENT"
    }
)

graph.add_edge("ORCHESTRATOR", "INTENT_CLASSIFIER")

# condition edge if intent classification gives restricted operations intent then go to orchestrator for handling else go to schema retriever
graph.add_conditional_edges(
    "INTENT_CLASSIFIER",
    route_after_intent_classification,
    {
        "query_planner": "SCHEMA RETRIEVER",
        "hard_rule_violation": "MEMORY AGENT"
    }
)


graph.add_edge("SCHEMA RETRIEVER", "SCHEMA SUMMARIZER")
graph.add_edge("SCHEMA SUMMARIZER", "QUERY PLANNER")
graph.add_edge("QUERY PLANNER", "SQL AGENT")
graph.add_edge("SQL AGENT", "VALIDATOR")



# condition edge if validator gives invalid as analysis then go back to sql agent else go to analytics agent
graph.add_conditional_edges(
    "VALIDATOR", 
    route_after_validation, 
    {
        "execute": "EXECUTOR AGENT",
        "regenerate": "SQL AGENT",
        "max_retries": "MEMORY AGENT"
    }
)

graph.add_edge("EXECUTOR AGENT", "STEP ROUTER")

graph.add_conditional_edges(
    "STEP ROUTER",
    route_after_execution,
    {
        "next_step": "SQL AGENT",
        # "regenerate": "SQL AGENT",
        "done": "ANALYTICS AGENT",
        "failed": "MEMORY AGENT"
    }
)

graph.add_edge("ANALYTICS AGENT", "MEMORY AGENT")
graph.add_edge("MEMORY AGENT", END)


app = graph.compile(checkpointer=memory)
