from core.prompt_registry import prompt_registry
from llm.openai_client import OpenAIClient
from schema.natural_response_schema import NLPResponseSchema

prompt_registry.load_prompts()

llm = OpenAIClient().get_llm()


def analytics_agent(state) :
    # Placeholder for analytics logic
    # This function would perform any necessary analysis based on the generated SQL query and the conversation history
    print("Performing analysis based on the generated SQL query and the conversation history...")
    
    context = state.resolved_query or state.current_user_query

        
    last_sql = state.completed_steps[-1]['sql'] if state.completed_steps else "N/A"
    prompt = prompt_registry.get("nl_response").invoke(input={
        "question": context,
        "sql": last_sql,
        "raw_data": state.raw_data or "N/A"
    })
    
    
    response = llm.with_structured_output(NLPResponseSchema).invoke(prompt)
    
    state.analysis = response.natural_response
    
    return state


if __name__ == '__main__':
    
    from pipeline.nodes.schema_retriver import schema_retriever
    from pipeline.nodes.intent_classifier import intent_classifier
    from pipeline.nodes.query_planner import query_planner
    from pipeline.nodes.sql_agent import sql_agent
    from pipeline.nodes.sql_executor import executor_agent
    from pipeline.nodes.step_router import step_router
    from pipeline.nodes.followup_detector import followup_detector
    from core.state import AgentState
    from langchain_core.messages import HumanMessage

    
    sample_state = AgentState(messages= [HumanMessage(content = "How many orders were placed in each city this month?")])
    
    sample_state = followup_detector(sample_state)

    sample_state = intent_classifier(sample_state)

    sample_state = schema_retriever(sample_state)

    sample_state = query_planner(sample_state)
    
    sample_state = sql_agent(sample_state)
    
    sample_state = executor_agent(sample_state)
    
    sample_state = step_router(sample_state)
    
    print(analytics_agent(sample_state))