from core.prompt_registry import prompt_registry
from schema.planner_schema import PlannerSchema
from llm.openai_client import OpenAIClient

llm = OpenAIClient().get_llm()

prompt_registry.load_prompts()

def query_planner(state):
    
    question = state.resolved_query or state.current_user_query
    
    prompt = prompt_registry.get("query_planner").invoke(input={
        "query": question,
        "intent_result": state.intent_result,
        "table_info": state.table_schema
    })
    
    try:
        response = llm.with_structured_output(PlannerSchema).invoke(prompt)
        print("Query planner: ", response)
        
        state.plan = response.plan
        state.plan_index = 0
        
    except Exception as e:
        state['error'] = str(e)
        
    return state



if __name__ == '__main__':
    from pipeline.nodes.schema_retriver import schema_retriever
    from pipeline.nodes.intent_classifier import intent_classifier
    from core.state import AgentState
    from langchain_core.messages import HumanMessage

    
    sample_state = AgentState(messages= [HumanMessage(content = "How many orders were placed in each city this month?")])

    sample_state = intent_classifier(sample_state)

    sample_state = schema_retriever(sample_state)

    print(query_planner(sample_state))

