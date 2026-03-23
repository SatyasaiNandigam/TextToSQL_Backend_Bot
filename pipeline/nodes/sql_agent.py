from core.prompt_registry import prompt_registry
from llm.openai_client import OpenAIClient
from schema.sql_agent_schema import SQLSchema

prompt_registry.load_prompts()


llm = OpenAIClient().get_llm()


def sql_agent(state):
    
    sql_context = state.plan[state.plan_index]
    
    issue = None
    repair_hint = None
    
    if state.validation_feedback:
        issue = state.validation_feedback["reason"]
        repair_hint = state.validation_feedback["repair_hint"]
        
    if state.error and state.validation_feedback['status'] == 'valid':
        issue = state.error
        repair_hint = "Fix the execution error: {}".format(state.error)
        
    
    
    prompt = prompt_registry.get("sql_agent").invoke(input={
        "question": state.resolved_query or state.current_user_query,
        "table_info": state.table_schema,
        "plan": sql_context,
        "issue": issue,
        "repair_hint": repair_hint
    })
    
    try:
        response = llm.with_structured_output(SQLSchema).invoke(prompt)
        print("sql generated: \n", response)
        state.sql = response.sql
        state.validation_feedback = None
        
    except Exception as e:
        print("exception on sql agent: ",e)
        state.error = str(e)
    
    
    return state


if __name__ == '__main__':
    from pipeline.nodes.schema_retriver import schema_retriever
    from pipeline.nodes.intent_classifier import intent_classifier
    from pipeline.nodes.query_planner import query_planner
    from core.state import AgentState
    from langchain_core.messages import HumanMessage

    
    sample_state = AgentState(messages= [HumanMessage(content = "How many orders were placed in each city this month?")])

    sample_state = intent_classifier(sample_state)

    sample_state = schema_retriever(sample_state)

    sample_state = query_planner(sample_state)
    
    print(sql_agent(sample_state))
