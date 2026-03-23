from llm.openai_client import OpenAIClient
from core.prompt_registry import prompt_registry
from schema.validation_schema import ValidationSchema

prompt_registry.load_prompts()

llm = OpenAIClient().get_llm()


def hard_rule_check(sql: str) -> dict:
    banned = ["DELETE", "DROP", "INSERT", "UPDATE", "TRUNCATE", "GRANT", "ALTER"]
    for kw in banned:
        if kw in sql.upper().split():
            return {"status": "invalid", "reason": f"Banned keyword: {kw}", "repair_hint": "Remove destructive SQL operation"}
    return {"status": "safe", "reason": "OK", "repair_hint": ""}



def validator(state) :
    # Placeholder for validation logic
    # This function would validate the generated SQL query and ensure it meets certain criteria
    print("Validating the generated SQL query...")
    
    objective = state.plan[state.plan_index].objective
    sql = state.sql
    table_schemas = state.table_schema
    
    
    # guard against non read queries making it to execution
    hard_check = hard_rule_check(sql)
    if hard_check["status"] == "invalid":
        print("HARD RULE VIOLATION: ", hard_check["reason"])
        state.validation_feedback = {"status": "invalid", "failed_check": "HARD RULE CHECK", "reason": hard_check["reason"], "repair_hint": hard_check["repair_hint"]}
        state.error = "hard_rule_violation"
        return state
    
    prompt = prompt_registry.get("sql_validaton").invoke(input={
        "question": state.resolved_query or state.current_user_query,
        "objective": objective,
        "sql": sql,
        "table_schema": table_schemas
    })
    
    try:
        response = llm.with_structured_output(ValidationSchema).invoke(prompt)
        print("validation result: ", response)
        
        state.validation_feedback = response.model_dump()
        
        if state.validation_feedback["status"] == "invalid":
            # increase the retry count to trigger regeneration with feedback
            state.retry_count += 1
        
    
    except Exception as e:
        print("Error validating query: ",e)
        state.validation_feedback = {"status": "valid", "failed_check": "",
                                     "reason": "Validator error — assuming valid.",
                                     "repair_hint": ""}
        

    return state


if __name__ == '__main__':
    
    from pipeline.nodes.schema_retriver import schema_retriever
    from pipeline.nodes.intent_classifier import intent_classifier
    from pipeline.nodes.query_planner import query_planner
    from pipeline.nodes.sql_agent import sql_agent
    from core.state import AgentState
    from langchain_core.messages import HumanMessage

    
    sample_state = AgentState(messages= [HumanMessage(content = "How many orders were placed in each city this month?")])

    sample_state = intent_classifier(sample_state)

    sample_state = schema_retriever(sample_state)

    sample_state = query_planner(sample_state)
    
    sample_state = sql_agent(sample_state)
    
    print(validator(sample_state))