def step_router(state):
    """
    Pure bookkeeping node — advances plan_index and resets
    """
    if not state.error:
        print(f"Step {state.plan_index + 1} complete. Saving and advancing...")
        
        step_entry = {
            "step_number": state.plan_index + 1,
            "objective":   state.plan[state.plan_index].objective,
            "sql":         state.sql,
            "columns":     [],
            "sample_rows": []
        }
        
        step_key = f"step_{state.plan_index + 1}"
        if state.raw_data and step_key in state.raw_data:
            rows = state.raw_data[step_key]          # list of dicts from executor
            step_entry["sample_rows"] = rows[:3]     # keep only first 3 rows
            step_entry["columns"]     = list(rows[0].keys()) if rows else []
        
        if not hasattr(state, 'completed_steps') or state.completed_steps is None:
            state.completed_steps = []
            
        state.completed_steps.append(step_entry)
        
        print("Complted steps:",state.completed_steps)
        
        state.plan_index += 1
        state.sql = None
        state.sql_executed = False
        state.validation_feedback = None
        state.retry_count = 0
    else:
        state.retry_count +=1
        print(f"Executor error (attempt {state.retry_count}/3): {state.error}")


    return state


if __name__ == '__main__':
    from pipeline.nodes.schema_retriver import schema_retriever
    from pipeline.nodes.intent_classifier import intent_classifier
    from pipeline.nodes.query_planner import query_planner
    from pipeline.nodes.sql_agent import sql_agent
    from pipeline.nodes.sql_executor import executor_agent
    from core.state import AgentState
    from langchain_core.messages import HumanMessage

    
    sample_state = AgentState(messages= [HumanMessage(content = "How many orders were placed in each city this month?")])

    sample_state = intent_classifier(sample_state)

    sample_state = schema_retriever(sample_state)

    sample_state = query_planner(sample_state)
    
    sample_state = sql_agent(sample_state)
    
    sample_state = executor_agent(sample_state)
    
    print(step_router(sample_state))
