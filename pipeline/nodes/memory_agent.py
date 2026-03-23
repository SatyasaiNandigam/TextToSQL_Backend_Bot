from langchain_core.messages import AIMessage

def memory_agent(state):
    print("Generating a narrative...")
    
    
    if state.retry_count >= 3:
        return {
        "last_user_query": state.resolved_query or state.current_user_query,
        "last_objective": state.plan[0].objective,
        "last_result_summary": None,
        "last_row_count": None,
        "messages": [AIMessage(content="Not able to answer this question. Maximum try limit reached")]
        }
    
    # 1. Initialize variables with defaults at the start
    objective = "N/A"
    rows = []
    result_summary = "No Results"
    
    last_question = state.resolved_query or state.current_user_query
    
    if state.raw_data:
        objective = "" # Reset if we have data
        if state.completed_steps:
            objective += state.completed_steps[0]['objective']
        
        last_step_key = f"step_{len(state.completed_steps)}"
        rows = state.raw_data.get(last_step_key, [])
        
        if rows:
            cols = list(rows[0].keys())
            result_summary = f"Returned {len(rows)} rows with columns: {', '.join(cols)}."

    # Now these are guaranteed to exist regardless of the IF branches
    assistant_summary = (
        f"Question: {last_question} "
        f"Objective: {objective} "
        f"Result: {result_summary} "
        f"Answered: {state.analysis or 'Query Executed'}"
    ) 
    
    state.last_objective = objective
    state.last_result_summary = assistant_summary
    state.last_row_count = len(rows)
    state.last_user_query = last_question
    
    # Note: Use a dictionary return for LangGraph states rather than mutating 'state' directly
    return {
        "last_user_query": last_question,
        "last_objective": objective,
        "last_result_summary": assistant_summary,
        "last_row_count": len(rows),
        "messages": [AIMessage(content=state.analysis or "Analysis complete.")]
    }
