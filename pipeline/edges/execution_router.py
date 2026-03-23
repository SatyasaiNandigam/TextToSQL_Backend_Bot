def route_after_execution(state):
    """
    After SQL_EXECUTOR runs:
      - "validate" → SQL executed, proceed to SQL_VALIDATOR
      - "regenerate" → Execution failed due to an error, retry SQL_GENERATOR with feedback
    """
    if state.error:
        if state.retry_count >= 3:
            print("Execution failed after 3 retries. Ending..")
            return "failed" 
        return "next_step"  # retry SQL generation with feedback
    
    if state.plan_index >= len(state.plan):
        print("All plan steps executed. Proceeding to analysis and visualization.")
        return "done"

    print(f"Moving to step {state.plan_index + 1} of {len(state.plan)}...")
    return "next_step"