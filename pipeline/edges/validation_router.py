def route_after_validation(state):
    """
    After SQL_VALIDATOR runs:
      - "execute"     → SQL is valid, proceed to SQL_EXECUTOR
      - "regenerate"  → SQL has fixable issues, retry SQL_GENERATOR
      - "max_retries" → Too many retries, escalate to ORCHESTRATOR
    """
    
    if state.error == "hard_rule_violation":
        return "max_retries"

    feedback = state.validation_feedback or {}
    
    if feedback.get("status") == "restricted":
        print("SQL contains restricted operations.")
        return "restricted"
    
    if feedback.get("status") == "valid":
        return "execute"
    
    if state.retry_count >= 3:
        print("Max retries reached. Escalating to orchestrator for fallback handling.")
        return "max_retries"
    
    print(f"Validation failed: {feedback.get('reason')}. Attempting to regenerate SQL (Retry {state.retry_count}/3).")
    return "regenerate"