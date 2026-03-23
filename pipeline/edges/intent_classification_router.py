def route_after_intent_classification(state):
    """
    Decide the next action based on the intent classification result and any errors.
    Possible return values:
      - "query_planner" → proceed to query planning
      - "orchestrator" → route to orchestrator for handling (e.g. for restricted operations or errors)
    """
    if state.error == "restricted_operations_intent":
        return "hard_rule_violation"
    return "query_planner"