from llm.groq_client import GroqClient
from schema.follow_up_schema import FollowUpSchema
from core.prompt_registry import prompt_registry

prompt_registry.load_prompts()

llm = GroqClient().get_llm()



def followup_detector(state):
    user = state.current_user_query
    print("last objective: ", state.last_objective)
    print("last summary: ", state.last_result_summary)
    
    prompt = prompt_registry.get("followup_detector").invoke({
        "user": user,
        "prior_objective": state.last_objective or "N/A",
        "prior_summary": state.last_result_summary or "N/A",
    })
    dec = llm.with_structured_output(FollowUpSchema).invoke(prompt)
    print("follow up status: ", dec)
    state.followup = dec
    
    return state


if __name__ == '__main__':
    pass

