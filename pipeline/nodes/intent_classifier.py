from core.prompt_registry import prompt_registry
from schema.intent_classification_schema import IntentClassifierSchema
from llm.groq_client import GroqClient
from langchain_core.messages import HumanMessage

from core.state import AgentState



prompt_registry.load_prompts()

llm = GroqClient().get_llm()


def intent_classifier(state):
    
    prompt = prompt_registry.get("intent_classifier").invoke(input = {
        "query": state.resolved_query or state.current_user_query
    })
    
    
    response = llm.with_structured_output(IntentClassifierSchema).invoke(prompt)
    print("intent response: ", response)
    
    state.intent_result = response
    
    return state
    
    
    
if __name__ == "__main__":
    sample_state = AgentState(messages= [HumanMessage(content = "How many orders were placed in each city this month?")])
    # sample_state = {
    #     "messages": [({"role": "user", "content": "How many orders were placed in each city this month?"})]
    # }
    print(intent_classifier(sample_state) )
    