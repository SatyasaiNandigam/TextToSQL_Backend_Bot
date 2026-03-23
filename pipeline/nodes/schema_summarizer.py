# from llm.groq_client import GroqClient
from llm.openai_client import OpenAIClient
from core.prompt_registry import prompt_registry
llm = OpenAIClient().get_llm()




def schema_summarizer(state):
    print("Summarizing the Schema")
    schema_context = state.table_schema
    question = state.resolved_query or state.current_user_query
    
    prompt = prompt_registry.get("schema_summarizer").invoke(input={
        "table_schema": schema_context,
        "user_question": question
    })
    
    try:
        response = llm.invoke(prompt)
        print("summairzed context: ", response)
        state.table_schema = response.content
        
        return state
        
    except Exception as e:
        state.error = str(e)
        
    
    
    
if __name__ == '__main__':
    from core.prompt_registry import prompt_registry
    from schema_rag_pipeline import init_embeddings
    from langchain_core.messages import HumanMessage
    from core.state import AgentState
    from pipeline.nodes.schema_retriver import schema_retriever
    prompt_registry.load_prompts()
    query = "How many orders were placed in each city this month?"
    init_embeddings()
    sample_state = AgentState(messages= [HumanMessage(content = query )], current_user_query= query )
    sample_state = schema_retriever(sample_state)
    
    schema_summarizer(sample_state)
    