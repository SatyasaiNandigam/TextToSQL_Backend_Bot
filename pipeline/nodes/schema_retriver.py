from schema_rag_pipeline import schema_extractor, retrieve_and_expand, init_embeddings
from core.state import AgentState
from langchain_core.messages import HumanMessage


def schema_retriever(state):
    question = state.resolved_query or state.current_user_query
    
    schema_info = schema_extractor()
    context, tables = retrieve_and_expand(question, schema_info)
    
    state.table_schema = context
    
    return state



if __name__ == '__main__':

    # sample_state = {
    #     "messages": [({"role": "user", "content": "How many orders were placed in each city this month?"})],
    # }
    query = "How many orders were placed in each city this month?"
    init_embeddings()
    sample_state = AgentState(messages= [HumanMessage(content = query )], current_user_query= query )
    print(schema_retriever(sample_state))
