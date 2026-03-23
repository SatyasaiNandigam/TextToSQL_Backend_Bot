from sqlalchemy import create_engine
import pandas as pd
from core.settings import settings

engine = create_engine(url= settings.READ_DATABASE_URL)


def executor_agent(state):
    print("executing sql query...")
    
    try:
        data = pd.read_sql_query(state.sql, con=engine)
        
        if state.raw_data is None:
            state.raw_data = {}
            
        # Now you can safely assign the key-value pair
        state.raw_data[f"step_{state.plan_index + 1}"] = data.to_dict(orient='records')
        state.sql_executed = True
        state.error = None
        
    except Exception as e:
        print("Got exception in executor agent: ",e)
        state.error = str(e)
        state.sql_executed = False
        
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
    
    print(executor_agent(sample_state))