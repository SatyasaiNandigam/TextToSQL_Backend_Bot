from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pipeline.graph import app
from core.state import AgentState
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from typing import Optional
import json
import warnings
import time
from contextlib import asynccontextmanager
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from schema_rag_pipeline import init_embeddings


@asynccontextmanager
async def lifespan(api_app: FastAPI):
    """Initialize resources at startup."""
    from core.prompt_registry import prompt_registry
    prompt_registry.load_prompts()
    init_embeddings()
    yield
    # Cleanup if needed


api = FastAPI(title="Ecom ChatBot Agent API", lifespan=lifespan)

# Add CORS middleware to allow requests from any origin
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class AgentRequest(BaseModel):
    thread_id: int
    user_query: str


class AgentResponse(BaseModel):
    thread_id: int
    user_query: str
    agent_output: str


class MessagesResponse(BaseModel):
    thread_id: int
    messages: list[dict]


class SessionsResponse(BaseModel):
    sessions: list[int]


@api.post("/invoke", response_model=AgentResponse)
async def invoke_agent(payload: AgentRequest):
    if not payload.user_query.strip():
        raise HTTPException(status_code=400, detail="user_query must be a non-empty string")

    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}
    state = AgentState(
        messages=[HumanMessage(content=payload.user_query)],
        current_user_query=payload.user_query,
    )

    response = app.invoke(state, config=config)

    # If your app.invoke returns an object, adjust this accordingly.
    agent_output = str(response)

    return AgentResponse(
        thread_id=payload.thread_id,
        user_query=payload.user_query,
        agent_output=agent_output,
    )


def _sse_format_event(data: str, event: str = "message", event_id: Optional[str] = None) -> str:
    payload = ""
    if event_id is not None:
        payload += f"id: {event_id}\n"
    payload += f"event: {event}\n"
    for line in data.splitlines():
        payload += f"data: {line}\n"
    payload += "\n"
    return payload


@api.post("/invoke/stream")
async def invoke_agent_stream(payload: AgentRequest):
    if not payload.user_query.strip():
        raise HTTPException(status_code=400, detail="user_query must be a non-empty string")

    config: RunnableConfig = {"configurable": {"thread_id": payload.thread_id}}
    initial_state = AgentState(
        messages=[HumanMessage(content=payload.user_query)],
        current_user_query=payload.user_query,
    )

    async def event_generator():
        # Use astream to get real updates from LangGraph
        # 'stream_mode="values"' sends the full state after each node completes
        # 'stream_mode="updates"' sends only what changed in that step
        counter = 0
        async for event in app.astream(initial_state, config=config, stream_mode="updates"):
            # Convert the event dict to a clean JSON string
            data_str = json.dumps(event, default=str)
            
            yield _sse_format_event(
                data=data_str, 
                event="message", 
                event_id=str(counter)
            )
            counter += 1

        yield _sse_format_event("[DONE]", event="done", event_id="done")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@api.get("/messages/{thread_id}", response_model=MessagesResponse)
async def get_messages(thread_id: int):
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    try:
        state = app.get_state(config)
        messages = [msg.dict() for msg in state.values.get("messages", [])] if state.values else []
        return MessagesResponse(thread_id=thread_id, messages=messages)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"No state found for thread_id {thread_id}: {str(e)}")


@api.get("/sessions", response_model=SessionsResponse)
async def get_sessions():
    """Get all available session (thread) IDs."""
    try:
        # Access the checkpointer's storage to get all thread IDs
        thread_ids = list(app.checkpointer.storage.keys())
        # Convert string keys to int if needed (assuming thread_ids are integers)
        sessions = [int(tid) for tid in thread_ids if tid.isdigit()]
        return SessionsResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sessions: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(api, host="0.0.0.0", port=8000)
