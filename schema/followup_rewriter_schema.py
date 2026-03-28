from pydantic import BaseModel


class FollowUpReWriterSchema(BaseModel):
    rewritten_query: str