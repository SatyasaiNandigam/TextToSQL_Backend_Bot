from pydantic import BaseModel


class FollowUpReWriterSchema(BaseModel):
    rewrittern_query : str