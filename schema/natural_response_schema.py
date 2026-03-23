from pydantic import BaseModel

class NLPResponseSchema(BaseModel):
    natural_response: str