from pydantic import BaseModel

class ValidationSchema(BaseModel):
    status: str
    failed_check: str
    reason: str
    repair_hint: str