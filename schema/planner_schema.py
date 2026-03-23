from pydantic import BaseModel, Field
from typing import List


class PlanStep(BaseModel):
    step_number: int
    objective: str = Field(description="the goal of this specific step")

class PlannerSchema(BaseModel):
    plan: List[PlanStep]
    
