from pydantic import BaseModel
from typing import Literal


class FollowUpSchema(BaseModel):
    is_followup: str
    type: Literal["new", "refine", "regroup", "drilldown", "explain", "visualize"]
    instruction: str