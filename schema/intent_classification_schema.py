from pydantic import BaseModel
from typing import Optional
from enum import Enum


class IntentType(str, Enum):
    KPI_LOOKUP = "kpi_lookup"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    TOP_N = "top_n"
    ANOMALY = "anomaly"
    FORECAST = "forecast"
    SEGMENTATION = "segmentation"
    OPERATIONAL = "operational"
    RESTRICTED_OPERATIONS = "restricted_operations"

class IntentClassifierSchema(BaseModel):
    intent: IntentType
    metric: Optional[str] = None
    group_by : Optional[str] = None
    date_range: Optional[str] = None
    needs_chart: bool