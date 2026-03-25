from typing import Literal

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    text: str
    model_type: Literal["lr", "distilbert", "roberta", "xgboost"] = "lr"


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    label: int
    probability: float
    source_language: str = "unknown"
    analysis_language: str = "en"
    was_translated: bool = False
    translated_text: str | None = None
