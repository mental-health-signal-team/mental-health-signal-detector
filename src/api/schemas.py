from typing import Literal

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    text: str
    model_type: Literal["lr", "distilbert", "mental_roberta", "xgboost"] = "lr"


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    label: int
    probability: float


class ExplainRequest(BaseModel):
    """Request model for explanation endpoint."""

    text: str
    model_type: Literal["lr", "distilbert", "mental_roberta", "xgboost"] = "lr"
    threshold: float = 0.005
    max_tokens: int = 40


class ExplainResponse(BaseModel):
    """Response model for explanation endpoint."""

    label: int
    probability: float
    display_confidence: float
    confidence_label: Literal["distress", "no_distress"]
    risk_level: Literal["low", "medium", "high"]
    colored_html: str
    word_importance: dict[str, float]
    note: str | None = None
