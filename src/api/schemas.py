from typing import Literal

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""

    text: str
    model_type: Literal["lr", "distilbert", "mental_roberta", "mentalbert", "xgboost"] = "lr"


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    label: int
    probability: float


class ExplainRequest(BaseModel):
    """Request model for explanation endpoint."""

    text: str
    model_type: Literal["lr", "distilbert", "mental_roberta", "mentalbert", "xgboost"] = "lr"
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


class DayCount(BaseModel):
    """Prediction count for a single day."""

    date: str
    count: int


class StatsResponse(BaseModel):
    """Response model for GET /stats endpoint."""

    total_predictions: int
    distress_count: int
    no_distress_count: int
    risk_level_counts: dict[str, int]
    model_usage: dict[str, int]
    predictions_by_day: list[DayCount]
    avg_confidence: float
    distress_by_model: dict[str, int]
