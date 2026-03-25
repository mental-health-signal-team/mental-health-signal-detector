from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

import src.api.services as services
from src.api.database import get_stats, init_db, log_prediction
from src.api.schemas import ExplainRequest, ExplainResponse, PredictionRequest, PredictionResponse, StatsResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise DB tables on startup."""
    init_db()
    yield


app = FastAPI(
    title="Mental Health Signal Detector API",
    description="API for detecting mental health signals in text using machine learning models.",
    version="1.0.0",
    lifespan=lifespan,
)

_RISK_THRESHOLDS = (0.33, 0.66)


def _risk_level(probability: float) -> str:
    if probability < _RISK_THRESHOLDS[0]:
        return "low"
    if probability < _RISK_THRESHOLDS[1]:
        return "medium"
    return "high"


@app.get("/")
def root():
    """Root endpoint with quick API usage hints."""
    return {
        "message": "Mental Health Signal Detector API",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "explain": "POST /explain",
            "stats": "GET /stats",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Endpoint to predict mental health signals from input text."""
    result = services.predict(request.text, request.model_type)
    background_tasks.add_task(
        log_prediction,
        request.text,
        request.model_type,
        result["label"],
        result["probability"],
        _risk_level(result["probability"]),
    )
    return PredictionResponse(**result)


@app.post("/explain")
def explain(request: ExplainRequest) -> ExplainResponse:
    """Endpoint to predict and return word-level explanation details."""
    try:
        result = services.explain(
            text=request.text,
            model_type=request.model_type,
            threshold=request.threshold,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ExplainResponse(**result)


@app.get("/stats")
def stats() -> StatsResponse:
    """Return aggregated prediction statistics from the database."""
    return StatsResponse(**get_stats())
