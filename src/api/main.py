from fastapi import FastAPI, HTTPException

import src.api.services as services
from src.api.schemas import ExplainRequest, ExplainResponse, PredictionRequest, PredictionResponse

app = FastAPI(
    title="Mental Health Signal Detector API",
    description="API for detecting mental health signals in text using machine learning models.",
    version="1.0.0",
)


@app.get("/")
def root():
    """Root endpoint with quick API usage hints."""
    return {
        "message": "Mental Health Signal Detector API",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "explain": "POST /explain",
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Endpoint to predict mental health signals from input text."""
    result = services.predict(request.text, request.model_type)
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
