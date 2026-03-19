from fastapi import FastAPI

import src.api.services as services
from src.api.schemas import PredictionRequest, PredictionResponse

app = FastAPI(
    title="Mental Health Signal Detector API",
    description="API for detecting mental health signals in text using machine learning models.",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    """Health check endpoint to verify that the API is running."""
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Endpoint to predict mental health signals from input text."""
    result = services.predict(request.text, request.model_type)
    return PredictionResponse(**result)
