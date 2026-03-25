from fastapi import FastAPI
from fastapi import HTTPException

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


@app.get("/diagnostics/roberta")
def roberta_diagnostics(load_model: bool = False):
    """Report which RoBERTa backend is active: local files or pickle."""
    try:
        return services.get_roberta_diagnostics(load_model=load_model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"RoBERTa diagnostics failed: {exc}") from exc


@app.get("/diagnostics/distilbert")
def distilbert_diagnostics(load_model: bool = False):
    """Report which DistilBERT backend is active: local files or pickle."""
    try:
        return services.get_distilbert_diagnostics(load_model=load_model)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"DistilBERT diagnostics failed: {exc}") from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> PredictionResponse:
    """Endpoint to predict mental health signals from input text."""
    try:
        result = services.predict(request.text, request.model_type)
        return PredictionResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifact missing: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed ({request.model_type}): {exc}") from exc
