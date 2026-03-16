from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.schemas import ExplainRequest, ExplainResponse, HealthResponse, PredictRequest, PredictResponse
from src.api.services import run_explain, run_prediction
from src.api.dependencies import get_model
from src.api.checkin_router import router as checkin_router
from src.common.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pré-chargement eager du modèle baseline pour réduire la latence de la
    # première requête. On tolère l'absence du fichier (CI, fresh-clone) :
    # les endpoints gèrent eux-mêmes l'indisponibilité avec un 503.
    try:
        logger.info("Démarrage de l'API — chargement du modèle baseline...")
        get_model("baseline")
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Modèle baseline introuvable au démarrage (ignoré) : {e}")
    yield
    logger.info("Arrêt de l'API.")


_settings = get_settings()

# En production, restreindre à l'origine du dashboard
_ALLOWED_ORIGINS = (
    ["*"]
    if _settings.env == "development"
    else ["http://localhost:8501", "http://dashboard:8501"]
)

app = FastAPI(
    title="Mental Health Signal Detector API",
    description="Détecte des signaux de détresse mentale dans un texte via NLP.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(checkin_router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        get_model("baseline")
        return HealthResponse(status="ok", model_loaded=True)
    except Exception:
        return HealthResponse(status="ok", model_loaded=False)


@app.post("/explain", response_model=ExplainResponse)
def explain_endpoint(request: ExplainRequest):
    """Retourne les contributions SHAP des mots les plus influents (baseline uniquement)."""
    try:
        model = get_model("baseline")
        return run_explain(request, model)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Modèle indisponible (explain) : {e}")
        raise HTTPException(status_code=503, detail="Modèle non disponible — entraînement requis.")
    except Exception as e:
        logger.error(f"Erreur explain : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne.")


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest):
    try:
        model = get_model(request.model_type)
        return run_prediction(request, model)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Modèle indisponible : {e}")
        raise HTTPException(status_code=503, detail="Modèle non disponible — entraînement requis.")
    except Exception as e:
        logger.error(f"Erreur prédiction : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne.")
