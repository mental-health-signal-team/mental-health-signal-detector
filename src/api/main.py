from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.rate_limit import limiter

from src.api.schemas import ExplainRequest, ExplainResponse, HealthResponse, PredictRequest, PredictResponse
from src.api.checkin_router import router as checkin_router
from src.api.solutions_router import router as solutions_router
from src.api.analyze_router import router as analyze_router
from src.common.config import get_settings

# Imports ML optionnels — absents dans le déploiement slim (sans modèle)
try:
    from src.api.services import run_explain, run_prediction
    from src.api.dependencies import get_model
    _ML_AVAILABLE = True
except ImportError as _e:
    logger.warning(f"Packages ML indisponibles — endpoints /predict et /explain désactivés : {_e}")
    _ML_AVAILABLE = False
    get_model = None  # type: ignore[assignment]
    run_explain = None  # type: ignore[assignment]
    run_prediction = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _ML_AVAILABLE:
        try:
            logger.info("Démarrage de l'API — chargement du modèle baseline...")
            get_model("baseline")
        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Modèle baseline introuvable au démarrage (ignoré) : {e}")
    else:
        logger.info("Démarrage de l'API — mode slim (sans modèle ML).")
    yield
    logger.info("Arrêt de l'API.")


_settings = get_settings()

# Origines CORS autorisées.
# En dev : "*" pour simplifier.
# En prod : lire ALLOWED_ORIGINS depuis l'env (liste séparée par des virgules).
#   Ex: ALLOWED_ORIGINS=https://monapp.vercel.app,https://www.mondomaine.com
_raw_origins = _settings.allowed_origins
if _settings.env != "production":
    # Dev, staging, ou env non configuré → origines ouvertes
    _ALLOWED_ORIGINS: list[str] | str = ["*"]
elif _raw_origins == "*":
    # Production avec ALLOWED_ORIGINS=* — alerte et accepte (choix explicite)
    logger.warning("CORS: ALLOWED_ORIGINS=* en production — verrouiller les origines autorisées !")
    _ALLOWED_ORIGINS = ["*"]
else:
    _ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(
    title="Mental Health Signal Detector API",
    description="Détecte des signaux de détresse mentale dans un texte via NLP.",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_MAX_REQUEST_BODY = 64 * 1024  # 64 KB — textes longs refusés (prompt injection, DoS)


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """
    Rejette les requêtes dont le body dépasse _MAX_REQUEST_BODY octets.

    Lit les octets réels du body (pas seulement Content-Length) pour couvrir
    le transfert chunked et les clients qui omettent ou falsifient l'en-tête.
    Starlette met le body en cache après la première lecture — les handlers
    appelés via call_next peuvent relire request.body() sans overhead.
    """

    async def dispatch(self, request: Request, call_next):
        # Rejet rapide sur Content-Length déclaré (évite de lire inutilement)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_REQUEST_BODY:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Corps de requête trop grand (max {_MAX_REQUEST_BODY // 1024} KB)."},
            )
        # Lecture réelle pour couvrir chunked transfer et Content-Length absent/inexact
        body = await request.body()
        if len(body) > _MAX_REQUEST_BODY:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Corps de requête trop grand (max {_MAX_REQUEST_BODY // 1024} KB)."},
            )
        return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
app.add_middleware(RequestSizeLimitMiddleware)

app.include_router(checkin_router)
app.include_router(solutions_router)
app.include_router(analyze_router)


@app.get("/health", response_model=HealthResponse)
def health_check():
    if not _ML_AVAILABLE:
        return HealthResponse(status="ok", model_loaded=False)
    try:
        get_model("baseline")
        return HealthResponse(status="ok", model_loaded=True)
    except (FileNotFoundError, OSError):
        # Modèle absent (état normal avant entraînement) — 200 sans alerter les healthchecks
        return HealthResponse(status="ok", model_loaded=False)
    except Exception as e:
        # Erreur inattendue (corruption, OOM…) → 503
        logger.error(f"Health check — erreur modèle : {e}")
        raise HTTPException(status_code=503, detail="Modèle indisponible — erreur inattendue.") from e


@app.post("/explain", response_model=ExplainResponse)
@limiter.limit("10/minute")
def explain_endpoint(request: Request, request_body: ExplainRequest):
    """Retourne les contributions SHAP des mots les plus influents (baseline uniquement)."""
    if not _ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modèle non disponible — déploiement slim.")
    try:
        model = get_model("baseline")
        return run_explain(request_body, model)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Modèle indisponible (explain) : {e}")
        raise HTTPException(status_code=503, detail="Modèle non disponible — entraînement requis.") from e
    except Exception:
        logger.exception("Erreur explain")
        raise HTTPException(status_code=500, detail="Erreur interne.") from None


@app.post("/predict", response_model=PredictResponse)
@limiter.limit("30/minute")
def predict_endpoint(request: Request, body: PredictRequest):
    if not _ML_AVAILABLE:
        raise HTTPException(status_code=503, detail="Modèle non disponible — déploiement slim.")
    try:
        model = get_model(body.model_type)
        return run_prediction(body, model)
    except (FileNotFoundError, OSError) as e:
        logger.warning(f"Modèle indisponible : {e}")
        raise HTTPException(status_code=503, detail="Modèle non disponible — entraînement requis.") from e
    except Exception:
        logger.exception("Erreur prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne.") from None
