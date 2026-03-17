"""
Router FastAPI — /checkin

Phase 2 : "Comment vas-tu ce matin ?" — check-in émotionnel (emoji + texte)
Phase 3 : /checkin/reminder — persistance des rappels de suivi
"""

from collections import deque

from fastapi import APIRouter, HTTPException, Request
from loguru import logger

from src.api.rate_limit import limiter

from src.checkin.schemas import CheckInRequest, CheckInResponse, ReminderRequest, ReminderResponse

# Imports ML optionnels — absents en mode slim
try:
    from src.api.dependencies import get_model as _get_model
    from src.training.predict import predict as _predict
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    _get_model = None  # type: ignore[assignment]
    _predict = None  # type: ignore[assignment]
from src.checkin.engine import build_response, compute_reminder

router = APIRouter(prefix="/checkin", tags=["checkin"])

# ─── Store en mémoire — rappels de suivi ─────────────────────────────────────
# Taille max = 1000 entrées (rotation FIFO) pour éviter la fuite mémoire.
# Remplaçable par une vraie persistance (SQLite, Redis) sans changer l'interface.
_MAX_REMINDERS = 1000
_reminders: deque[dict] = deque(maxlen=_MAX_REMINDERS)


@router.post("", response_model=CheckInResponse)
@limiter.limit("20/minute")
def checkin_endpoint(request: Request, body: CheckInRequest):
    """
    Point d'entrée du check-in quotidien.

    Accepte un emoji et/ou un texte libre.
    Si un texte est fourni, il est analysé par le modèle NLP Phase 1.
    Retourne un message, un tip, une question de suivi et des ressources si nécessaire.
    """
    if not body.emoji and not body.text:
        raise HTTPException(status_code=422, detail="Fournis au moins un emoji ou un texte.")

    text_score = None
    detected_lang = "fr"

    if body.text and body.text.strip() and _ML_AVAILABLE:
        try:
            model = _get_model("baseline")
            result = _predict(body.text, model=model, model_type="baseline")
            text_score = result["score_distress"]
            detected_lang = result.get("detected_lang", "fr")
        except Exception as e:
            logger.warning(f"Analyse NLP échouée, score emoji seul utilisé : {e}")

    response_data = build_response(
        emoji=body.emoji,
        text=body.text,
        text_score=text_score,
        step=body.step,
    )

    return CheckInResponse(
        **response_data,
        detected_lang=detected_lang,
    )


@router.post("/reminder", response_model=ReminderResponse)
def reminder_endpoint(request: ReminderRequest) -> ReminderResponse:
    """
    Enregistre une intention de rappel de suivi.

    Calcule l'heure réelle de rappel (now + offset) et retourne un libellé
    lisible ("aujourd'hui à 15h30", "demain à 9h00").

    Le store est en mémoire (FIFO, max 1000 entrées).
    Remplaçable par SQLite / Redis sans changer l'interface publique.
    """
    reminder = compute_reminder(offset=request.offset, mode=request.mode)
    reminder["emotion_id"] = request.emotion_id
    reminder["distress_level"] = request.distress_level

    _reminders.append(reminder)

    logger.info(
        f"[reminder] id={reminder['id']} offset={request.offset} "
        f"scheduled={reminder['scheduled_at']} mode={request.mode} "
        f"emotion={request.emotion_id} distress={request.distress_level}"
    )

    return ReminderResponse(**reminder)


# GET /checkin/reminders supprimé — exposait emotion_id + distress_level
# de tous les utilisateurs sans authentification (fuite inter-utilisateurs, RGPD art. 9).
