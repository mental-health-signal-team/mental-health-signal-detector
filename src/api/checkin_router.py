"""
Router FastAPI — /checkin
Phase 2 : "Comment vas-tu ce matin ?"
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from src.checkin.schemas import CheckInRequest, CheckInResponse
from src.checkin.engine import build_response

router = APIRouter(prefix="/checkin", tags=["checkin"])


@router.post("", response_model=CheckInResponse)
def checkin_endpoint(request: CheckInRequest):
    """
    Point d'entrée du check-in quotidien.

    Accepte un emoji et/ou un texte libre.
    Si un texte est fourni, il est analysé par le modèle NLP Phase 1.
    Retourne un message, un tip, une question de suivi et des ressources si nécessaire.
    """
    if not request.emoji and not request.text:
        raise HTTPException(status_code=422, detail="Fournis au moins un emoji ou un texte.")

    text_score = None
    detected_lang = "fr"

    if request.text and request.text.strip():
        try:
            from src.api.dependencies import get_model
            from src.training.predict import predict

            model = get_model("baseline")
            result = predict(request.text, model=model, model_type="baseline")
            text_score = result["score_distress"]
            detected_lang = result.get("detected_lang", "fr")
        except Exception as e:
            logger.warning(f"Analyse NLP échouée, score emoji seul utilisé : {e}")

    response_data = build_response(
        emoji=request.emoji,
        text=request.text,
        text_score=text_score,
        step=request.step,
    )

    return CheckInResponse(
        **response_data,
        detected_lang=detected_lang,
    )
