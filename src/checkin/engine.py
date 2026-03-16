"""
Moteur de conversation — "Comment vas-tu ce matin ?"

Combine le score emoji + l'analyse NLP du texte libre pour
déterminer le niveau de réponse (VERT / JAUNE / ROUGE).
"""

import random
from datetime import datetime
from enum import Enum

from loguru import logger

from src.checkin.content import (
    EMOJI_SCORES,
    GREEN_RESPONSES,
    GREEN_TIPS,
    RED_FOLLOWUP_QUESTIONS,
    RED_RESPONSES,
    RESOURCES,
    YELLOW_FOLLOWUP_QUESTIONS,
    YELLOW_RESPONSES,
    YELLOW_TIPS,
    get_greeting,
)


class DistressLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


# Seuils de détresse
THRESHOLD_GREEN = 0.35
THRESHOLD_RED = 0.65

# Plancher de sécurité par emoji — l'emoji est un choix explicite de l'utilisateur.
# Le modèle NLP peut échouer (faute d'orthographe, langue non reconnue) :
# le niveau ne doit JAMAIS être inférieur à ce plancher.
EMOJI_FLOOR: dict[str, DistressLevel] = {
    "😢": DistressLevel.RED,     # désespoir → toujours ROUGE
    "😔": DistressLevel.YELLOW,  # tristesse → au minimum JAUNE
}


def compute_score(emoji: str | None, text_score: float | None) -> float:
    """
    Fusionne le score emoji et le score NLP.
    Règle de sécurité : on prend le MAXIMUM des deux scores.
    Le modèle NLP peut sous-estimer (faute d'orthographe, traduction échouée) ;
    l'emoji reste le signal de référence.
    """
    emoji_score = EMOJI_SCORES.get(emoji) if emoji else None

    if emoji_score is not None and text_score is not None:
        return round(max(emoji_score, text_score), 3)
    elif emoji_score is not None:
        return emoji_score
    elif text_score is not None:
        return text_score
    else:
        return 0.45  # fallback neutre


def get_level(score: float, emoji: str | None = None) -> DistressLevel:
    """
    Détermine le niveau de détresse.
    Applique un plancher de sécurité basé sur l'emoji choisi :
    un utilisateur qui clique 😢 est toujours en ROUGE, quelle que soit
    l'analyse NLP du texte.
    """
    floor = EMOJI_FLOOR.get(emoji) if emoji else None

    if score < THRESHOLD_GREEN:
        level = DistressLevel.GREEN
    elif score < THRESHOLD_RED:
        level = DistressLevel.YELLOW
    else:
        level = DistressLevel.RED

    # Appliquer le plancher de sécurité
    order = [DistressLevel.GREEN, DistressLevel.YELLOW, DistressLevel.RED]
    if floor and order.index(floor) > order.index(level):
        logger.warning(f"Plancher de sécurité appliqué : emoji={emoji} score={score} {level}→{floor}")
        return floor

    return level


def build_response(
    emoji: str | None,
    text: str | None,
    text_score: float | None,
    step: int = 1,
) -> dict:
    """
    Construit la réponse conversationnelle.

    step=1 : première réponse après le check-in initial
    step=2 : réponse après la question de suivi
    """
    score = compute_score(emoji, text_score)
    level = get_level(score, emoji)
    hour = datetime.now().hour
    greeting = get_greeting(hour)

    logger.debug(f"CheckIn — emoji={emoji} text_score={text_score} → score={score} level={level}")

    if level == DistressLevel.GREEN:
        return {
            "level": level,
            "score": score,
            "message": random.choice(GREEN_RESPONSES),
            "tip": random.choice(GREEN_TIPS),
            "follow_up": None,
            "resources": [],
            "greeting": greeting,
        }

    elif level == DistressLevel.YELLOW:
        follow_up = random.choice(YELLOW_FOLLOWUP_QUESTIONS) if step == 1 else None
        return {
            "level": level,
            "score": score,
            "message": random.choice(YELLOW_RESPONSES),
            "tip": random.choice(YELLOW_TIPS),
            "follow_up": follow_up,
            "resources": [RESOURCES["mon_soutien_psy"], RESOURCES["medecin"]]
            if step == 2
            else [],
            "greeting": greeting,
        }

    else:  # RED
        follow_up = random.choice(RED_FOLLOWUP_QUESTIONS) if step == 1 else None
        resources = [
            RESOURCES["crisis_3114"],
            RESOURCES["mon_soutien_psy"],
            RESOURCES["fil_sante_jeunes"],
        ]
        return {
            "level": level,
            "score": score,
            "message": random.choice(RED_RESPONSES),
            "tip": None,
            "follow_up": follow_up,
            "resources": resources,
            "greeting": greeting,
        }
