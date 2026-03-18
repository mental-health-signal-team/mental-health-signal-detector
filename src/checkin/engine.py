"""
Moteur de conversation — "Comment vas-tu ce matin ?"

4 niveaux de reponse : CRITICAL / RED / YELLOW / GREEN
Le niveau CRITICAL est detecte par mots-cles AVANT tout scoring ML.
"""

import random
import uuid
from datetime import datetime, timedelta
from enum import Enum

from loguru import logger

from src.checkin.content import (
    CRITICAL_KEYWORDS,
    CRITICAL_RESPONSES,
    EMOJI_SCORES,
    GREEN_RESPONSES,
    GREEN_TIPS,
    INTENSITY_MODIFIERS,
    INTENSITY_SCORE_BOOST,
    RED_FOLLOWUP_QUESTIONS,
    RED_RESPONSES,
    RESOURCES_CRITICAL,
    RESOURCES_RED,
    RESOURCES_YELLOW,
    YELLOW_FOLLOWUP_QUESTIONS,
    YELLOW_RESPONSES,
    YELLOW_TIPS,
    get_greeting,
)


class DistressLevel(str, Enum):
    CRITICAL = "critical"   # ideation suicidaire — securite absolue
    RED      = "red"        # detresse forte
    YELLOW   = "yellow"     # detresse moderee
    GREEN    = "green"      # bien-etre


THRESHOLD_GREEN = 0.35
THRESHOLD_RED   = 0.65

# Plancher de securite : l'emoji choisi ne peut pas etre contredit par le NLP
EMOJI_FLOOR: dict[str, DistressLevel] = {
    "😢": DistressLevel.RED,
    "😔": DistressLevel.YELLOW,
}


def _normalize(s: str) -> str:
    """Normalise pour la detection : minuscules, suppression des apostrophes variantes."""
    return s.lower().replace("\u2019", "").replace("\u2018", "").replace("'", "").replace("'", "")


# Pre-normalisation des mots-cles pour correspondance robuste
_CRITICAL_KEYWORDS_NORMALIZED = [_normalize(kw) for kw in CRITICAL_KEYWORDS]


def check_critical(text: str | None) -> bool:
    """
    Detecte l'ideation suicidaire par mots-cles.
    Appelee AVANT tout scoring NLP/emoji — securite absolue.
    Normalise le texte ET les mots-cles (apostrophes, casse) pour maximiser la detection.
    """
    if not text:
        return False
    normalized = _normalize(text)
    return any(kw in normalized for kw in _CRITICAL_KEYWORDS_NORMALIZED)


def apply_intensity_boost(text: str | None, score: float) -> float:
    """
    Augmente le score si des modificateurs d'intensite/frequence sont detectes.
    Ex : 'je suis triste tout le temps' → score booste de +0.15
    """
    if not text:
        return score
    normalized = text.lower()
    if any(mod in normalized for mod in INTENSITY_MODIFIERS):
        boosted = min(score + INTENSITY_SCORE_BOOST, 1.0)
        logger.debug(f"Intensity boost applique : {score:.3f} -> {boosted:.3f}")
        return boosted
    return score


def compute_score(emoji: str | None, text_score: float | None, text: str | None = None) -> float:
    """
    Fusionne le score emoji et le score NLP.
    Regle de securite : on prend le MAXIMUM des deux scores.
    Applique ensuite le boost d'intensite si des modificateurs sont detectes.
    """
    emoji_score = EMOJI_SCORES.get(emoji) if emoji else None

    if emoji_score is not None and text_score is not None:
        base = max(emoji_score, text_score)
    elif emoji_score is not None:
        base = emoji_score
    elif text_score is not None:
        base = text_score
    else:
        base = 0.45

    return round(apply_intensity_boost(text, base), 3)


def get_level(score: float, emoji: str | None = None) -> DistressLevel:
    """
    Determine le niveau de detresse a partir du score fusionne.
    Applique le plancher de securite emoji.
    """
    if score < THRESHOLD_GREEN:
        level = DistressLevel.GREEN
    elif score < THRESHOLD_RED:
        level = DistressLevel.YELLOW
    else:
        level = DistressLevel.RED

    floor = EMOJI_FLOOR.get(emoji) if emoji else None
    order = [DistressLevel.GREEN, DistressLevel.YELLOW, DistressLevel.RED, DistressLevel.CRITICAL]
    if floor and order.index(floor) > order.index(level):
        logger.warning(f"Plancher securite : emoji={emoji} score={score} {level}->{floor}")
        return floor

    return level


def build_response(
    emoji: str | None,
    text: str | None,
    text_score: float | None,
    step: int = 1,
) -> dict:
    """
    Construit la reponse conversationnelle.

    Ordre de priorite :
      1. Verification CRITICAL (mots-cles ideation suicidaire) — independant du score
      2. Scoring fusionne emoji + NLP + boost intensite
      3. Reponse par niveau

    step=1 : premiere reponse apres check-in
    step=2 : reponse apres question de suivi
    """
    hour     = datetime.now().hour  # noqa: DTZ005
    greeting = get_greeting(hour)

    # --- PRIORITE 1 : detection CRITICAL ---
    if check_critical(text):
        import hashlib
        text_hash = hashlib.sha256((text or "").encode()).hexdigest()[:12]
        logger.warning(f"CRITICAL detecte : text_hash={text_hash} len={len(text or '')}")
        return {
            "level":      DistressLevel.CRITICAL,
            "score":      1.0,
            "message":    random.choice(CRITICAL_RESPONSES),  # noqa: S311
            "tip":        None,
            "follow_up":  None,
            "resources":  RESOURCES_CRITICAL,
            "greeting":   greeting,
            "critical":   True,
        }

    # --- PRIORITE 2 : scoring fusionne ---
    score = compute_score(emoji, text_score, text)
    level = get_level(score, emoji)
    logger.debug(f"CheckIn — emoji={emoji} text_score={text_score} score={score} level={level}")

    if level == DistressLevel.GREEN:
        return {
            "level":     level,
            "score":     score,
            "message":   random.choice(GREEN_RESPONSES),   # noqa: S311
            "tip":       random.choice(GREEN_TIPS),         # noqa: S311
            "follow_up": None,
            "resources": [],
            "greeting":  greeting,
            "critical":  False,
        }

    elif level == DistressLevel.YELLOW:
        follow_up = random.choice(YELLOW_FOLLOWUP_QUESTIONS) if step == 1 else None  # noqa: S311
        return {
            "level":     level,
            "score":     score,
            "message":   random.choice(YELLOW_RESPONSES),   # noqa: S311
            "tip":       random.choice(YELLOW_TIPS),         # noqa: S311
            "follow_up": follow_up,
            "resources": RESOURCES_YELLOW if step == 2 else [],
            "greeting":  greeting,
            "critical":  False,
        }

    else:  # RED
        follow_up = random.choice(RED_FOLLOWUP_QUESTIONS) if step == 1 else None  # noqa: S311
        return {
            "level":     level,
            "score":     score,
            "message":   random.choice(RED_RESPONSES),   # noqa: S311
            "tip":       None,
            "follow_up": follow_up,
            "resources": RESOURCES_RED,
            "greeting":  greeting,
            "critical":  False,
        }


def compute_reminder(offset: str, mode: str) -> dict:
    now = datetime.now()  # noqa: DTZ005
    if offset == "1h":
        scheduled_at = now + timedelta(hours=1)
        label = f"aujourd'hui a {scheduled_at.strftime('%Hh%M')}"
    elif offset == "4h":
        scheduled_at = now + timedelta(hours=4)
        day = "demain" if scheduled_at.date() > now.date() else "aujourd'hui"
        label = f"{day} a {scheduled_at.strftime('%Hh%M')}"
    else:
        scheduled_at = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
        label = "demain a 9h00"

    message = f"Super ! On se retrouve {label} 💙" if mode == "kids" else f"Rappel programme pour {label}."
    return {
        "id":              str(uuid.uuid4()),
        "offset":          offset,
        "scheduled_at":    scheduled_at.isoformat(),
        "scheduled_label": label,
        "message":         message,
    }
