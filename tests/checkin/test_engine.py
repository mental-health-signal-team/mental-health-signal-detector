"""
Tests unitaires — moteur de conversation (src/checkin/engine.py)

Couvre le plancher de sécurité EMOJI_FLOOR :
- 😢 (désespoir) → toujours RED, même si le score NLP est bas
- 😔 (tristesse) → jamais en dessous de YELLOW, même si le score NLP est vert
- Emojis neutres/positifs → niveau determiné par le score uniquement
"""
from src.checkin.engine import compute_score, get_level, build_response, DistressLevel


# ---------------------------------------------------------------------------
# compute_score
# ---------------------------------------------------------------------------

def test_compute_score_emoji_only():
    assert compute_score("😄", None) == 0.05
    assert compute_score("😢", None) == 0.85


def test_compute_score_text_only():
    assert compute_score(None, 0.7) == 0.7


def test_compute_score_takes_max():
    """Le plancher de sécurité est le MAX des deux scores."""
    assert compute_score("😄", 0.9) == 0.9   # NLP plus pessimiste → gagne
    assert compute_score("😢", 0.1) == 0.85  # emoji plus pessimiste → gagne


def test_compute_score_fallback():
    assert compute_score(None, None) == 0.45


# ---------------------------------------------------------------------------
# get_level — plancher de sécurité EMOJI_FLOOR
# ---------------------------------------------------------------------------

def test_safety_floor_cry_emoji_always_red():
    """😢 doit toujours produire RED, quelle que soit l'analyse NLP."""
    # Score très bas (NLP optimiste)
    assert get_level(0.05, "😢") == DistressLevel.RED
    # Score moyen
    assert get_level(0.45, "😢") == DistressLevel.RED
    # Score déjà rouge
    assert get_level(0.80, "😢") == DistressLevel.RED


def test_safety_floor_sad_emoji_minimum_yellow():
    """😔 ne peut jamais produire GREEN."""
    # Score vert selon les seuils (< 0.35)
    assert get_level(0.10, "😔") == DistressLevel.YELLOW
    assert get_level(0.20, "😔") == DistressLevel.YELLOW
    # Score jaune → reste jaune
    assert get_level(0.50, "😔") == DistressLevel.YELLOW
    # Score rouge → reste rouge
    assert get_level(0.80, "😔") == DistressLevel.RED


def test_no_floor_positive_emoji():
    """😄 et 🙂 ne modifient pas le niveau calculé par le score."""
    assert get_level(0.10, "😄") == DistressLevel.GREEN
    assert get_level(0.50, "😄") == DistressLevel.YELLOW
    assert get_level(0.80, "😄") == DistressLevel.RED


def test_level_thresholds_without_emoji():
    assert get_level(0.00) == DistressLevel.GREEN
    assert get_level(0.34) == DistressLevel.GREEN
    assert get_level(0.35) == DistressLevel.YELLOW
    assert get_level(0.64) == DistressLevel.YELLOW
    assert get_level(0.65) == DistressLevel.RED
    assert get_level(1.00) == DistressLevel.RED


# ---------------------------------------------------------------------------
# build_response — structure de sortie
# ---------------------------------------------------------------------------

def test_build_response_green_has_tip():
    resp = build_response(emoji="😄", text=None, text_score=None)
    assert resp["level"] == DistressLevel.GREEN
    assert resp["tip"] is not None
    assert resp["resources"] == []


def test_build_response_red_has_resources():
    resp = build_response(emoji="😢", text=None, text_score=None)
    assert resp["level"] == DistressLevel.RED
    assert len(resp["resources"]) > 0
    assert resp["tip"] is None


def test_build_response_yellow_step1_has_followup():
    resp = build_response(emoji="😔", text=None, text_score=None, step=1)
    assert resp["level"] in (DistressLevel.YELLOW, DistressLevel.RED)
    assert resp["follow_up"] is not None


def test_build_response_step2_no_followup():
    resp = build_response(emoji="😐", text=None, text_score=0.50, step=2)
    assert resp["follow_up"] is None
