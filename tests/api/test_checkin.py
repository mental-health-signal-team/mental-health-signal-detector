"""
Tests API — endpoint /checkin

Couvre :
- 422 si ni emoji ni texte fournis
- 200 avec emoji seul (pas de NLP)
- 200 avec texte seul
- Fallback emoji quand le modèle NLP est indisponible
- Plancher de sécurité : 😢 → niveau red, 😔 → niveau yellow minimum
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_checkin_missing_emoji_and_text(client):
    """422 si aucun emoji ni texte n'est fourni."""
    response = client.post("/checkin", json={})
    assert response.status_code == 422


def test_checkin_emoji_only(client):
    """200 avec emoji seul — pas d'appel NLP."""
    response = client.post("/checkin", json={"emoji": "😄"})
    assert response.status_code == 200
    data = response.json()
    assert data["level"] == "green"
    assert 0.0 <= data["score"] <= 1.0
    assert data["message"]


def test_checkin_text_only_nlp_fallback(client):
    """200 avec texte seul — si NLP échoue, on utilise le score par défaut."""
    with patch("src.api.checkin_router.get_model", side_effect=FileNotFoundError("no model")):
        response = client.post("/checkin", json={"text": "Je me sens fatigué"})
    assert response.status_code == 200
    data = response.json()
    assert data["level"] in ("green", "yellow", "red")


def test_checkin_emoji_safety_floor_red(client):
    """😢 → niveau red garanti même si le NLP retourne un score faible."""
    with patch("src.api.checkin_router.get_model"), \
         patch("src.api.checkin_router.predict") as mock_predict:
        mock_predict.return_value = {
            "score_distress": 0.1,  # NLP optimiste
            "detected_lang": "fr",
        }
        response = client.post("/checkin", json={"emoji": "😢", "text": "ça va"})
    assert response.status_code == 200
    assert response.json()["level"] == "red"


def test_checkin_emoji_safety_floor_yellow(client):
    """😔 → niveau minimum yellow même si le NLP retourne un score vert."""
    with patch("src.api.checkin_router.get_model"), \
         patch("src.api.checkin_router.predict") as mock_predict:
        mock_predict.return_value = {
            "score_distress": 0.05,  # NLP très optimiste
            "detected_lang": "fr",
        }
        response = client.post("/checkin", json={"emoji": "😔", "text": "ça va"})
    assert response.status_code == 200
    assert response.json()["level"] in ("yellow", "red")


def test_checkin_step2_no_followup(client):
    """step=2 → pas de question de suivi dans la réponse."""
    response = client.post("/checkin", json={"emoji": "😐", "step": 2})
    assert response.status_code == 200
    assert response.json()["follow_up"] is None
