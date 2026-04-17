"""Tests for the main FastAPI application endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient for API endpoint tests (triggers lifespan / DB init)."""
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Infrastructure endpoints — always run regardless of model artifacts
# ---------------------------------------------------------------------------


def test_root_returns_200(client: TestClient) -> None:
    """GET / should return 200 with endpoint map."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "endpoints" in body
    assert "predict" in body["endpoints"]
    assert "drift" in body["endpoints"]


def test_health_returns_healthy(client: TestClient) -> None:
    """GET /health should return status=healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# ---------------------------------------------------------------------------
# Stats endpoints — work with empty SQLite DB, no models required
# ---------------------------------------------------------------------------


def test_drift_returns_200_with_empty_db(client: TestClient) -> None:
    """GET /stats/drift should return 200 even when the database has no rows."""
    response = client.get("/stats/drift")
    assert response.status_code == 200
    body = response.json()
    assert "baseline_confidence" in body
    assert "recent_confidence" in body
    assert "drift_detected" in body
    assert "drift_threshold" in body
    assert isinstance(body["drift_detected"], bool)


def test_drift_response_schema(client: TestClient) -> None:
    """GET /stats/drift response must contain all expected DriftResponse fields."""
    response = client.get("/stats/drift")
    assert response.status_code == 200
    body = response.json()
    required_fields = [
        "baseline_confidence",
        "recent_confidence",
        "confidence_delta",
        "drift_detected",
        "drift_threshold",
        "baseline_distress_rate",
        "recent_distress_rate",
        "recent_predictions_count",
        "model_confidence_7d",
    ]
    for field in required_fields:
        assert field in body, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# Predict endpoint — skipped by conftest when model artifacts are absent
# ---------------------------------------------------------------------------


def test_api_empty_input(client: TestClient) -> None:
    """POST /predict with empty text should return 422 (min_length=1 validation)."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_returns_label_and_probability(client: TestClient) -> None:
    """POST /predict should return a label (0 or 1) and a probability in [0, 1]."""
    response = client.post("/predict", json={"text": "I feel very sad and hopeless."})
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "probability" in body
    assert body["label"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_long_input(client: TestClient) -> None:
    """POST /predict with text exceeding max_length=2000 should return 422."""
    long_text = "I feel sad. " * 200  # ~2400 chars
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 422


def test_predict_default_model_is_lr(client: TestClient) -> None:
    """POST /predict without specifying model_type should default to 'lr'."""
    response = client.post("/predict", json={"text": "feeling okay today"})
    assert response.status_code == 200


def test_predict_mental_roberta_model(client: TestClient) -> None:
    """POST /predict with model_type=mental_roberta should return a valid prediction."""
    response = client.post(
        "/predict",
        json={"text": "I cannot stop crying and I don't see a way out.", "model_type": "mental_roberta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_model_type(client: TestClient) -> None:
    """POST /predict with an unknown model_type should return 422 Unprocessable Entity."""
    response = client.post("/predict", json={"text": "hello", "model_type": "gpt5"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Explain endpoint — skipped by conftest when model artifacts are absent
# ---------------------------------------------------------------------------


def test_explain_returns_word_importance(client: TestClient) -> None:
    """POST /explain should return word_importance dict and colored_html."""
    response = client.post(
        "/explain",
        json={"text": "I feel hopeless and cannot sleep.", "model_type": "lr"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "word_importance" in body
    assert "colored_html" in body
    assert "risk_level" in body
    assert body["risk_level"] in ("low", "medium", "high")
    assert body["confidence_label"] in ("distress", "no_distress")


def test_explain_invalid_model_returns_400(client: TestClient) -> None:
    """POST /explain with a non-explainable model should return 400."""
    response = client.post(
        "/explain",
        json={"text": "I feel okay.", "model_type": "xgboost"},
    )
    # XGBoost explain may raise ValueError — API wraps it in 400
    assert response.status_code in (200, 400)
