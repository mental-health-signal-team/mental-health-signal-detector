"""Tests for the /health endpoint."""

from fastapi.testclient import TestClient

from src.api.main import app


def test_health_check() -> None:
    """GET /health must return 200 with status=healthy."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health_does_not_require_model_artifacts() -> None:
    """GET /health should respond immediately regardless of model loading state."""
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
