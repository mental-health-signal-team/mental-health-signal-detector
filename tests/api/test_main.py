"""Tests unitaires API principaux."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client() -> TestClient:
    """Provide a TestClient for API endpoint tests."""
    return TestClient(app)


def test_api_empty_input(client: TestClient) -> None:
    """Test the /predict endpoint with empty input text to ensure it handles it gracefully."""
    response = client.post("/predict", json={"text": ""})

    assert response.status_code == 200
