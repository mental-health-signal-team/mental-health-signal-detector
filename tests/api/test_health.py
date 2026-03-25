from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
	"""Health endpoint should return healthy status."""
	response = client.get("/health")
	assert response.status_code == 200
	assert response.json() == {"status": "healthy"}


def test_distilbert_diagnostics_endpoint():
	"""DistilBERT diagnostics endpoint should return backend and artifact status keys."""
	response = client.get("/diagnostics/distilbert")
	assert response.status_code == 200
	payload = response.json()
	assert "backend" in payload
	assert "loaded" in payload
	assert "local_files" in payload
	assert "pickle_path" in payload
	assert "pickle_exists" in payload
	assert "last_error" in payload
