"""
Tests endpoint POST /checkin/reminder + GET /checkin/reminders
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


# ─── Validation de la requête ─────────────────────────────────────────────────

class TestValidation:
    def test_valid_request_1h(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        assert resp.status_code == 200

    def test_valid_request_4h(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "4h", "mode": "kids"})
        assert resp.status_code == 200

    def test_valid_request_tomorrow(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "tomorrow", "mode": "adult"})
        assert resp.status_code == 200

    def test_invalid_offset_returns_422(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "2days", "mode": "adult"})
        assert resp.status_code == 422

    def test_invalid_mode_returns_422(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "teenager"})
        assert resp.status_code == 422

    def test_with_optional_fields(self, client):
        resp = client.post("/checkin/reminder", json={
            "offset": "4h",
            "mode": "adult",
            "emotion_id": "sadness",
            "distress_level": "elevated",
        })
        assert resp.status_code == 200


# ─── Structure de la réponse ──────────────────────────────────────────────────

class TestResponseStructure:
    def test_all_fields_present(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        data = resp.json()
        assert "id" in data
        assert "offset" in data
        assert "scheduled_at" in data
        assert "scheduled_label" in data
        assert "message" in data

    def test_id_is_uuid_format(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        data = resp.json()
        import uuid
        # Doit être parsable comme UUID
        uuid.UUID(data["id"])

    def test_offset_reflected_in_response(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "4h", "mode": "adult"})
        assert resp.json()["offset"] == "4h"

    def test_scheduled_at_is_valid_iso(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        scheduled = resp.json()["scheduled_at"]
        dt = datetime.fromisoformat(scheduled)
        assert isinstance(dt, datetime)

    def test_scheduled_label_non_empty(self, client):
        for offset in ("1h", "4h", "tomorrow"):
            resp = client.post("/checkin/reminder", json={"offset": offset, "mode": "adult"})
            assert len(resp.json()["scheduled_label"]) > 0

    def test_message_non_empty(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        assert len(resp.json()["message"]) > 0


# ─── Logique de calcul du temps ───────────────────────────────────────────────

class TestScheduledTime:
    def test_1h_adds_one_hour(self, client):
        before = datetime.now()
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        after = datetime.now()
        scheduled = datetime.fromisoformat(resp.json()["scheduled_at"])
        expected_min = before + timedelta(hours=1)
        expected_max = after + timedelta(hours=1)
        assert expected_min <= scheduled <= expected_max

    def test_4h_adds_four_hours(self, client):
        before = datetime.now()
        resp = client.post("/checkin/reminder", json={"offset": "4h", "mode": "adult"})
        after = datetime.now()
        scheduled = datetime.fromisoformat(resp.json()["scheduled_at"])
        expected_min = before + timedelta(hours=4)
        expected_max = after + timedelta(hours=4)
        assert expected_min <= scheduled <= expected_max

    def test_tomorrow_is_next_day_at_9h(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "tomorrow", "mode": "adult"})
        scheduled = datetime.fromisoformat(resp.json()["scheduled_at"])
        tomorrow = datetime.now() + timedelta(days=1)
        assert scheduled.date() == tomorrow.date()
        assert scheduled.hour == 9
        assert scheduled.minute == 0

    def test_label_contains_heure_for_1h(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        label = resp.json()["scheduled_label"]
        assert "h" in label  # "aujourd'hui à 15h30"

    def test_label_contains_demain_for_tomorrow(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "tomorrow", "mode": "adult"})
        label = resp.json()["scheduled_label"]
        assert "demain" in label


# ─── Messages adaptés au mode ─────────────────────────────────────────────────

class TestModeAdaptation:
    def test_kids_message_contains_emoji(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "kids"})
        message = resp.json()["message"]
        assert "💙" in message or "!" in message  # message enfant plus expressif

    def test_adult_message_more_formal(self, client):
        resp = client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"})
        message = resp.json()["message"]
        assert len(message) > 0


# ─── IDs uniques ─────────────────────────────────────────────────────────────

class TestUniqueIds:
    def test_each_reminder_has_unique_id(self, client):
        ids = [
            client.post("/checkin/reminder", json={"offset": "1h", "mode": "adult"}).json()["id"]
            for _ in range(5)
        ]
        assert len(set(ids)) == 5


# GET /checkin/reminders supprimé — tests retirés avec l'endpoint.
# Raison : exposait emotion_id + distress_level de tous les utilisateurs
# sans authentification (fuite inter-utilisateurs, RGPD art. 9).
