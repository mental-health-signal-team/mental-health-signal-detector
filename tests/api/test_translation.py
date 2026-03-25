from src.api import services


class _FakeTranslator:
    def __init__(self, source, target):
        assert source == "auto"
        assert target == "en"

    def translate(self, text):
        return "i feel sad today" if text else text


def test_predict_translates_non_english_text(monkeypatch):
    """Non-English input should be translated before LR inference and tagged in metadata."""
    monkeypatch.setattr(services, "detect", lambda _text: "fr")
    monkeypatch.setattr(services, "GoogleTranslator", _FakeTranslator)

    captured = {}

    def _fake_lr_predict(_model, _vectorizer, text):
        captured["text"] = text
        return {"label": 1, "probability": 0.9}

    monkeypatch.setattr(services.predictor, "lr_predict", _fake_lr_predict)

    result = services.predict("je me sens triste", model_type="lr")

    assert captured["text"] == "i feel sad today"
    assert result["label"] == 1
    assert result["probability"] == 0.9
    assert result["source_language"] == "fr"
    assert result["analysis_language"] == "en"
    assert result["was_translated"] is True
    assert result["translated_text"] == "i feel sad today"


def test_predict_keeps_english_text_without_translation(monkeypatch):
    """English input should bypass translation and keep translated_text empty."""
    monkeypatch.setattr(services, "detect", lambda _text: "en")

    captured = {}

    def _fake_lr_predict(_model, _vectorizer, text):
        captured["text"] = text
        return {"label": 0, "probability": 0.2}

    monkeypatch.setattr(services.predictor, "lr_predict", _fake_lr_predict)

    result = services.predict("i am okay", model_type="lr")

    assert captured["text"] == "i am okay"
    assert result["source_language"] == "en"
    assert result["analysis_language"] == "en"
    assert result["was_translated"] is False
    assert result["translated_text"] is None
