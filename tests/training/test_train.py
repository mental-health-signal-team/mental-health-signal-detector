import pandas as pd
import pytest
from src.training.preprocess import clean_text
from src.training.train import train_baseline
from src.training.evaluate import predict_proba_text


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "text": [
            "I feel so hopeless and empty, nothing matters anymore",
            "I feel really sad and don't want to get out of bed",
            "Had a great day at the park with my family",
            "Excited about the new project at work, feeling motivated",
            "Can't stop crying, everything feels overwhelming",
            "Life is beautiful and I'm grateful for everything",
        ],
        "label": [1, 1, 0, 0, 1, 0],
    })


def test_clean_text():
    assert clean_text("Hello World!! http://test.com @user") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_clean_text_removes_special_chars():
    result = clean_text("I'm feeling sad... 😢")
    assert "http" not in result
    assert result == result.lower()


def test_train_baseline(sample_df):
    pipeline = train_baseline(sample_df)
    assert pipeline is not None
    preds = pipeline.predict(sample_df["text"])
    assert len(preds) == len(sample_df)
    assert set(preds).issubset({0, 1})


def test_predict_proba_text(sample_df):
    pipeline = train_baseline(sample_df)
    result = predict_proba_text(pipeline, "I feel hopeless and sad")
    assert "label" in result
    assert "score_distress" in result
    assert 0.0 <= result["score_distress"] <= 1.0
    assert result["label"] in (0, 1)
