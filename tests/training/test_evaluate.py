import numpy as np
import pandas as pd

from src.training.evaluate import evaluate


class DummyVectorizer:
    def __init__(self):
        self.seen = None

    def transform(self, X_test):
        self.seen = list(X_test)
        return np.array([[0.0], [1.0], [2.0], [3.0]])


class DummyModel:
    def __init__(self):
        self.seen = None

    def predict(self, X_test):
        self.seen = X_test
        return np.array([1, 0, 1, 0])


def test_evaluate():
    """evaluate returns metric keys and valid metric values."""
    model = DummyModel()
    vectorizer = DummyVectorizer()
    X_test = pd.Series(["a", "b", "c", "d"])
    y_test = np.array([1, 0, 1, 0])

    metrics = evaluate(model, vectorizer, X_test, y_test)

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "classification_report",
    }
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert isinstance(metrics["classification_report"], str)
    assert "precision" in metrics["classification_report"]
    assert vectorizer.seen == ["a", "b", "c", "d"]
    assert model.seen.shape == (4, 1)
