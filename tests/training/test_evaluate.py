from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.training.evaluate import evaluate


def test_evaluate():
    """evaluate returns metric keys and valid metric values."""
    vectorizer = Mock()
    vectorizer.transform.return_value = np.array([[0.0], [1.0], [2.0], [3.0]])

    model = Mock()
    model.predict.return_value = np.array([1, 0, 1, 0])

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
    vectorizer.transform.assert_called_once()
    assert list(vectorizer.transform.call_args.args[0]) == ["a", "b", "c", "d"]
    model.predict.assert_called_once()
    assert model.predict.call_args.args[0].shape == (4, 1)
