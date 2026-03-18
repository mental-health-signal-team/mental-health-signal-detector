import numpy as np
from sklearn.linear_model import LogisticRegression

from src.training.evaluate import LogisticRegression_fit, accuracy


def test_LogisticRegression_fit():
    """Test the LogisticRegression_fit function by checking if it returns a fitted model"""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression_fit(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert model.class_weight == "balanced"
    assert set(model.classes_) == {0, 1}


def test_accuracy():
    """Test the accuracy function by checking if it returns a score between 0 and 1"""
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegression_fit(X_train, y_train)
    score = accuracy(model, X_train, y_train)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
