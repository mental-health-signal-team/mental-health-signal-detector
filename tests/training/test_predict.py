import numpy as np

from src.training.predict import (
    LogisticRegression_predict,
    LogisticRegression_predict_proba,
)


def test_logistic_regression_predict():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_to_predict = np.array([[0.5], [4.5]])

    y_pred = LogisticRegression_predict(X_train, y_train, X_to_predict)

    assert y_pred.shape == (2,)
    assert set(y_pred).issubset({0, 1})


def test_logistic_regression_predict_proba():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_to_predict = np.array([[0.5], [4.5]])

    y_pred_proba = LogisticRegression_predict_proba(X_train, y_train, X_to_predict)

    assert y_pred_proba.shape == (2, 2)
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0)
    assert np.all((y_pred_proba >= 0.0) & (y_pred_proba <= 1.0))
