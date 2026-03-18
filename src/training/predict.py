from src.training.evaluate import LogisticRegression_fit


def LogisticRegression_predict(X_train, y_train, X_to_predict):
    """Fit a Logistic Regression model and predict labels for new samples."""
    log_reg = LogisticRegression_fit(X_train, y_train)
    y_pred = log_reg.predict(X_to_predict)
    return y_pred


def LogisticRegression_predict_proba(X_train, y_train, X_to_predict):
    """Fit a Logistic Regression model and predict class probabilities."""
    log_reg = LogisticRegression_fit(X_train, y_train)
    y_pred_proba = log_reg.predict_proba(X_to_predict)
    return y_pred_proba
