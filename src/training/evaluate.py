from sklearn.linear_model import LogisticRegression

def LogisticRegression_fit(X_train, y_train, max_iter=1000):
    """Train a Logistic Regression model."""
    log_reg = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    log_reg.fit(X_train, y_train)
    return log_reg

def accuracy(model, X_test, y_test):
    """Evaluate the accuracy of model on the test set."""
    return model.score(X_test, y_test)