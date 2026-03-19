from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(model, vectorizer, X_test, y_test) -> dict:
    """Evaluate the trained model on the test set and return a dictionary of metrics.
    - The function takes the trained model,
        the test features, and the test labels as input.
    - It returns a dictionary containing the accuracy,
        precision, recall, and F1-score of the model on the test set."""
    X_test = vectorizer.transform(X_test)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
    }
