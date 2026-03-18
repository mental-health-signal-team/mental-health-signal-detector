import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def create_X_y(df: pd.DataFrame) -> tuple:
    """Create the feature matrix X and target vector y from the cleaned DataFrame."""
    X = df["clean_title"]
    y = df["label"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42) -> tuple:
    """Split the data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def vectorize_data(X_train, X_test, max_features=10000) -> tuple:
    """Vectorize the text data using TF-IDF vectorization."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer
