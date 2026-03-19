import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import src.common.config as config
import src.data_cleaning.data as data
import src.training.preprocess as preprocess


def load_and_prepare_data(load_data_fn=None, clean_data_fn=None, preprocess_fn=None) -> tuple:
    """
    Load the dataset, preprocess the text,
    and split it into train, validation, and test sets.
    Optional callables allow lightweight tests without monkeypatching.
    """
    if load_data_fn is None:
        load_data_fn = data.load_data
    if clean_data_fn is None:
        clean_data_fn = data.clean_data
    if preprocess_fn is None:
        preprocess_fn = preprocess.preprocess_text

    df = load_data_fn()
    df_cleaned = clean_data_fn(df)
    df_cleaned["title"] = df_cleaned["title"].apply(preprocess_fn)
    X = df_cleaned["title"]
    y = df_cleaned["label"]
    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.50, random_state=42, stratify=y_test_val)
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train) -> tuple:
    """Train a logistic regression model using TF-IDF features.
    - We use a TfidfVectorizer to convert the text data into TF-IDF features,
        with a maximum of 50,000 features, unigrams and bigrams, and a minimum document frequency of 5.
    - We train a LogisticRegression model with balanced class weights
        and a maximum of 1000 iterations.
    - The function returns the fitted vectorizer and model"""
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
    vectorizer.fit(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(vectorizer.transform(X_train), y_train)
    return vectorizer, model


def save_artifacts(vectorizer, model, models_dir=None, vectorizer_path=None, model_path=None):
    """Save the trained model and vectorizer to disk using joblib.
    Optional paths make the function easy to test without monkeypatching."""
    if models_dir is None:
        models_dir = config.MODELS_DIR
    if vectorizer_path is None:
        vectorizer_path = config.VECTORIZER_PATH
    if model_path is None:
        model_path = config.LR_MODEL_PATH

    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data()
    vectorizer, model = train_model(X_train, y_train)
    save_artifacts(vectorizer, model)
