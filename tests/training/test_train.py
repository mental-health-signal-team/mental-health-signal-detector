import joblib
import pandas as pd

from src.training import train as train_module


def test_load_and_prepare_data():
    """load_and_prepare_data applies preprocessing and returns 70/15/15 splits."""
    df = pd.DataFrame(
        {
            "title": [f"post_{i}" for i in range(20)],
            "label": [0] * 10 + [1] * 10,
        }
    )

    X_train, y_train, X_val, y_val, X_test, y_test = train_module.load_and_prepare_data(
        load_data_fn=lambda: df.copy(),
        clean_data_fn=lambda data: data.copy(),
        preprocess_fn=lambda text: f"prep_{text}",
    )

    assert len(X_train) == 14
    assert len(y_train) == 14
    assert len(X_val) == 3
    assert len(y_val) == 3
    assert len(X_test) == 3
    assert len(y_test) == 3
    assert X_train.str.startswith("prep_").all()
    assert X_val.str.startswith("prep_").all()
    assert X_test.str.startswith("prep_").all()
    assert len(X_train) + len(X_val) + len(X_test) == 20


def test_train_model():
    """train_model returns a fitted TF-IDF vectorizer and logistic model."""
    X_train = pd.Series(["happy calm"] * 6 + ["sad dark"] * 6)
    y_train = pd.Series([1] * 6 + [0] * 6)

    vectorizer, model = train_module.train_model(X_train, y_train)

    assert "happy" in vectorizer.vocabulary_
    assert "sad" in vectorizer.vocabulary_
    assert model.class_weight == "balanced"
    assert set(model.classes_) == {0, 1}


def test_save_artifacts(tmp_path):
    """save_artifacts persists vectorizer/model to configured paths."""
    models_dir = tmp_path / "models"
    vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
    model_path = models_dir / "lr_model.pkl"

    vectorizer_obj = {"artifact": "vectorizer"}
    model_obj = {"artifact": "model"}

    train_module.save_artifacts(
        vectorizer_obj,
        model_obj,
        models_dir=models_dir,
        vectorizer_path=vectorizer_path,
        model_path=model_path,
    )

    assert vectorizer_path.exists()
    assert model_path.exists()
    assert joblib.load(vectorizer_path) == vectorizer_obj
    assert joblib.load(model_path) == model_obj


def test_train_xgboost_model_uses_notebook_style_settings():
    """train_xgboost_model fits TF-IDF + model using notebook-like pipeline."""

    class DummyModel:
        def __init__(self):
            self.fit_called = False
            self.fit_args = None

        def fit(self, X, y):
            self.fit_called = True
            self.fit_args = (X, y)
            return self

    model_instance = DummyModel()
    X_train = pd.Series(["this is okay"] * 6 + ["really bad mood"] * 6)
    y_train = pd.Series([0] * 6 + [1] * 6)

    vectorizer, model = train_module.train_xgboost_model(
        X_train,
        y_train,
        model_factory=lambda: model_instance,
    )

    assert model is model_instance
    assert model.fit_called is True
    assert model.fit_args[0].shape[0] == len(X_train)
    assert model.fit_args[1].equals(y_train)
    assert vectorizer.max_features == 10000
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.stop_words == "english"
