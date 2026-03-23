import io
import pickle

import joblib
import torch

import src.common.config as config
import src.training.predict as predictor

_lr_model = joblib.load(config.LR_MODEL_PATH)
_lr_vectorizer = joblib.load(config.VECTORIZER_PATH)
_distilbert_model = None  # Lazy load the DistilBERT model when needed
_roberta_model = None
_xgboost_model = None
_xgboost_vectorizer = None


def _load_artifact(path, prefer_torch: bool = False):
    """Load model artifacts using multiple strategies for compatibility."""

    def _joblib_loader(target_path):
        return joblib.load(target_path)

    def _cpu_pickle_loader(target_path):
        with open(target_path, "rb") as f:
            return CPUUnpickler(f).load()

    def _pickle_loader(target_path):
        with open(target_path, "rb") as f:
            return pickle.load(f)

    loaders = [
        ("cpu_pickle", _cpu_pickle_loader),
        ("joblib", _joblib_loader),
        ("pickle", _pickle_loader),
    ]
    if not prefer_torch:
        loaders = [
            ("joblib", _joblib_loader),
            ("cpu_pickle", _cpu_pickle_loader),
            ("pickle", _pickle_loader),
        ]

    errors = []
    for loader_name, loader in loaders:
        try:
            return loader(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")

    joined_errors = " | ".join(errors)
    raise RuntimeError(f"Unable to load artifact at {path}. Tried multiple loaders. {joined_errors}")


class CPUUnpickler(pickle.Unpickler):
    """Custom unpickler to load GPU-trained PyTorch models on CPU."""

    def find_class(self, module, name):
        """Override find_class to handle CUDA->CPU mapping and missing transformer classes."""
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        if module == "torch" and name == "load":
            return lambda f, *a, **kw: torch.load(f, map_location="cpu", weights_only=False)
        try:
            return super().find_class(module, name)
        except AttributeError as exc:
            if "roberta" in module.lower() or "sdpa" in name.lower():
                return type(name, (object,), {})
            raise


def _get_distilbert_model():
    """Load the DistilBERT model from disk if it hasn't been loaded yet, and return it."""
    global _distilbert_model
    if _distilbert_model is None:
        _distilbert_model = _load_artifact(config.DISTILBERT_MODEL_PATH, prefer_torch=True)
    return _distilbert_model


def _get_roberta_model():
    """Load the RoBERTa model from disk if it hasn't been loaded yet, and return it."""
    global _roberta_model
    if _roberta_model is None:
        _roberta_model = _load_artifact(config.ROBERTA_MODEL_PATH)
    return _roberta_model


def _get_xgboost_artifacts():
    """Load and cache XGBoost model/vectorizer on first use."""
    global _xgboost_model, _xgboost_vectorizer
    if _xgboost_model is None or _xgboost_vectorizer is None:
        _xgboost_model = joblib.load(config.XGBOOST_MODEL_PATH)
        _xgboost_vectorizer = joblib.load(config.XGBOOST_VECTORIZER_PATH)
    return _xgboost_model, _xgboost_vectorizer


def predict(text: str, model_type: str = "lr") -> dict:
    """Predict the probability of a mental health signal
    in the given text using the specified model type."""
    if model_type == "lr":
        return predictor.lr_predict(_lr_model, _lr_vectorizer, text)
    if model_type == "distilbert":
        return predictor.distilbert_predict(_get_distilbert_model(), text)
    if model_type == "roberta":
        return predictor.roberta_predict(_get_roberta_model(), text)
    if model_type == "xgboost":
        xgb_model, xgb_vectorizer = _get_xgboost_artifacts()
        return predictor.xgboost_predict(xgb_model, xgb_vectorizer, text)

    raise ValueError(f"Unsupported model_type: {model_type}")
