import io
import pickle
from pathlib import Path

import joblib
import torch
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory, detect
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import src.common.config as config
import src.training.predict as predictor

_lr_model = joblib.load(config.LR_MODEL_PATH)
_lr_vectorizer = joblib.load(config.VECTORIZER_PATH)
_distilbert_model = None  # Lazy load the DistilBERT model when needed
_distilbert_backend = "not_loaded"
_distilbert_last_error = None
_roberta_model = None
_roberta_backend = "not_loaded"
_roberta_last_error = None
_xgboost_model = None
_xgboost_vectorizer = None

DetectorFactory.seed = 0


def _distilbert_local_files_status() -> dict:
    """Return status of expected local Hugging Face DistilBERT files."""
    local_dir = Path(config.DISTILBERT_LOCAL_DIR)
    required = {
        "config.json": (local_dir / "config.json").exists(),
        "tokenizer_config.json": (local_dir / "tokenizer_config.json").exists(),
        "tokenizer.json": (local_dir / "tokenizer.json").exists(),
    }
    has_weights = (local_dir / "model.safetensors").exists() or (local_dir / "pytorch_model.bin").exists()
    return {
        "directory": str(local_dir),
        "directory_exists": local_dir.is_dir(),
        "required_files": required,
        "has_weights": has_weights,
        "ready": local_dir.is_dir() and all(required.values()) and has_weights,
    }


def _roberta_local_files_status() -> dict:
    """Return status of expected local Hugging Face RoBERTa files."""
    local_dir = Path(config.ROBERTA_LOCAL_DIR)
    required = {
        "config.json": (local_dir / "config.json").exists(),
        "tokenizer_config.json": (local_dir / "tokenizer_config.json").exists(),
        "tokenizer.json": (local_dir / "tokenizer.json").exists(),
    }
    has_weights = (local_dir / "model.safetensors").exists() or (local_dir / "pytorch_model.bin").exists()
    return {
        "directory": str(local_dir),
        "directory_exists": local_dir.is_dir(),
        "required_files": required,
        "has_weights": has_weights,
        "ready": local_dir.is_dir() and all(required.values()) and has_weights,
    }


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
        except AttributeError:
            # Backward/forward compatibility for Transformers class renames.
            if module == "transformers.models.roberta.modeling_roberta" and "Sdpa" in name:
                alias = name.replace("Sdpa", "")
                try:
                    roberta_mod = __import__(module, fromlist=[alias])
                    return getattr(roberta_mod, alias)
                except Exception:  # noqa: BLE001
                    pass

            if "roberta" in module.lower() or "sdpa" in name.lower():
                class _CompatMissingModule(torch.nn.Module):
                    def __init__(self, *args, **kwargs):
                        super().__init__()

                    def forward(self, *args, **kwargs):
                        raise RuntimeError(
                            f"Missing transformer class during unpickling: {module}.{name}. "
                            "Use a compatible transformers version for this artifact."
                        )

                _CompatMissingModule.__name__ = name
                return _CompatMissingModule
            raise


def _get_distilbert_model():
    """Load the DistilBERT model from disk if it hasn't been loaded yet, and return it."""
    global _distilbert_model, _distilbert_backend, _distilbert_last_error
    if _distilbert_model is None:
        status = _distilbert_local_files_status()
        try:
            if status["ready"]:
                local_dir = Path(status["directory"])
                tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(str(local_dir), local_files_only=True)
                _distilbert_model = (model, tokenizer)
                _distilbert_backend = "local_files"
            else:
                _distilbert_model = _load_artifact(config.DISTILBERT_MODEL_PATH, prefer_torch=True)
                _distilbert_backend = "pickle"
            _distilbert_last_error = None
        except Exception as exc:  # noqa: BLE001
            _distilbert_backend = "load_failed"
            _distilbert_last_error = f"{type(exc).__name__}: {exc}"
            raise
    return _distilbert_model


def _get_roberta_model():
    """Load the RoBERTa model from disk if it hasn't been loaded yet, and return it."""
    global _roberta_model, _roberta_backend, _roberta_last_error
    if _roberta_model is None:
        status = _roberta_local_files_status()
        try:
            if status["ready"]:
                local_dir = Path(status["directory"])
                tokenizer = AutoTokenizer.from_pretrained(str(local_dir), local_files_only=True)
                model = AutoModelForSequenceClassification.from_pretrained(str(local_dir), local_files_only=True)
                _roberta_model = (model, tokenizer)
                _roberta_backend = "local_files"
            else:
                _roberta_model = _load_artifact(config.ROBERTA_MODEL_PATH)
                _roberta_backend = "pickle"
            _roberta_last_error = None
        except Exception as exc:  # noqa: BLE001
            _roberta_backend = "load_failed"
            _roberta_last_error = f"{type(exc).__name__}: {exc}"
            raise
    return _roberta_model


def get_roberta_diagnostics(load_model: bool = False) -> dict:
    """Return explicit diagnostics about which RoBERTa backend is used."""
    global _roberta_backend

    if load_model:
        try:
            _get_roberta_model()
        except Exception:  # noqa: BLE001
            pass

    loaded = _roberta_model is not None
    if loaded and _roberta_backend == "not_loaded":
        _roberta_backend = "local_files" if isinstance(_roberta_model, tuple) else "pickle"

    return {
        "backend": _roberta_backend,
        "loaded": loaded,
        "local_files": _roberta_local_files_status(),
        "pickle_path": str(config.ROBERTA_MODEL_PATH),
        "pickle_exists": Path(config.ROBERTA_MODEL_PATH).exists(),
        "last_error": _roberta_last_error,
    }


def get_distilbert_diagnostics(load_model: bool = False) -> dict:
    """Return explicit diagnostics about which DistilBERT backend is used."""
    global _distilbert_backend

    if load_model:
        try:
            _get_distilbert_model()
        except Exception:  # noqa: BLE001
            pass

    loaded = _distilbert_model is not None
    if loaded and _distilbert_backend == "not_loaded":
        _distilbert_backend = "local_files" if isinstance(_distilbert_model, tuple) else "pickle"

    return {
        "backend": _distilbert_backend,
        "loaded": loaded,
        "local_files": _distilbert_local_files_status(),
        "pickle_path": str(config.DISTILBERT_MODEL_PATH),
        "pickle_exists": Path(config.DISTILBERT_MODEL_PATH).exists(),
        "last_error": _distilbert_last_error,
    }


def _get_xgboost_artifacts():
    """Load and cache XGBoost model/vectorizer on first use."""
    global _xgboost_model, _xgboost_vectorizer
    if _xgboost_model is None or _xgboost_vectorizer is None:
        _xgboost_model = joblib.load(config.XGBOOST_MODEL_PATH)
        _xgboost_vectorizer = joblib.load(config.XGBOOST_VECTORIZER_PATH)
    return _xgboost_model, _xgboost_vectorizer


def _prepare_text_for_inference(text: str) -> dict:
    """Detect language and translate non-English text to English for inference."""
    source_language = "unknown"
    was_translated = False
    translated_text = None
    text_for_inference = text

    try:
        source_language = detect(text) if text and text.strip() else "unknown"
    except Exception:  # noqa: BLE001
        source_language = "unknown"

    if source_language not in {"unknown", "en"}:
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            if translated and isinstance(translated, str):
                translated_text = translated
                text_for_inference = translated
                was_translated = True
        except Exception:  # noqa: BLE001
            # If translation fails, continue with original text instead of failing prediction.
            pass

    return {
        "text_for_inference": text_for_inference,
        "source_language": source_language,
        "analysis_language": "en",
        "was_translated": was_translated,
        "translated_text": translated_text,
    }


def predict(text: str, model_type: str = "lr") -> dict:
    """Predict the probability of a mental health signal
    in the given text using the specified model type."""
    prepared = _prepare_text_for_inference(text)
    text_for_inference = prepared["text_for_inference"]

    if model_type == "lr":
        result = predictor.lr_predict(_lr_model, _lr_vectorizer, text_for_inference)
    elif model_type == "distilbert":
        result = predictor.distilbert_predict(_get_distilbert_model(), text_for_inference)
    elif model_type == "roberta":
        result = predictor.roberta_predict(_get_roberta_model(), text_for_inference)
    elif model_type == "xgboost":
        xgb_model, xgb_vectorizer = _get_xgboost_artifacts()
        result = predictor.xgboost_predict(xgb_model, xgb_vectorizer, text_for_inference)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    result.update(
        {
            "source_language": prepared["source_language"],
            "analysis_language": prepared["analysis_language"],
            "was_translated": prepared["was_translated"],
            "translated_text": prepared["translated_text"],
        }
    )
    return result
