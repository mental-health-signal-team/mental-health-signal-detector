"""
Inférence : charge le modèle et prédit sur un texte.
Supporte le baseline (pkl) et DistilBERT (HuggingFace).
"""

from pathlib import Path

import joblib
from loguru import logger

from src.common.config import get_settings
from src.common.language import prepare_text
from src.training.preprocess import clean_text

# Répertoire de confiance pour les modèles — protège contre le path traversal.
# Utilise __file__ (indépendant du CWD au démarrage) pour pointer vers
# <project_root>/models/, quel que soit le répertoire de lancement du process.
_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"


def _safe_load_joblib(path: Path):
    """Load a joblib file only if it resolves inside the trusted models directory.

    Prevents path-traversal attacks by refusing any path whose resolved form
    falls outside ``_MODELS_DIR``.

    Args:
        path: Candidate file path to load.

    Returns:
        The deserialized Python object stored in the joblib file.

    Raises:
        ValueError: If the resolved path is outside the trusted models
            directory.
        FileNotFoundError: If the resolved path does not exist on disk.
    """
    resolved = path.resolve()
    if not resolved.is_relative_to(_MODELS_DIR.resolve()):
        raise ValueError(f"Chemin de modèle non autorisé : {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable : {resolved}")
    return joblib.load(resolved)


def load_model(model_type: str = "baseline"):
    """Load and return the requested model from disk.

    Supported model types:

    * ``"baseline"`` — sklearn ``Pipeline`` (TF-IDF + Logistic Regression)
      loaded from ``models/baseline.joblib`` (falls back to
      ``models/baseline.pkl``).
    * ``"distilbert"`` — HuggingFace ``DistilBertForSequenceClassification``
      loaded from the path configured in ``settings.model_path``.
    * ``"mental_bert_v3"`` — HuggingFace ``BertForSequenceClassification``
      loaded from the path configured in ``settings.model_path_v3``.

    Args:
        model_type: Identifier for the model to load.  Defaults to
            ``"baseline"``.

    Returns:
        For ``"baseline"``: a fitted sklearn ``Pipeline`` object.
        For transformer models: a dict with keys ``"tokenizer"`` and
        ``"model"`` (the HuggingFace tokenizer and model set to eval mode).

    Raises:
        ValueError: If ``model_type`` is not one of the recognised identifiers.
        FileNotFoundError: If the baseline model file cannot be found on disk.
    """
    settings = get_settings()
    if model_type == "baseline":
        joblib_path = _MODELS_DIR / "baseline.joblib"
        path = joblib_path if joblib_path.exists() else _MODELS_DIR / "baseline.pkl"
        return _safe_load_joblib(path)
    elif model_type == "distilbert":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}
    elif model_type == "mental_bert_v3":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(settings.model_path_v3)
        model = AutoModelForSequenceClassification.from_pretrained(settings.model_path_v3)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}
    else:
        raise ValueError(f"model_type inconnu : {model_type}")


def predict(text: str, model=None, model_type: str = "baseline") -> dict:
    """Predict the mental-health distress risk score for the given text.

    Applies language detection and text cleaning before inference.  If no
    pre-loaded model is provided, the model is loaded from disk via
    :func:`load_model`.

    Args:
        text: Raw input text in any supported language.
        model: Optional pre-loaded model object.  When ``None`` the model is
            loaded automatically using ``model_type``.  For ``"baseline"`` this
            must be a fitted sklearn ``Pipeline``; for transformer models it
            must be the dict returned by :func:`load_model`.
        model_type: Identifier for the model to use when ``model`` is
            ``None``.  One of ``"baseline"``, ``"distilbert"``, or
            ``"mental_bert_v3"``.  Defaults to ``"baseline"``.

    Returns:
        A dict with the following keys:

        * ``"label"`` (``int``): ``1`` if distress is detected, ``0``
          otherwise.
        * ``"score_distress"`` (``float``): Probability of the distress class
          in ``[0.0, 1.0]``.
        * ``"model"`` (``str``): The ``model_type`` identifier used.
        * ``"detected_lang"`` (``str``): BCP-47 language code detected in the
          input text (e.g. ``"en"``, ``"fr"``).

    Raises:
        ValueError: If ``model_type`` is unrecognised and no ``model`` is
            provided (propagated from :func:`load_model`).
    """
    if model is None:
        model = load_model(model_type)

    text_en, detected_lang = prepare_text(text)
    text_clean = clean_text(text_en)

    if model_type == "baseline":
        proba = model.predict_proba([text_clean])[0]
        label = int(proba.argmax())
        score = float(proba[1])
    else:
        # Import torch uniquement pour DistilBERT — évite l'overhead au
        # démarrage et pour chaque requête baseline (~1-2s d'import).
        import torch
        tokenizer = model["tokenizer"]
        bert = model["model"]
        inputs = tokenizer(
            text_clean,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )
        # DistilBERT ne supporte pas token_type_ids ; BERT (mental_bert_v3) les utilise
        if model_type == "distilbert":
            inputs.pop("token_type_ids", None)
        with torch.no_grad():
            logits = bert(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0].tolist()
        label = int(proba[1] > 0.5)
        score = float(proba[1])

    logger.debug(f"Prédiction [{model_type}] lang={detected_lang} → label={label}, score={score:.3f}")
    return {"label": label, "score_distress": score, "model": model_type, "detected_lang": detected_lang}
