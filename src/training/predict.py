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


def load_model(model_type: str = "baseline"):
    settings = get_settings()
    if model_type == "baseline":
        joblib_path = Path("models/baseline.joblib")
        path = joblib_path if joblib_path.exists() else Path("models/baseline.pkl")
        return joblib.load(path)
    elif model_type == "distilbert":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(settings.model_path)
        model = AutoModelForSequenceClassification.from_pretrained(settings.model_path)
        model.eval()
        return {"tokenizer": tokenizer, "model": model}
    else:
        raise ValueError(f"model_type inconnu : {model_type}")


def predict(text: str, model=None, model_type: str = "baseline") -> dict:
    """
    Prédit le score de risque pour un texte.
    Retourne : {"label": 0|1, "score_distress": float, "model": str}
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
        # DistilBERT ne supporte pas token_type_ids
        inputs.pop("token_type_ids", None)
        with torch.no_grad():
            logits = bert(**inputs).logits
        proba = torch.softmax(logits, dim=-1)[0].tolist()
        label = int(proba[1] > 0.5)
        score = float(proba[1])

    logger.debug(f"Prédiction [{model_type}] lang={detected_lang} → label={label}, score={score:.3f}")
    return {"label": label, "score_distress": score, "model": model_type, "detected_lang": detected_lang}
