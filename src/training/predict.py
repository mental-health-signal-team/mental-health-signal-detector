import torch
from transformers import AutoTokenizer

from src.training.preprocess import preprocess_text


def lr_predict(model, vectorizer, text: str, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a trained logistic regression pipeline."""
    preprocessed_text = preprocess_fn(text)
    features = vectorizer.transform([preprocessed_text])
    probability = model.predict_proba(features)[0][1]
    return {"label": int(probability >= 0.5), "probability": probability}


def distilbert_predict(model, text: str, tokenizer=None, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a trained DistilBERT classifier."""
    preprocessed_text = preprocess_fn(text)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    inputs = tokenizer(
        preprocessed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        probability = probabilities[0][1].item()
    return {"label": int(probability >= 0.5), "probability": probability}


def roberta_predict(model, text: str, tokenizer=None, preprocess_fn=preprocess_text) -> dict:
    """Predict with a RoBERTa artifact, supporting transformers, sklearn, or pipeline objects."""
    preprocessed_text = preprocess_fn(text)

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba([preprocessed_text])[0][1])
        return {"label": int(probability >= 0.5), "probability": probability}

    if hasattr(model, "predict") and not hasattr(model, "eval"):
        predicted = float(model.predict([preprocessed_text])[0])
        probability = max(0.0, min(1.0, predicted))
        return {"label": int(probability >= 0.5), "probability": probability}

    if callable(model) and not hasattr(model, "eval"):
        result = model(preprocessed_text)
        if isinstance(result, list) and result and isinstance(result[0], dict):
            score = float(result[0].get("score", 0.0))
            label = str(result[0].get("label", "")).lower()
            if any(token in label for token in ["1", "pos", "depress", "true"]):
                probability = score
            else:
                probability = 1.0 - score
            probability = max(0.0, min(1.0, probability))
            return {"label": int(probability >= 0.5), "probability": probability}

    if tokenizer is None:
        tokenizer_name = getattr(getattr(model, "config", None), "_name_or_path", "roberta-base")

        # Some fine-tuned checkpoints keep a gated/private repo id in _name_or_path.
        # For RoBERTa fine-tunes, using the base tokenizer is usually compatible.
        if isinstance(tokenizer_name, str) and tokenizer_name.startswith("mental/"):
            tokenizer_name = "roberta-base"

        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as exc:  # noqa: BLE001
            fallback_name = "roberta-base"
            if tokenizer_name != fallback_name:
                tokenizer = AutoTokenizer.from_pretrained(fallback_name)
            else:
                raise ValueError(
                    "Unable to load a RoBERTa tokenizer. "
                    "Bundle tokenizer files with the model artifact or ensure Hugging Face access."
                ) from exc

    inputs = tokenizer(
        preprocessed_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        probability = probabilities[0][1].item()
    return {"label": int(probability >= 0.5), "probability": probability}


def xgboost_predict(model, vectorizer, text: str, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a trained XGBoost classifier."""
    preprocessed_text = preprocess_fn(text)
    features = vectorizer.transform([preprocessed_text])
    probability = model.predict_proba(features)[0][1]
    return {"label": int(probability >= 0.5), "probability": probability}
