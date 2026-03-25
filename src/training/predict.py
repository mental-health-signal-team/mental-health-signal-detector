import torch
from transformers import AutoTokenizer

from src.common import config
from src.training.preprocess import preprocess_text

_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


def lr_predict(model, vectorizer, text: str, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a trained logistic regression pipeline."""
    preprocessed_text = preprocess_fn(text)
    features = vectorizer.transform([preprocessed_text])
    probability = model.predict_proba(features)[0][1]
    return {"label": int(probability >= 0.5), "probability": probability}


def _transformer_predict(
    model,
    text: str,
    tokenizer_name: str,
    preprocess_fn=preprocess_text,
    tokenizer=None,
) -> dict:
    """Generic inference for any HuggingFace sequence classification model."""
    preprocessed_text = preprocess_fn(text)
    if tokenizer is None:
        if tokenizer_name not in _TOKENIZER_CACHE:
            _TOKENIZER_CACHE[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer = _TOKENIZER_CACHE[tokenizer_name]
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
        probabilities = torch.softmax(outputs.logits, dim=-1)
        probability = probabilities[0][1].item()
    return {"label": int(probability >= 0.5), "probability": probability}


def distilbert_predict(model, text: str, tokenizer=None, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a fine-tuned DistilBERT classifier."""
    return _transformer_predict(model, text, str(config.DISTILBERT_MODEL_HF_PATH), preprocess_fn, tokenizer)


def mental_roberta_predict(model, text: str, tokenizer=None, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a fine-tuned mental/mental-roberta-base classifier."""
    return _transformer_predict(model, text, str(config.MENTAL_ROBERTA_HF_PATH), preprocess_fn, tokenizer)


def xgboost_predict(model, vectorizer, text: str, preprocess_fn=preprocess_text) -> dict:
    """Predict class label/probability with a trained XGBoost classifier."""
    preprocessed_text = preprocess_fn(text)
    features = vectorizer.transform([preprocessed_text])
    probability = model.predict_proba(features)[0][1]
    return {"label": int(probability >= 0.5), "probability": probability}
