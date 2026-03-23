import html
import io
import pickle
import re
from typing import Any, Mapping

import joblib
import torch
from transformers import AutoTokenizer

import src.common.config as config
import src.training.predict as predictor
from src.training.preprocess import preprocess_text

_lr_model = joblib.load(config.LR_MODEL_PATH)
_lr_vectorizer = joblib.load(config.VECTORIZER_PATH)
_distilbert_model = None  # Lazy load the DistilBERT model when needed
_distilbert_tokenizer = None
_predictor_module = None


class CPUUnpickler(pickle.Unpickler):
    """Custom unpickler to load PyTorch models on CPU."""

    def find_class(self, module, name):
        """Override the find_class method to handle loading PyTorch models on CPU."""
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _get_distilbert_model():
    """Load the DistilBERT model from disk if it hasn't been loaded yet, and return it."""
    global _distilbert_model
    if _distilbert_model is None:
        with open(config.DISTILBERT_MODEL_PATH, "rb") as f:
            _distilbert_model = CPUUnpickler(f).load()
    return _distilbert_model


def _get_distilbert_tokenizer():
    """Load DistilBERT tokenizer lazily and return it."""
    global _distilbert_tokenizer
    if _distilbert_tokenizer is None:
        _distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return _distilbert_tokenizer


def predict(text: str, model_type: str = "lr") -> dict:
    """Predict the probability of a mental health signal
    in the given text using the specified model type."""
    if model_type == "lr":
        return predictor.lr_predict(_lr_model, _lr_vectorizer, text)
    elif model_type == "distilbert":
        return predictor.distilbert_predict(_get_distilbert_model(), text)


def color_text_full(
    text: str,
    word_importance: Mapping[str, float],
    vectorizer: Any,
    threshold: float = 0.005,
) -> str:
    """Color words using SHAP importance values from a fitted vectorizer."""
    analyzer = vectorizer.build_analyzer()

    words = re.findall(r"\w+|\W+", text)
    colored_words = []

    for word in words:
        clean_tokens = analyzer(word)

        if clean_tokens:
            token = clean_tokens[0]
            importance = float(word_importance.get(token, 0.0))
        else:
            importance = 0.0

        if importance > threshold:
            color = "red"
        elif importance < -threshold:
            color = "green"
        else:
            color = "white"

        safe_word = html.escape(word)
        colored_word = f'<span style="color:{color}">{safe_word}</span>'
        colored_words.append(colored_word)

    return "".join(colored_words)


def _risk_level(probability: float) -> str:
    """Convert depression probability into categorical risk level."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


def _lr_word_importance(text: str) -> dict[str, float]:
    """Compute local LR contributions for tokens present in input text."""
    preprocessed_text = preprocess_text(text)
    features = _lr_vectorizer.transform([preprocessed_text])
    coeffs = _lr_model.coef_[0]
    feature_names = _lr_vectorizer.get_feature_names_out()
    contribution_vector = features.multiply(coeffs).tocsr()

    return {feature_names[idx]: float(value) for idx, value in zip(contribution_vector.indices, contribution_vector.data)}


def _distilbert_word_importance(text: str, max_tokens: int = 40) -> dict[str, float]:
    """Compute token importance for DistilBERT with gradient x input attribution."""
    model = _get_distilbert_model()
    tokenizer = _get_distilbert_tokenizer()
    preprocessed_text = preprocess_text(text)

    encoded = tokenizer(
        preprocessed_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    embedding_layer = model.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids).detach()
    inputs_embeds.requires_grad_(True)

    model.zero_grad()
    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = int(torch.argmax(logits, dim=-1).item())
    target_logit = logits[0, predicted_class]
    target_logit.backward()

    grads = inputs_embeds.grad[0]
    token_scores = (grads * inputs_embeds[0]).sum(dim=-1).detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    special_tokens = set(tokenizer.all_special_tokens)

    merged: dict[str, float] = {}
    current_word = ""
    current_score = 0.0

    for token, score in zip(tokens, token_scores):
        if token in special_tokens:
            if current_word:
                key = current_word.lower()
                merged[key] = merged.get(key, 0.0) + float(current_score)
                current_word = ""
                current_score = 0.0
            continue

        if token.startswith("##"):
            piece = token[2:]
            current_word += piece
            current_score += float(score)
            continue

        if current_word:
            key = current_word.lower()
            merged[key] = merged.get(key, 0.0) + float(current_score)

        current_word = token
        current_score = float(score)

    if current_word:
        key = current_word.lower()
        merged[key] = merged.get(key, 0.0) + float(current_score)

    if len(merged) > max_tokens:
        top_items = sorted(merged.items(), key=lambda item: abs(float(item[1])), reverse=True)[:max_tokens]
        return dict(top_items)
    return merged


def _color_text_distilbert(
    text: str,
    word_importance: Mapping[str, float],
    threshold: float = 0.005,
) -> str:
    """Color text for DistilBERT explanation using lowercased word matching."""
    words = re.findall(r"\w+|\W+", text)
    colored_words = []

    for word in words:
        key = word.lower()
        importance = float(word_importance.get(key, 0.0)) if key.isalnum() else 0.0

        if importance > threshold:
            color = "red"
        elif importance < -threshold:
            color = "green"
        else:
            color = "white"

        safe_word = html.escape(word)
        colored_words.append(f'<span style="color:{color}">{safe_word}</span>')

    return "".join(colored_words)


def explain(text: str, model_type: str = "lr", threshold: float = 0.005, max_tokens: int = 40) -> dict:
    """Predict and return a colorized token-importance explanation payload."""
    if threshold < 0:
        raise ValueError("threshold must be >= 0.")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0.")

    prediction = predict(text, model_type)
    label = int(prediction["label"])
    probability = float(prediction["probability"])

    if label == 1:
        display_confidence = probability
        confidence_label = "distress"
    else:
        display_confidence = 1.0 - probability
        confidence_label = "no_distress"

    note = None
    if model_type == "lr":
        word_importance = _lr_word_importance(text)
        vectorizer = _lr_vectorizer
        colored_html = color_text_full(text, word_importance, vectorizer, threshold=threshold)
    elif model_type == "distilbert":
        word_importance = _distilbert_word_importance(text, max_tokens=max_tokens)
        colored_html = _color_text_distilbert(text, word_importance, threshold=threshold)
        note = "DistilBERT importance is gradient-based and can vary slightly between runs and tokenization."  # noqa: E501
    else:
        raise ValueError("Unsupported model_type. Use 'lr' or 'distilbert'.")

    return {
        "label": label,
        "probability": probability,
        "display_confidence": display_confidence,
        "confidence_label": confidence_label,
        "risk_level": _risk_level(probability),
        "colored_html": colored_html,
        "word_importance": word_importance,
        "note": note,
    }
