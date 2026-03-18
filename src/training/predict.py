import torch
from transformers import AutoTokenizer

from src.training.preprocess import preprocess_text


def lr_predict(model, vectorizer, text: str) -> dict:
    """Predict the probability of a mental health signal in the given text using a trained logistic regression model.
    - The function loads the trained model and vectorizer from disk,
    preprocesses the input text, and returns a dictionary containing the predicted probability of a mental health signal."""

    preprocessed_text = preprocess_text(text)
    features = vectorizer.transform([preprocessed_text])
    probability = model.predict_proba(features)[0][1]
    return {"label": int(probability >= 0.5), "probability": probability}


def distilbert_predict(model, text: str) -> dict:
    """Predict the probability of a mental health signal in the given text using
        a trained DistilBERT model.
    - The function loads the trained DistilBERT model and tokenizer from disk,
    preprocesses the input text, and returns a dictionary containing the predicted
    probability of a mental health signal."""
    # Placeholder implementation - replace with actual DistilBERT prediction logic
    preprocessed_text = preprocess_text(text, model_type="distilbert")
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
