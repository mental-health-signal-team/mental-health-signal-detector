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
