import numpy as np
import torch

from src.training import predict as predict_module


class FakeVectorizer:
    def __init__(self):
        self.seen = None

    def transform(self, values):
        self.seen = values
        return np.array([[1.0]])


class FakeLRModel:
    def __init__(self):
        self.seen = None

    def predict_proba(self, features):
        self.seen = features
        return np.array([[0.2, 0.8]])


def test_lr_predict():
    """lr_predict preprocesses, vectorizes and returns thresholded output."""
    model = FakeLRModel()
    vectorizer = FakeVectorizer()

    result = predict_module.lr_predict(
        model,
        vectorizer,
        "raw input",
        preprocess_fn=lambda text: f"clean::{text}",
    )

    assert result == {"label": 1, "probability": 0.8}
    assert vectorizer.seen == ["clean::raw input"]
    assert model.seen.shape == (1, 1)


class FakeTokenizer:
    def __init__(self):
        self.seen_text = None

    def __call__(self, text, return_tensors, truncation, padding, max_length):
        self.seen_text = text
        return {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }


class FakeDistilBertModel:
    def __init__(self):
        self.eval_called = False
        self.seen_inputs = None

    def eval(self):
        self.eval_called = True

    def __call__(self, **inputs):
        self.seen_inputs = inputs
        return type("Output", (), {"logits": torch.tensor([[0.1, 2.0]])})()


def test_distilbert_predict():
    """distilbert_predict tokenizes input and computes class-1 probability."""
    tokenizer = FakeTokenizer()
    model = FakeDistilBertModel()

    result = predict_module.distilbert_predict(
        model,
        "hello",
        tokenizer=tokenizer,
        preprocess_fn=lambda text: f"prep::{text}",
    )

    assert model.eval_called is True
    assert tokenizer.seen_text == "prep::hello"
    assert set(model.seen_inputs) == {"input_ids", "attention_mask"}
    assert result["label"] == 1
    assert 0.5 < result["probability"] < 1.0
