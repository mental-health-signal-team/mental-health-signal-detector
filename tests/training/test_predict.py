from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch

from src.training import predict as predict_module


def test_lr_predict():
    """lr_predict preprocesses, vectorizes and returns thresholded output."""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.2, 0.8]])
    vectorizer = Mock()
    vectorizer.transform.return_value = np.array([[1.0]])

    result = predict_module.lr_predict(
        model,
        vectorizer,
        "raw input",
        preprocess_fn=lambda text: f"clean::{text}",
    )

    assert result == {"label": 1, "probability": 0.8}
    vectorizer.transform.assert_called_once_with(["clean::raw input"])
    model.predict_proba.assert_called_once()
    assert model.predict_proba.call_args.args[0].shape == (1, 1)


def test_distilbert_predict():
    """distilbert_predict tokenizes input and computes class-1 probability."""
    tokenizer = Mock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    model = Mock()
    model.return_value = SimpleNamespace(logits=torch.tensor([[0.1, 2.0]]))

    result = predict_module.distilbert_predict(
        model,
        "hello",
        tokenizer=tokenizer,
        preprocess_fn=lambda text: f"prep::{text}",
    )

    model.eval.assert_called_once()
    tokenizer.assert_called_once_with(
        "prep::hello",
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512,
    )
    assert set(model.call_args.kwargs) == {"input_ids", "attention_mask"}
    assert result["label"] == 1
    assert 0.5 < result["probability"] < 1.0
