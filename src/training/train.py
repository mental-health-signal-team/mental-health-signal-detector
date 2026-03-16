"""
Entraînement des modèles.

Deux modèles :
  - Baseline : Logistic Regression + TF-IDF (rapide, interprétable)
  - Avancé   : DistilBERT fine-tuned (HuggingFace Transformers)
"""

from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.training.preprocess import build_dataset


MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Baseline : Logistic Regression + TF-IDF
# ---------------------------------------------------------------------------

def train_baseline(
    train_df: pd.DataFrame,
    max_features: int = 50_000,
    C: float = 1.0,
) -> Pipeline:
    """Entraîne un pipeline TF-IDF + Logistic Regression."""
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=1000, class_weight="balanced")),
    ])
    pipeline.fit(train_df["text"], train_df["label"])
    logger.info("Baseline entraîné.")
    return pipeline


def save_baseline(pipeline: Pipeline, path: Path | None = None) -> Path:
    path = path or MODELS_DIR / "baseline.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path, compress=3)
    logger.info(f"Baseline sauvegardé → {path}")
    return path


def load_baseline(path: Path | None = None) -> Pipeline:
    # Rétrocompatibilité : accepte l'ancien .pkl s'il existe encore
    if path is None:
        joblib_path = MODELS_DIR / "baseline.joblib"
        path = joblib_path if joblib_path.exists() else MODELS_DIR / "baseline.pkl"
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Modèle avancé : DistilBERT fine-tuned
# ---------------------------------------------------------------------------

def train_distilbert(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/fine_tuned",
    epochs: int = 2,
    batch_size: int = 32,
    max_samples: int = 10_000,
) -> None:
    """Fine-tune DistilBERT sur le dataset de détresse mentale."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    logger.info(f"Fine-tuning {model_name} sur {max_samples} exemples max (CPU-friendly)...")

    # Sous-échantillonnage pour CPU
    if len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42, replace=False).reset_index(drop=True)
    if len(test_df) > max_samples // 4:
        test_df = test_df.sample(n=max_samples // 4, random_state=42).reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True)
    train_ds = train_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="weighted"),
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs/hf",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"DistilBERT sauvegardé → {output_dir}")


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "distilbert"], default="baseline")
    parser.add_argument("--kaggle-path", type=str, default=None, help="Chemin vers le CSV Kaggle")
    parser.add_argument("--smhd-path", type=str, default=None, help="Chemin vers le CSV SMHD")
    parser.add_argument("--no-dair", action="store_true", help="Ne pas utiliser DAIR-AI/emotion")
    parser.add_argument("--go-emotions", action="store_true", help="Ajouter GoEmotions au dataset")
    parser.add_argument("--kaggle-samples", type=int, default=100_000, help="Nb max de samples Kaggle")
    args = parser.parse_args()

    train_df, test_df = build_dataset(
        kaggle_path=args.kaggle_path,
        use_dair=not args.no_dair,
        use_go_emotions=args.go_emotions,
        smhd_path=args.smhd_path,
        kaggle_max_samples=args.kaggle_samples,
    )

    if args.model == "baseline":
        pipeline = train_baseline(train_df)
        save_baseline(pipeline)
    else:
        train_distilbert(train_df, test_df)
