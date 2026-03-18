"""
Entraînement des modèles.

Deux modèles :
  - Baseline : Logistic Regression + TF-IDF (rapide, interprétable)
  - Avancé   : Transformers fine-tuned (DistilBERT ou Mental-BERT selon --model-name)

Axe v3 : Mental-BERT comme base, métrique de sécurité (sensitivité), mode clinical_only.
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
# Modèle avancé : Transformers fine-tuned (DistilBERT / Mental-BERT)
# ---------------------------------------------------------------------------

def train_distilbert(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "models/fine_tuned",
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    max_samples: int = 10_000,
) -> None:
    """
    Fine-tune un modèle Transformers sur le dataset de détresse mentale.

    Modèles recommandés :
      - "distilbert-base-uncased"               : rapide, bon baseline
      - "mental/mental-bert-base-uncased"        : pré-entraîné santé mentale Reddit
      - "facebook/roberta-base"                  : plus robuste au long terme
      - "almanach/camembert-base"                : pour entraînement natif français

    Métrique primaire : recall_1 (sensitivité) — on tolère plus les faux positifs
    que les faux négatifs dans un outil de détection de détresse.
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from datasets import Dataset
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

    logger.info(f"Fine-tuning {model_name} sur {max_samples} exemples max...")
    logger.info(f"lr={learning_rate} · warmup={warmup_ratio} · epochs={epochs} · batch={batch_size}")

    # Sous-échantillonnage si nécessaire
    if len(train_df) > max_samples:
        train_df = train_df.sample(n=max_samples, random_state=42, replace=False).reset_index(drop=True)
    if len(test_df) > max_samples // 4:
        test_df = test_df.sample(n=max_samples // 4, random_state=42).reset_index(drop=True)

    logger.info(f"Train: {len(train_df)} | Test: {len(test_df)}")
    logger.info(f"Train distribution:\n{train_df['label'].value_counts()}")

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
        # Probabilité de la classe 1 pour AUC
        proba_pos = logits[:, 1] if logits.ndim == 2 else logits
        try:
            auc = roc_auc_score(labels, proba_pos)
        except Exception:
            auc = 0.0

        return {
            "accuracy":      accuracy_score(labels, preds),
            "f1_macro":      f1_score(labels, preds, average="macro"),
            "f1_weighted":   f1_score(labels, preds, average="weighted"),
            # Métriques de sécurité clinique — priorité 1
            "recall_1":      recall_score(labels, preds, pos_label=1),   # sensitivité
            "recall_0":      recall_score(labels, preds, pos_label=0),   # spécificité
            "auc_roc":       auc,
        }

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,                      # L2 régularisation (anti-overfitting)
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="recall_1",       # Priorité sensitivité (sécurité clinique)
        greater_is_better=True,
        logging_dir="logs/hf",
        logging_steps=100,
        report_to="none",
        fp16=True,                              # Mixed precision si GPU disponible
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Résumé final des métriques
    metrics = trainer.evaluate()
    logger.info("=" * 60)
    logger.info(f"  Modèle         : {model_name}")
    logger.info(f"  Accuracy       : {metrics.get('eval_accuracy', 0):.4f}")
    logger.info(f"  F1 Macro       : {metrics.get('eval_f1_macro', 0):.4f}")
    logger.info(f"  Sensitivité    : {metrics.get('eval_recall_1', 0):.4f}  ← sécurité clinique")
    logger.info(f"  Spécificité    : {metrics.get('eval_recall_0', 0):.4f}")
    logger.info(f"  AUC-ROC        : {metrics.get('eval_auc_roc', 0):.4f}")
    logger.info("=" * 60)
    logger.info(f"Modèle sauvegardé → {output_dir}")


# ---------------------------------------------------------------------------
# Entrypoint CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "distilbert"], default="baseline")
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help=(
            "Modèle HuggingFace base. Recommandé : "
            "'mental/mental-bert-base-uncased' (domaine santé mentale) "
            "ou 'almanach/camembert-base' (natif français)"
        ),
    )
    parser.add_argument("--kaggle-path", type=str, default=None)
    parser.add_argument("--smhd-path", type=str, default=None)
    parser.add_argument("--erisk25-path", type=str, default=None)
    parser.add_argument("--no-dair", action="store_true", help="Exclure DAIR-AI/emotion")
    parser.add_argument("--go-emotions", action="store_true", help="Ajouter GoEmotions")
    parser.add_argument(
        "--clinical-only",
        action="store_true",
        help="Mode clinique pur : eRisk25 + Kaggle uniquement (exclut DAIR-AI et GoEmotions)",
    )
    parser.add_argument("--kaggle-samples", type=int, default=100_000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--output-dir", type=str, default="models/fine_tuned")
    args = parser.parse_args()

    train_df, test_df = build_dataset(
        kaggle_path=args.kaggle_path,
        use_dair=not args.no_dair and not args.clinical_only,
        use_go_emotions=args.go_emotions and not args.clinical_only,
        erisk25_path=args.erisk25_path,
        smhd_path=args.smhd_path,
        kaggle_max_samples=args.kaggle_samples,
    )

    if args.model == "baseline":
        pipeline = train_baseline(train_df)
        save_baseline(pipeline)
    else:
        train_distilbert(
            train_df,
            test_df,
            model_name=args.model_name,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
