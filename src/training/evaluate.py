"""
Évaluation des modèles : accuracy, F1, matrice de confusion, SHAP.
"""

import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False
from pathlib import Path
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


REPORTS_DIR = Path("reports")


def evaluate_baseline(pipeline, test_df: pd.DataFrame) -> dict:
    """Évalue le pipeline baseline et retourne les métriques."""
    y_true = test_df["label"]
    y_pred = pipeline.predict(test_df["text"])

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

    logger.info(f"Baseline metrics: {metrics}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['non-détresse', 'détresse'])}")
    return metrics


def plot_confusion_matrix(pipeline, test_df: pd.DataFrame, save: bool = True) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib non disponible — confusion matrix ignorée")
        return

    y_true = test_df["label"]
    y_pred = pipeline.predict(test_df["text"])

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["non-détresse", "détresse"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Matrice de confusion — Baseline")

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "confusion_matrix_baseline.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Matrice de confusion sauvegardée → {path}")
    plt.close(fig)


def explain_with_shap(pipeline, texts: list[str], n_samples: int = 100) -> None:
    """
    Génère une explication SHAP pour le pipeline TF-IDF + LR.
    Affiche un summary plot des features les plus importantes.
    """
    import shap

    logger.info("Calcul SHAP...")
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    X = vectorizer.transform(texts[:n_samples])
    explainer = shap.LinearExplainer(clf, X, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X)

    feature_names = vectorizer.get_feature_names_out()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if _MATPLOTLIB_AVAILABLE:
        plt.savefig(REPORTS_DIR / "shap_summary.png", bbox_inches="tight")
        logger.info(f"SHAP summary sauvegardé → {REPORTS_DIR / 'shap_summary.png'}")
        plt.close()


def predict_proba_text(pipeline, text: str) -> dict:
    """Retourne le score de risque pour un texte donné (utilisé par l'API)."""
    proba = pipeline.predict_proba([text])[0]
    return {
        "label": int(np.argmax(proba)),
        "score_distress": float(proba[1]),
        "score_non_distress": float(proba[0]),
    }
