"""
Évaluation des modèles : métriques standard + métriques de sécurité clinique.

Métriques ajoutées v3 :
  - Sensitivité (recall classe 1) — métrique primaire de sécurité
  - Spécificité (recall classe 0)
  - AUC-ROC
  - Brier score (calibration des probabilités)
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
    recall_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    brier_score_loss,
)


REPORTS_DIR = Path("reports")


def evaluate_baseline(pipeline, test_df: pd.DataFrame) -> dict:
    """
    Évalue le pipeline baseline et retourne les métriques cliniques complètes.

    Métriques de sécurité :
      - recall_1 (sensitivité) : proportion de vrais positifs détectés
        → Un recall_1 faible = cas de détresse non détectés → risque clinique
      - recall_0 (spécificité) : proportion de vrais négatifs correctement classés
      - auc_roc : qualité globale de discrimination indépendante du seuil
      - brier_score : calibration (0=parfait, 0.25=baseline aléatoire)
    """
    y_true = test_df["label"]
    y_pred = pipeline.predict(test_df["text"])
    y_proba = pipeline.predict_proba(test_df["text"])[:, 1]

    metrics = {
        "accuracy":        accuracy_score(y_true, y_pred),
        "f1_weighted":     f1_score(y_true, y_pred, average="weighted"),
        "f1_macro":        f1_score(y_true, y_pred, average="macro"),
        # Sécurité clinique
        "recall_1":        recall_score(y_true, y_pred, pos_label=1),
        "recall_0":        recall_score(y_true, y_pred, pos_label=0),
        "auc_roc":         roc_auc_score(y_true, y_proba),
        "brier_score":     brier_score_loss(y_true, y_proba),
    }

    logger.info("=" * 60)
    logger.info("  ÉVALUATION BASELINE")
    logger.info(f"  Accuracy       : {metrics['accuracy']:.4f}")
    logger.info(f"  F1 Macro       : {metrics['f1_macro']:.4f}")
    logger.info(f"  Sensitivité    : {metrics['recall_1']:.4f}  ← sécurité clinique")
    logger.info(f"  Spécificité    : {metrics['recall_0']:.4f}")
    logger.info(f"  AUC-ROC        : {metrics['auc_roc']:.4f}")
    logger.info(f"  Brier score    : {metrics['brier_score']:.4f}  ← calibration (0=parfait)")
    logger.info("=" * 60)
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


def plot_calibration(pipeline, test_df: pd.DataFrame, save: bool = True) -> None:
    """
    Courbe de calibration : vérifie si predict_proba(0.7) = 70% de vrais positifs.
    Un modèle mal calibré donne des scores de détresse peu interprétables cliniquement.
    """
    if not _MATPLOTLIB_AVAILABLE:
        return
    from sklearn.calibration import calibration_curve

    y_true = test_df["label"]
    y_proba = pipeline.predict_proba(test_df["text"])[:, 1]

    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Calibration parfaite")
    ax.plot(prob_pred, prob_true, "s-", label="Baseline TF-IDF+LR")
    ax.set_xlabel("Score prédit (probabilité)")
    ax.set_ylabel("Fraction de vrais positifs")
    ax.set_title("Calibration des probabilités")
    ax.legend()

    if save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "calibration_curve.png"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Courbe de calibration sauvegardée → {path}")
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
    """Retourne le score de risque pour un texte donné (usage évaluation/tests).

    Applique le même preprocessing que le pipeline d'inférence (prepare_text +
    clean_text) pour éviter un mismatch train/infer si le texte est brut ou en
    français. L'API utilise src.training.predict.predict() qui fait ce travail.
    """
    from src.common.language import prepare_text
    from src.training.preprocess import clean_text

    text_en, _ = prepare_text(text)
    text_clean = clean_text(text_en)
    proba = pipeline.predict_proba([text_clean])[0]
    return {
        "label": int(np.argmax(proba)),
        "score_distress": float(proba[1]),
        "score_non_distress": float(proba[0]),
    }
