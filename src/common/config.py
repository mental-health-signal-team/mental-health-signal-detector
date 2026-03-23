from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LR_MODEL_PATH = MODELS_DIR / "depression_classifier.pkl"
DISTILBERT_MODEL_PATH = MODELS_DIR / "distilbert_model.pkl"
ROBERTA_MODEL_PATH = MODELS_DIR / "mental_roberta_base.pkl"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgb_depression_classifier.pkl"
XGBOOST_VECTORIZER_PATH = MODELS_DIR / "xgb_tfidf_vectorizer.pkl"
