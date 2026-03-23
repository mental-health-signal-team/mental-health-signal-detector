from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
# Keep LR model aligned with tfidf_vectorizer.pkl (both 5,000 features).
LR_MODEL_PATH = MODELS_DIR / "depression_classifier.pkl"
DISTILBERT_MODEL_PATH = MODELS_DIR / "distilbert_model.pkl"
