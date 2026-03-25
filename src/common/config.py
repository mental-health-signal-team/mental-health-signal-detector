import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Google Drive folder ID for model storage.
GDRIVE_MODEL_FOLDER_ID = os.getenv("GDRIVE_MODEL_FOLDER_ID", "https://drive.google.com/drive/folders/1vOUdphH-1I0Eq4IYCLH90ZJLVsPzuN0W?usp=sharing")

VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
# Keep LR model aligned with tfidf_vectorizer.pkl (both 50,000 features).
LR_MODEL_PATH = MODELS_DIR / "lr_model.pkl"
DISTILBERT_MODEL_HF_PATH = MODELS_DIR / "distilbert_hf"
MENTAL_ROBERTA_HF_PATH = MODELS_DIR / "mental_roberta_hf"
XGBOOST_MODEL_PATH = MODELS_DIR / "xgb_depression_classifier.pkl"
XGBOOST_VECTORIZER_PATH = MODELS_DIR / "xgb_tfidf_vectorizer.pkl"
