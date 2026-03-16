#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export TRANSFORMERS_NO_TF=1
export USE_TF=0

MODEL=${1:-baseline}
KAGGLE_PATH=${2:-data/raw/reddit_depression_dataset.csv}
SAMPLES=${3:-100000}

echo "Entraînement du modèle : $MODEL"
python -m src.training.train \
  --model "$MODEL" \
  --kaggle-path "$KAGGLE_PATH" \
  --kaggle-samples "$SAMPLES"
