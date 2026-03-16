#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Démarrage de l'API FastAPI sur http://localhost:8000"
echo "Docs Swagger : http://localhost:8000/docs"
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
