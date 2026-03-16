#!/bin/bash
set -e
cd "$(dirname "$0")/.."
echo "Démarrage du dashboard Streamlit sur http://localhost:8501"
streamlit run src/dashboard/app.py --server.port 8501
