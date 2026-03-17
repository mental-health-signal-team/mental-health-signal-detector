# Mental Health Signal Detector

**Artefact School of Data — Bootcamp Data Science, Mars 2026**

Système AI de détection de signaux de détresse mentale via NLP sur textes Reddit, avec une application de check-in matinal « Comment vas-tu ce matin ? » à destination des adolescents et adultes.

---

## Architecture

```
src/
├── api/          FastAPI REST API (/health, /predict, /checkin)
├── checkin/      App check-in "Comment vas-tu ce matin ?" (Gradio)
├── common/       Config, logging, détection/traduction langue FR↔EN
├── dashboard/    Dashboard Streamlit + explainability SHAP
└── training/     Preprocessing, entraînement, évaluation des modèles
```

---

## Modèles

| Modèle | Dataset | Accuracy | F1 Macro | Notes |
|--------|---------|----------|----------|-------|
| Baseline TF-IDF + LR | 388K | 88.9% | — | Référence prod |
| DistilBERT v1 | DAIR-AI 16K | 96.8% | — | Fine-tuning initial |
| **DistilBERT v2** | **388K combiné** | **89.0%** | **86.5%** | **Best epoch 2 / 3** |
| DistilBERT v2.1 *(prêt)* | 388K + CustomTrainer | — | — | EarlyStop + class weights |

DistilBERT v2 — résultats Colab T4 GPU (3 epochs, batch=32) :

| Epoch | Train Loss | Val Loss | Accuracy | F1 Macro |
|-------|-----------|----------|----------|----------|
| 1 | 0.294 | 0.284 | 88.5% | 85.4% |
| **2** | **0.229** | **0.283** | **89.0%** | **86.5%** ✅ |
| 3 | 0.114 | 0.348 | 88.97% | 86.5% (overfit) |

> `load_best_model_at_end=True` → checkpoint epoch 2 conservé. Bat le baseline LR (78.6% sur eRisk25).

---

## Datasets

| Source | Exemples | Type | Label |
|--------|----------|------|-------|
| Kaggle Reddit Depression | 100 000 | Posts Reddit | Communautaire |
| DAIR-AI/emotion | 18 000 | Phrases courtes | 6 émotions → binaire |
| GoEmotions (Google) | 53 000 | Commentaires Reddit | 28 émotions → binaire |
| **eRisk25 (CLEF 2025)** | **217 000** | **Posts Reddit cliniques** | **Dépression validée** |
| **Total** | **~388 000** | | |

---

## App "Comment vas-tu ce matin ?"

Application de check-in matinal ludique pour ados et adultes.

### 4 niveaux de réponse

| Niveau | Déclencheur | Réponse |
|--------|-------------|---------|
| 🔵 CRITIQUE | Mots-clés idéation suicidaire (détectés avant tout scoring) | 3114 + SAMU immédiatement |
| 🔴 ROUGE | Score ≥ 0.65 | Empathie + orientation professionnelle |
| 🟡 JAUNE | 0.35 ≤ score < 0.65 | Question de suivi + tips |
| 🟢 VERT | Score < 0.35 | Encouragement + tip du jour |

### Structure clinique (5 axes)

Inspiré de PHQ-9, GAD-7, MBI (Maslach Burnout Inventory) :

- **AXE 1 — Affect** : tristesse, anxiété, irritabilité, vide
- **AXE 2 — Cognitions** : inutilité, culpabilité, désespoir, catastrophisme
- **AXE 3 — Somatique** : sommeil, fatigue, tension, respiration
- **AXE 4 — Comportement** : retrait social, évitement, perte d'activité
- **AXE 5 — Risque** : idéation suicidaire (CRITIQUE)

### Sécurité

- Détection CRITIQUE **avant** tout scoring ML (indépendant du modèle NLP)
- Règle de sécurité emoji : le choix de l'utilisateur ne peut pas être contredit par le NLP
- Boost d'intensité +0.15 si modificateurs fréquence détectés ("tout le temps", "depuis des semaines"...)
- Ressources : 3114 (24h/7j), Mon Soutien Psy (12 séances remboursées), Fil Santé Jeunes

### Support multilingue

- Détection automatique FR/EN via `langdetect` (seed fixé pour déterminisme)
- Traduction FR→EN via `deep-translator` avant analyse NLP
- Seed fixé : `DetectorFactory.seed = 0`

---

## Installation

```bash
# Cloner le repo
git clone git@github.com:stanislav-grinchenko/mental-health-signal-detector.git
cd mental-health-signal-detector
git checkout Fabrice

# Environnement
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Variables d'environnement
cp .env.example .env
```

---

## Lancement

```bash
# API FastAPI
bash scripts/run_api.sh          # → http://localhost:8000

# Dashboard Streamlit
bash scripts/run_dashboard.sh    # → http://localhost:8501

# App check-in Gradio
python -m src.checkin.app        # → http://localhost:7860
```

---

## Entraînement

```bash
# Baseline avec toutes les sources
python -m src.training.train \
  --model baseline \
  --kaggle-path data/raw/reddit_depression_dataset.csv \
  --go-emotions \
  --erisk25-path data/raw/erisk25/ \
  --kaggle-samples 100000
```

**DistilBERT v2/v2.1 → Google Colab (GPU T4 requis)**

```
notebooks/distilbert_finetune_colab.ipynb
```

Améliorations v2.1 intégrées dans le notebook :
- `CustomTrainer` avec `CrossEntropyLoss` pondérée (class imbalance 34/66)
- `EarlyStoppingCallback(patience=1)` sur `f1_macro`
- `num_train_epochs=5` + `greater_is_better=True`
- Poids calculés depuis les données réelles via `sklearn.compute_class_weight`

---

## Sécurité appliquée

### Revue 1 — P0/P1 (2026-03-17)

| Niveau | Correction | Fichier |
|--------|-----------|---------|
| P0 | Texte patient hashé avant log (anti-PHI) | `src/checkin/engine.py` |
| P0 | Allowlist `model_type` (path traversal) | `src/api/dependencies.py` |
| P0 | `joblib.load` restreint au dossier `models/` | `src/training/predict.py` |
| P1 | Emoji validé par pattern Pydantic (5 valeurs) | `src/checkin/schemas.py` |
| P1 | Texte : `min_length=1`, `max_length=1000` | `src/checkin/schemas.py` |
| P1 | Middleware 64 KB — octets réels lus (anti-DoS/chunked) | `src/api/main.py` |
| P1 | `/health` retourne 503 sur erreur inattendue | `src/api/main.py` |
| P1 | `API_URL` validé regex + IPs privées bloquées en prod | `src/checkin/app.py`, `src/dashboard/app.py` |
| P1 | Rate limiting slowapi : 20/min checkin, 30/min predict, 10/min explain | `src/api/rate_limit.py` |
| P1 | CORS restreint via `ALLOWED_ORIGINS` env en production | `src/api/main.py` |

### Revue 2 — findings (2026-03-17)

| Sévérité | Correction | Fichier |
|----------|-----------|---------|
| High | `GET /checkin/reminders` supprimé (fuite inter-utilisateurs, RGPD art. 9) | `src/api/checkin_router.py` |
| Medium | Middleware taille : lecture octets réels (couvre chunked transfer) | `src/api/main.py` |
| Medium | PYSEC-2022-252 deep-translator documenté (`.pip-audit-ignore`) | `requirements.txt` |
| Low | SSRF : blocage RFC-1918 + loopback + link-local en production | `src/checkin/app.py` |

### Posture actuelle — `ruff` ✅ · `pip-audit` ✅ (1 exception documentée) · 76/76 tests ✅

---

## Tests & CI

```bash
pytest tests/ -q --cov=src
ruff check src/
```

CI GitHub Actions sur push → branche Fabrice ✅

---

## Docker

```bash
cd docker/
docker-compose up --build
```

---

## Ressources d'aide intégrées

| Ressource | Contact | Disponibilité |
|-----------|---------|---------------|
| 3114 — Prévention suicide | Appel gratuit | 24h/24, 7j/7 |
| Mon Soutien Psy | monsoutienpsy.ameli.fr | 12 séances/an remboursées |
| Fil Santé Jeunes (12-25 ans) | 0 800 235 236 | 9h–23h |
| SAMU | 15 | Urgences 24h/24 |

---

## Contexte

Ce projet s'inscrit dans la **Grande Cause Nationale 2025-2026 « Parlons santé mentale ! »**.

Selon le baromètre Santé publique France 2024 : 1 adulte sur 6 a vécu un épisode dépressif, et 1 sur 2 n'a pas consulté de professionnel spécialisé. Les principaux freins : coût, stigmatisation, manque d'information sur les ressources disponibles.

Sources : Assurance Maladie, témoignages MSP (Jade, Sarah, Bruno, Enzo), Journal de bien-être émotionnel pour ados, CLEF eRisk 2025.
