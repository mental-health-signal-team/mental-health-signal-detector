# Mental Health Signal Detector

Système de détection de signaux de détresse mentale dans des textes via NLP — projet final Artefact School of Data.

L'objectif est d'identifier automatiquement des posts Reddit exprimant une détresse psychologique (dépression, anxiété, désespoir) en combinant plusieurs modèles de machine learning, une API REST sécurisée et un dashboard interactif.

---

## Table des matières

- [Architecture](#architecture)
- [Datasets](#datasets)
- [Modèles](#modèles)
- [Performances](#performances)
- [Installation](#installation)
- [Entraînement](#entraînement)
- [API](#api)
- [Dashboard](#dashboard)
- [Docker](#docker)
- [Tests & CI](#tests--ci)
- [Sécurité](#sécurité)
- [Structure du projet](#structure-du-projet)

---

## Architecture

```
Texte (FR ou EN)
    │
    ▼
Détection langue (langdetect)
    │
    ▼ si FR
Traduction FR→EN (deep-translator, timeout 5s)
    │
    ▼
Nettoyage texte (clean_text)
    │
    ├──► Baseline : TF-IDF + Logistic Regression  ──► score_distress [0–1]
    │         └──► Explication SHAP (contributions par mot)
    │
    └──► Avancé   : DistilBERT fine-tuned          ──► score_distress [0–1]
```

| Couche | Technologie |
|--------|-------------|
| ML / NLP | scikit-learn, HuggingFace Transformers |
| Sérialisation modèle | joblib (format compact, 2× plus rapide que pickle — pour artefacts ML internes de confiance) |
| API REST | FastAPI + Uvicorn + CORS middleware |
| Check-in conversationnel | FastAPI router + moteur emoji+NLP (`src/checkin/`) |
| Dashboard | Streamlit + visualisation SHAP |
| Traduction | deep-translator + timeout via `concurrent.futures` (compatible ASGI) |
| Infrastructure | Docker (non-root) + GitHub Actions CI |

---

## Datasets

Trois sources publiques combinées — **170 000 exemples** après nettoyage :

| Dataset | Source | Taille | Label détresse |
|---------|--------|--------|---------------|
| Reddit Depression | [Kaggle](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch) | 2,47M posts → 100K sous-échantillonnés | `label=1` (dépression/suicide) |
| DAIR-AI/emotion | [HuggingFace](https://huggingface.co/datasets/dair-ai/emotion) | 18K phrases | sadness + fear → 1 |
| GoEmotions (Google) | [HuggingFace](https://huggingface.co/datasets/google-research-datasets/go_emotions) | 54K commentaires Reddit | sadness, grief, fear, nervousness, remorse, embarrassment, disappointment → 1 |

**Distribution finale :** 69% non-détresse / 31% détresse

### Téléchargement Kaggle

```bash
# Nécessite kaggle.json dans ~/.kaggle/
kaggle datasets download -d nikhileswarkomati/suicide-watch -p data/raw/ --unzip
```

DAIR-AI et GoEmotions sont téléchargés automatiquement depuis HuggingFace lors de l'entraînement.

---

## Modèles

### Baseline — TF-IDF + Logistic Regression

Rapide (~2 min CPU), interprétable via SHAP, utilisé par défaut dans l'API.

- TF-IDF : 50 000 features, unigrammes + bigrammes, `sublinear_tf=True`
- Logistic Regression : `C=1.0`, `class_weight="balanced"`, `max_iter=1000`
- Sérialisé via **joblib** dans `models/baseline.joblib`

### Avancé — DistilBERT fine-tuned

Fine-tuning de `distilbert-base-uncased` sur HuggingFace Trainer.

- 3 epochs, batch 16, `max_length=128`
- Entraîné sur Colab T4 GPU (~8 min)
- Support FR/EN via la même pipeline de traduction que le baseline
- Sauvegardé dans `models/fine_tuned/` (format HuggingFace safetensors)

> Le fine-tuning CPU est estimé à ~36h — utiliser `notebooks/distilbert_finetune_colab.ipynb`

---

## Performances

| Modèle | Dataset | Accuracy | F1 weighted | F1 macro |
|--------|---------|----------|-------------|----------|
| Baseline TF-IDF+LR | Kaggle + DAIR-AI + GoEmotions (170K) | **90.9%** | **91.0%** | **89.4%** |
| DistilBERT fine-tuned | DAIR-AI (16K) | **96.8%** | — | — |

**Détail baseline par classe (34K exemples test) :**

```
              precision    recall  f1-score
non-détresse       0.94      0.93      0.93
    détresse       0.84      0.87      0.85
```

---

## Installation

**Prérequis :** Python 3.11+

```bash
git clone https://github.com/stanislav-grinchenko/mental-health-signal-detector.git
cd mental-health-signal-detector

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt

cp .env.example .env             # puis éditer .env
```

---

## Entraînement

### Baseline (recommandé — CPU, ~2 min)

```bash
# DAIR-AI uniquement (rapide)
python -m src.training.train --model baseline

# Kaggle + DAIR-AI + GoEmotions (170K exemples — recommandé)
python -m src.training.train --model baseline \
    --kaggle-path data/raw/reddit_depression_dataset.csv \
    --go-emotions \
    --kaggle-samples 100000
```

Le modèle est sauvegardé dans `models/baseline.joblib`.

### DistilBERT (GPU — Google Colab T4 recommandé)

Ouvrir `notebooks/distilbert_finetune_colab.ipynb` sur Colab avec runtime T4 GPU.
Le modèle est exporté dans `models/fine_tuned/`.

### Options CLI complètes

| Option | Description | Défaut |
|--------|-------------|--------|
| `--model` | `baseline` ou `distilbert` | `baseline` |
| `--kaggle-path` | Chemin vers le CSV Kaggle | — |
| `--smhd-path` | Chemin vers le CSV SMHD | — |
| `--no-dair` | Exclure DAIR-AI/emotion | — |
| `--go-emotions` | Inclure GoEmotions | — |
| `--kaggle-samples` | Nb max de posts Kaggle | `100000` |

---

## API

### Démarrage

```bash
bash scripts/run_api.sh
# ou
TRANSFORMERS_NO_TF=1 uvicorn src.api.main:app --reload --port 8000
```

### Endpoints

#### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

> L'API démarre même si `models/baseline.joblib` est absent (CI, fresh-clone) — les endpoints `/predict` et `/explain` retournent alors un `503`.

---

#### `POST /predict`

Prédit le niveau de détresse d'un texte (FR ou EN).

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Je me sens tellement triste", "model_type": "baseline"}'
```

**Corps de la requête :**

| Champ | Type | Valeurs | Description |
|-------|------|---------|-------------|
| `text` | string | 1–5000 caractères | Texte à analyser (FR ou EN) |
| `model_type` | string | `baseline` \| `distilbert` | Modèle à utiliser |

**Réponse :**

```json
{
  "label": 1,
  "score_distress": 0.923,
  "model": "baseline",
  "text_preview": "Je me sens tellement triste",
  "detected_lang": "fr"
}
```

| Champ | Description |
|-------|-------------|
| `label` | `0` = non-détresse, `1` = détresse |
| `score_distress` | Probabilité de détresse [0.0–1.0] |
| `detected_lang` | Langue détectée (`fr`, `en`, …) |

---

#### `POST /explain`

Retourne les mots les plus influents avec leur contribution au score (baseline uniquement).

```bash
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel hopeless and empty", "n_features": 10}'
```

**Corps de la requête :**

| Champ | Type | Défaut | Description |
|-------|------|--------|-------------|
| `text` | string | — | Texte à analyser (FR ou EN) |
| `n_features` | int | `15` | Nombre de mots à retourner (3–30) |

**Réponse :**

```json
{
  "label": 1,
  "score_distress": 0.983,
  "detected_lang": "en",
  "features": [
    {"word": "hopeless", "shap_value": 1.13},
    {"word": "empty",    "shap_value": 0.87},
    {"word": "great",    "shap_value": -0.40}
  ]
}
```

> `shap_value` est une contribution linéaire `tfidf(w) × coef_LR(w)` — approximation des valeurs de Shapley valide pour les modèles linéaires. Pour du SHAP strict, `evaluate.py::explain_with_shap()` utilise `shap.LinearExplainer`.

> Valeur positive → pousse vers détresse · Valeur négative → pousse vers non-détresse

---

#### `POST /checkin`

Point d'entrée de l'app "Comment vas-tu ce matin ?" (Phase 2). Combine un emoji de humeur et/ou un texte libre pour produire une réponse conversationnelle contextualisée.

```bash
curl -X POST http://localhost:8000/checkin \
  -H "Content-Type: application/json" \
  -d '{"emoji": "😔", "text": "Je me sens fatigué depuis quelques jours", "step": 1}'
```

**Corps de la requête :**

| Champ | Type | Valeurs | Description |
|-------|------|---------|-------------|
| `emoji` | string \| null | `😄 🙂 😐 😔 😢` | Emoji de humeur (optionnel si texte fourni) |
| `text` | string \| null | max 1000 caractères | Texte libre (optionnel si emoji fourni) |
| `step` | int | `1` \| `2` | `1` = check-in initial, `2` = réponse à la question de suivi |

> Au moins un `emoji` ou un `text` est requis — sinon `422`.

**Réponse :**

```json
{
  "level": "yellow",
  "score": 0.65,
  "greeting": "Bonjour ☀️",
  "message": "Je t'entends. C'est courageux de le reconnaître. 💛",
  "tip": "💡 Rappel : Parfois, mettre des mots sur ce qu'on ressent suffit à l'alléger.",
  "follow_up": "Depuis combien de temps tu te sens comme ça ?",
  "resources": [],
  "detected_lang": "fr"
}
```

| Champ | Description |
|-------|-------------|
| `level` | `green` / `yellow` / `red` — déterminé par max(score emoji, score NLP) |
| `score` | Score de détresse fusionné [0.0–1.0] |
| `follow_up` | Question de suivi (step=1 uniquement, null sinon) |
| `resources` | Ressources d'aide (vides en vert, Mon Soutien Psy en jaune step=2, 3114 en rouge) |

**Plancher de sécurité :**

| Emoji | Niveau minimum garanti |
|-------|----------------------|
| 😢 | `red` — toujours, quelle que soit l'analyse NLP |
| 😔 | `yellow` — jamais en dessous, même si le NLP est optimiste |

---

**Codes d'erreur :**

| Code | Cause |
|------|-------|
| `503` | Modèle non trouvé — lancer l'entraînement d'abord |
| `500` | Erreur interne — voir les logs serveur |

---

## Dashboard

```bash
bash scripts/run_dashboard.sh
# ou
streamlit run src/dashboard/app.py
```

Ouvrir [http://localhost:8501](http://localhost:8501).

**Fonctionnalités :**
- Saisie de texte libre (FR ou EN — traduction automatique)
- Choix du modèle (baseline / DistilBERT)
- Score de risque, label et langue détectée
- Graphique SHAP horizontal (baseline) : barres rouges = mots de détresse, bleues = mots positifs

---

## Docker

### Prérequis

Le fichier `models/baseline.joblib` doit exister avant de builder l'image API.

### Démarrer les deux services

```bash
cd docker/
docker-compose up --build
```

- API : [http://localhost:8000](http://localhost:8000)
- Dashboard : [http://localhost:8501](http://localhost:8501)

### Services

| Service | Dockerfile | Port | Utilisateur |
|---------|-----------|------|-------------|
| `api` | `Dockerfile.api` | 8000 | `appuser` (uid=1000) |
| `dashboard` | `Dockerfile.dashboard` | 8501 | `appuser` (uid=1000) |

Le service dashboard attend que l'API soit healthy avant de démarrer (`depends_on: condition: service_healthy`).

---

## Tests & CI

### Lancer les tests localement

```bash
pip install -r requirements-dev.txt

ruff check src/ tests/          # lint
pytest tests/ --cov=src --cov-report=term-missing
```

**Couverture des tests :**

| Fichier | Ce qui est couvert |
|---------|--------------------|
| `tests/training/test_train.py` | Preprocessing, entraînement baseline, scoring |
| `tests/api/test_health.py` | `/health`, `/predict` (modèle absent, type invalide) |
| `tests/api/test_checkin.py` | `/checkin` : 422, emoji seul, fallback NLP, planchers de sécurité, step=2 |
| `tests/checkin/test_engine.py` | `compute_score`, `get_level`, planchers 😢→RED et 😔→YELLOW, `build_response` |

### CI GitHub Actions

Déclenché sur push vers `main`, `Fabrice`, `Stan`, `Thomas`, `aimen`.

```
ruff → pytest --cov → upload coverage artifact
```

---

## Sécurité

| Mesure | Détail |
|--------|--------|
| **Sérialisation** | joblib pour les artefacts ML internes — modèles chargés uniquement depuis `models/` (non exposé au réseau) |
| **CORS** | `*` en développement, origines restreintes au dashboard en production |
| **Erreurs API** | Messages génériques côté client — détails loggés uniquement côté serveur |
| **Données sensibles** | Le contenu des textes n'est jamais loggué (longueur uniquement) |
| **Traduction externe** | Timeout 5s via `concurrent.futures` (compatible threadpool ASGI) + fallback silencieux |
| **Validation URL** | `API_URL` dans le dashboard normalisée puis validée par regex (protection SSRF, slash final toléré) |
| **Docker** | Conteneurs exécutés en utilisateur non-root (`appuser`, uid=1000) |
| **Secrets** | `.env` dans `.gitignore` — jamais commité |

> Ce système est un détecteur de signal NLP. Il ne constitue pas un diagnostic clinique.

---

## Variables d'environnement

Copier `.env.example` en `.env` :

| Variable | Description | Défaut |
|----------|-------------|--------|
| `ENV` | `development` \| `production` | `development` |
| `LOG_LEVEL` | Niveau de log | `INFO` |
| `MODEL_NAME` | Modèle HuggingFace de base | `distilbert-base-uncased` |
| `MODEL_PATH` | Chemin DistilBERT fine-tuned | `./models/fine_tuned` |
| `DATABASE_URL` | URL base de données | `sqlite:///./mhdetector.db` |
| `REDDIT_CLIENT_ID` | ID app Reddit (PRAW) | — |
| `REDDIT_CLIENT_SECRET` | Secret app Reddit | — |
| `REDDIT_USER_AGENT` | User-agent PRAW | — |

---

## Structure du projet

```
mental-health-signal-detector/
├── configs/
│   ├── api.yaml              # Host, port, CORS, timeout traduction
│   ├── dashboard.yaml        # API URL, timeout, nb features SHAP
│   └── train.yaml            # Sources données, hyperparamètres
├── data/
│   └── raw/                  # Datasets bruts (non versionnés)
├── docker/
│   ├── Dockerfile.api        # Python 3.11-slim, non-root, curl healthcheck
│   ├── Dockerfile.dashboard  # Python 3.11-slim, non-root
│   └── docker-compose.yml    # Services api + dashboard, healthcheck
├── models/
│   ├── baseline.joblib       # Modèle TF-IDF + LR (joblib, compressé)
│   └── fine_tuned/           # DistilBERT fine-tuned (safetensors)
├── notebooks/
│   └── distilbert_finetune_colab.ipynb
├── reports/
│   └── confusion_matrix_baseline.png
├── scripts/
│   ├── run_api.sh
│   ├── run_dashboard.sh
│   └── run_train.sh
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI, lifespan gracieux, CORS, endpoints
│   │   ├── schemas.py        # Pydantic : PredictRequest/Response, ExplainRequest/Response
│   │   ├── services.py       # run_prediction(), run_explain() (contributions TF-IDF×coef LR)
│   │   ├── checkin_router.py # Router /checkin : fusion emoji+NLP, fallback gracieux
│   │   └── dependencies.py   # get_model() avec @lru_cache(maxsize=2) — baseline + distilbert
│   ├── checkin/
│   │   ├── engine.py         # Moteur de décision : compute_score, get_level, planchers sécurité
│   │   ├── content.py        # Réponses/tips/ressources par niveau (VERT/JAUNE/ROUGE)
│   │   └── schemas.py        # CheckInRequest / CheckInResponse (Pydantic)
│   ├── common/
│   │   ├── config.py         # Settings pydantic-settings
│   │   ├── language.py       # Détection langue + traduction FR→EN (timeout concurrent.futures)
│   │   └── logging.py        # Loguru : stderr + rotation fichier (crée logs/ si absent)
│   ├── dashboard/
│   │   └── app.py            # Streamlit : prédiction + graphique SHAP + protection SSRF
│   └── training/
│       ├── preprocess.py     # Chargement et nettoyage (Kaggle, DAIR-AI, GoEmotions, SMHD)
│       ├── train.py          # Entraînement baseline (joblib) et DistilBERT
│       ├── evaluate.py       # Métriques, matrice de confusion, SHAP summary (LinearExplainer)
│       └── predict.py        # Inférence baseline + DistilBERT (torch import lazy pour baseline)
├── tests/
│   ├── api/
│   │   ├── test_health.py    # /health, /predict (modèle absent, type invalide)
│   │   └── test_checkin.py   # /checkin : 422, emoji seul, fallback NLP, planchers sécurité
│   ├── checkin/
│   │   └── test_engine.py    # compute_score, get_level, planchers 😢→RED 😔→YELLOW
│   └── training/
│       └── test_train.py     # Preprocessing, entraînement, scoring
├── .env.example              # Template variables d'environnement (commenté)
├── .gitignore                # .env, data/, models/ exclus
├── .github/workflows/ci.yml  # ruff + pytest + coverage
├── pytest.ini
├── requirements.txt
└── requirements-dev.txt
```

---

## Auteurs

Projet réalisé dans le cadre du bootcamp **Data Science** à l'[Artefact School of Data](https://www.artefact.com/data-consulting-transformation/artefact-school-of-data/).

- Fabrice Moncaut
- Stanislas Grinchenko
- Thomas
- Aimen
