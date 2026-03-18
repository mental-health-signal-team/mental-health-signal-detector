# Mental Health Signal Detector

**Artefact School of Data — Bootcamp Data Science, Mars 2026**

Système AI de détection de signaux de détresse mentale via NLP, avec une application web React mobile-first « Comment vas-tu ce matin ? » à destination des adolescents et adultes.

Pipeline clinique en 4 étapes : valorisation du choix d'émotion → analyse ML du texte libre → détection d'aberrations (masking, idéation) → solutions thérapeutiques personnalisées (stepped-care NICE).

---

## Architecture

```
frontend/                      React web app (Vite + TypeScript + Tailwind CSS v4)
├── src/screens/               6 écrans : Welcome → EmotionSelection → SelfReport
│                                         → Expression → Support → Solutions
├── src/lib/solutionEngine.ts  Moteur local stepped-care (triage 0-4, CBT/mindfulness)
└── src/types/diagnostic.ts    Contrat DiagnosticProfile frontend ↔ backend

src/
├── api/          FastAPI REST API (/health, /predict, /solutions)
├── solutions/    Moteur recommandation : triage, micro-actions, ressources, schémas
├── common/       Config, logging, détection/traduction langue FR↔EN
├── dashboard/    Dashboard Streamlit + explainability SHAP
└── training/     Preprocessing, entraînement, évaluation des modèles
```

**Déploiement**

| Service | URL | Stack |
|---------|-----|-------|
| Backend | https://mental-health-signal-detector.onrender.com | Docker slim (FastAPI + baseline ML) |
| Frontend | https://mental-health-signal-detector.vercel.app | Vercel SPA (React + Vite) |

---

## Modèles

| Modèle | Dataset | Accuracy | F1 Macro | Notes |
|--------|---------|----------|----------|-------|
| Baseline TF-IDF + LR | 388K | 88.9% | — | Référence prod |
| DistilBERT v1 | DAIR-AI 16K | 96.8% | — | Fine-tuning initial |
| **DistilBERT v2** | **388K combiné** | **89.0%** | **86.5%** | **Best epoch 2 / 3** |
| DistilBERT v2.1 *(évalué)* | 388K + EarlyStop + class weights | — | 86.06% | eval_loss ↑ (overfit) — v2 reste champion |

DistilBERT v2 — résultats Colab T4 GPU (3 epochs, batch=32) :

| Epoch | Train Loss | Val Loss | Accuracy | F1 Macro |
|-------|-----------|----------|----------|----------|
| 1 | 0.294 | 0.284 | 88.5% | 85.4% |
| **2** | **0.229** | **0.283** | **89.0%** | **86.5%** ✅ |
| 3 | 0.114 | 0.348 | 88.97% | 86.5% (overfit) |

> `load_best_model_at_end=True` → checkpoint epoch 2 conservé. Bat le baseline LR (78.6% sur eRisk25).

**Verdict v2.1 :** eval_loss explose epochs 3-4 (0.430 → 0.638) malgré F1 Macro en hausse → overfitting. Best F1 Macro 86.06% < v2 86.5%. **DistilBERT v2 reste le modèle de production.**

**Prod :** baseline TF-IDF+LR déployé sur Render slim (CPU, 989 KB). DistilBERT v2 réservé aux instances avec GPU.

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

Application web React mobile-first, 6 écrans, modes **enfant** et **adulte**.

### Pipeline clinique en 4 étapes

| Étape | Écran | Mécanisme |
|-------|-------|-----------|
| 1 — Valorisation émotion | EmotionSelection | 8 émotions × plancher de sécurité clinique |
| 2 — Analyse ML du texte | Expression | POST /predict → TF-IDF+LR (prod) / DistilBERT (local) |
| 3 — Détection aberrations | Support | masking (émotion positive + ML score élevé) + keywords critiques |
| 4 — Solutions personnalisées | Solutions | Stepped-care NICE : triage 0-4 → micro-actions CBT/mindfulness |

### Score fusionné

```
finalScore = min(1.0, max(mlScore + maskingBonus, emotionFloor))
```

- **emotionFloor** : plancher par émotion (sadness/fear : 0.35 · anger/stress : 0.25 · joy/calm/pride : 0.0)
- **masking** : émotion positive + mlScore > 0.25 → bonus +0.20
- **keywords critiques** : idéation suicidaire → niveau 4 (URGENCE), indépendamment du ML

### Niveaux de triage (stepped-care NICE)

| Niveau | Score | Réponse |
|--------|-------|---------|
| 0 — Bien-être | < 0.20 | Mindfulness, ancrage |
| 1 — Léger | 0.20–0.35 | CBT activation comportementale |
| 2 — Modéré | 0.35–0.55 | CBT restructuration cognitive |
| 3 — Élevé | 0.55–0.75 | Orientation professionnelle + 3114 |
| 4 — Urgent | ≥ 0.75 ou keywords | 3114 + SAMU — ressources urgentes en tête |

### Indicateur discret (mode adulte, niveaux 1-3)

Panneau "Analyse" sur l'écran Solutions : points de triage, barre de score %, dimensions cliniques détectées (Épuisement / Anxiété / Humeur dépressive / Dysrégulation). Masqué en mode enfant et niveau 4.

### Sécurité applicative

- Détection keywords critiques **avant** tout scoring ML
- Whitelist `VALID_EMOTIONS` et `VALID_EMOTION_COLORS` (anti-injection router state)
- AbortController + isMountedRef (fetch React, no memory leak)
- Moteur local `solutionEngine.ts` : Solutions s'affiche instantanément (pas de flash)
- Appel `POST /solutions` en background → mise à jour silencieuse (fondation LLM futur)

### Support multilingue

- Détection automatique FR/EN via `langdetect` (seed fixé pour déterminisme)
- Traduction FR→EN via `deep-translator` avant analyse NLP

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
| 119 — Allô Enfance en danger | Appel gratuit, confidentiel | 24h/24, 7j/7 — enfants, ados, jeunes majeurs |
| 3018 — Cyberharcèlement | Appel gratuit, anonyme | 7j/7, 9h–23h — réseaux sociaux et internet |
| 3020 — Harcèlement scolaire | Appel gratuit | Lun–ven, 9h–20h — harcèlement à l'école |
| 3919 — Arrêtons les violences | Appel gratuit, anonyme | 24h/24, 7j/7 — femmes victimes de violences |
| Mon Soutien Psy | monsoutienpsy.ameli.fr | 12 séances/an remboursées |
| Fil Santé Jeunes (12-25 ans) | 0 800 235 236 | 9h–23h |
| SAMU | 15 | Urgences 24h/24 |

---

## Contexte

Ce projet s'inscrit dans la **Grande Cause Nationale 2025-2026 « Parlons santé mentale ! »**.

Selon le baromètre Santé publique France 2024 : 1 adulte sur 6 a vécu un épisode dépressif, et 1 sur 2 n'a pas consulté de professionnel spécialisé. Les principaux freins : coût, stigmatisation, manque d'information sur les ressources disponibles.

Sources : Assurance Maladie, témoignages MSP (Jade, Sarah, Bruno, Enzo), Journal de bien-être émotionnel pour ados, CLEF eRisk 2025.
