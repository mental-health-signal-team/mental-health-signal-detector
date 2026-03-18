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

| Modèle | Dataset | Accuracy | F1 Macro | Sensitivité | Notes |
|--------|---------|----------|----------|-------------|-------|
| Baseline TF-IDF + LR | 388K | 88.9% | — | — | Référence prod slim |
| DistilBERT v1 | DAIR-AI 16K | 96.8% | — | — | Fine-tuning initial |
| **DistilBERT v2** | **388K combiné** | **89.0%** | **86.5%** | — | **Champion prod GPU** |
| DistilBERT v2.1 *(évalué)* | 388K + EarlyStop | — | 86.06% | — | eval_loss ↑ — rejeté |
| **Mental-BERT v3** ✨ | **Kaggle 100K + eRisk25 (clinique)** | **92.7%** | **92.5%** | **95.9%** | **AUC-ROC 98.2% · model_type=mental_bert_v3** |

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

Panneau "Analyse" sur l'écran Solutions : points de triage, barre de score %, dimensions cliniques détectées (Épuisement / Anxiété / Humeur dépressive / Dysrégulation), tendance longitudinale (↓ amélioration / → stable / ↑ dégradation) vs session précédente. Masqué en mode enfant et niveau 4.

### Self-report clinique (QuickCheck)

3 micro-questions adaptatives (PHQ-9 / GAD-7 / PSS inspirées) selon l'émotion. Score DSM-5 pondéré : durée×1.5 + impact×1.5 > intensité×1.0. Les réponses sont mappées en dimensions cliniques (`detectDimensionsFromSelfReport`) et fusionnées avec l'analyse textuelle — sans perte d'information. Phase "bridge" empathique avant les questions (moins abrupt).

### Suivi longitudinal

`sessionHistory.ts` : historique localStorage 30 jours, 10 sessions max, déduplication 5 min. Tendance calculée sur les 2 dernières sessions, affichée dans le panneau Analyse (adulte) ou en message narratif (enfant).

### Feedback micro-actions

Boutons discrets (👍 / Pas vraiment) dans chaque ActionCard dépliée. Stockage local `mh_action_feedback` (50 entrées) — aucune transmission serveur.

### Ressources avec liens web

Champ `website?: string` optionnel sur les ressources : lien secondaire "Voir le site →" rendu dans ResourceCard. 5 ressources enrichies : Fil Santé Jeunes, 3018, 3020, 3919, Médecin traitant.

### Accessibilité (WCAG 2.1 AA)

- `role="group"` + `aria-label` sur grilles émotions et boutons rapides
- `role="progressbar"` sur barre de progression QuickCheck
- `role="radiogroup"` + `role="radio"` + `aria-label` sur options
- `aria-pressed` sur boutons toggle, `aria-expanded` sur ActionCard
- `aria-live="polite"` sur texte de phase respiration ("Inspirez/Expirez")
- `aria-hidden="true"` sur tous les éléments décoratifs

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

### Revue 3 — Code review sécurité complète (2026-03-18)

| Criticité | Bug/Finding | Correction | Fichier |
|-----------|-------------|-----------|---------|
| High | S1 — `emotion_id`/`distress_level` (RGPD Art.9) persistés sans auth | Non stockés dans le store mémoire | `src/api/checkin_router.py` |
| Medium | B1 — `/checkin/reminder` sans rate limit | `@limiter.limit("10/minute")` ajouté | `src/api/checkin_router.py` |
| Medium | B2 — Client Anthropic créé à chaque requête | Singleton `_get_anthropic_client()` | `src/api/analyze_router.py` |
| Medium | B3/S3 — Rate limit inefficace derrière proxy | `_get_client_ip()` lit `X-Forwarded-For` | `src/api/rate_limit.py` |
| Medium | S2 — CORS wildcard silencieux en prod mal configuré | Guard `env != production` + warning explicite | `src/api/main.py` |
| Low | B4 — `_MODELS_DIR` dépend du CWD | Basé sur `__file__` (CWD-indépendant) | `src/training/predict.py` |
| Low | B5 — `run_explain` crash hors vocabulaire | Guard `nonzero_idx.size == 0` | `src/api/services.py` |
| Low | S4 — Injection prompt via évolution modèle | `_build_user_prompt` reçoit scalaires explicites | `src/api/analyze_router.py` |
| Low | S6 — Fuite chemin interne dans logs | `logger.exception()` remplace `logger.error(f"...{e}")` | `src/api/main.py` |

### Posture actuelle — `ruff` ✅ · `pip-audit` ✅ (1 exception documentée) · 117/117 tests ✅

---

## Tests & CI

```bash
# Backend Python
pytest tests/ -q --cov=src      # 117 tests : API, sécurité, moteur, entraînement

# Frontend TypeScript
cd frontend && npm run test      # 180 tests Vitest (scoringEngine, solutionEngine)
npm run test:e2e                 # 18 tests Playwright (happy path, crisis flow)

# Linting
ruff check src/
```

**Total : 180 Vitest + 18 Playwright + 117 pytest = 315 tests ✅**

CI GitHub Actions sur push → branche Fabrice ✅

---

## Docker

```bash
# Stack complète (API + Frontend + Dashboard)
cd docker/
docker-compose up --build

# API seule avec Mental-BERT v3 (modèles montés en volume)
docker build -f docker/Dockerfile.api -t mh-api-full:v3 .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  mh-api-full:v3
```

**Images disponibles :**

| Image | Taille | Stack | Modèles |
|-------|--------|-------|---------|
| `Dockerfile.api.slim` | ~600 MB | baseline TF-IDF+LR | `baseline.joblib` baked in |
| `Dockerfile.api` (`mh-api-full:v3`) | ~3.2 GB | PyTorch CPU-only + transformers | Volume mount requis |
| `Dockerfile.frontend` | ~50 MB | nginx multi-stage | — |

> **Note :** `models/fine_tuned/` et `models/fine_tuned_v3/` sont exclus du build context (`.dockerignore`) — montez-les avec `-v $(pwd)/models:/app/models`.

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
