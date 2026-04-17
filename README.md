# Mental Health Signal Detector

![CI](https://github.com/stanislav-grinchenko/mental-health-signal/actions/workflows/ci.yml/badge.svg)

An end-to-end NLP system that detects early psychological distress signals in text using fine-tuned transformer models. Built as the final certification project for the **Développeur en Intelligence Artificielle** diploma (RNCP37827BC02 — Bloc 2) at Artefact School of Data, Paris.

> **Disclaimer**: This system is NOT a clinical diagnostic tool. It is an early-warning smoke detector — designed to flag risk before situations escalate.

---

## What It Does

- Classifies text as **distress** (1) or **no distress** (0) with a confidence score
- Returns a three-tier **risk level**: low / medium / high
- Provides **word-level explanations** via gradient × input attribution
- Logs predictions (anonymised by SHA-256) and exposes monitoring stats
- Tracks **confidence drift** to detect model degradation over time

---

## Architecture

```
Reddit posts (PRAW)
        |
        v
  Preprocessing (NLTK + custom tokenisation)
        |
        v
  ML Models (LR / XGBoost / DistilBERT / MentalRoBERTa)
        |
        v
  FastAPI REST API  ──────────────────────────────>  PostgreSQL (Neon)
        |                                                    |
        v                                                    v
  Streamlit Dashboard  <──────  GET /stats  ──────  prediction_logs table
```

**Deployed:**
- API: [GCP Cloud Run](https://mental-health-signal-api-*.a.run.app) (europe-west1, 4Gi RAM, 2 CPU)
- Dashboard: [Streamlit Community Cloud](https://mental-health-signal.streamlit.app)

---

## Models

| Model | Type | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression + TF-IDF | Classical ML | 0.93 | 0.93 | 0.92 |
| XGBoost + TF-IDF | Gradient Boosting | 0.93 | 0.93 | 0.92 |
| DistilBERT (fine-tuned) | Transformer | 0.96 | 0.94 | 0.97 |
| **MentalRoBERTa (fine-tuned)** | **Domain Transformer** | **0.96** | **0.96** | **0.97** |

**Primary deployment model**: MentalRoBERTa — pretrained on 13.8M mental health posts (Reddit), then fine-tuned on our dataset. Higher precision than DistilBERT reduces false positives, which is critical for a health-adjacent application.

**Training dataset**: `balanced_dataset_30k.csv` — 30,000 Reddit posts, 1:1 class balance, Reddit-only sources (3 iteration dataset rebuild to eliminate spurious correlations — see [docs/technology_watch.md](docs/technology_watch.md)).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | API info and endpoint map |
| GET | `/health` | Health check |
| POST | `/predict` | Predict distress label + probability |
| POST | `/explain` | Predict + word-level explanation (gradient attribution) |
| GET | `/stats` | Aggregated prediction statistics |
| GET | `/stats/drift` | Confidence drift report (7-day vs all-time baseline) |

Full interactive docs available at `/docs` (Swagger UI) and `/redoc`.

### Example: POST /predict

```bash
curl -X POST https://<api-url>/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel hopeless and cannot sleep.", "model_type": "mental_roberta"}'
```

```json
{
  "label": 1,
  "probability": 0.94
}
```

### Example: GET /stats/drift

```json
{
  "baseline_confidence": 0.71,
  "recent_confidence": 0.69,
  "confidence_delta": -0.02,
  "drift_detected": false,
  "drift_threshold": 0.05,
  "baseline_distress_rate": 0.43,
  "recent_distress_rate": 0.41,
  "recent_predictions_count": 87,
  "model_confidence_7d": {"mental_roberta": 0.69}
}
```

---

## Local Setup

**Requirements**: Python 3.11+, [Poetry](https://python-poetry.org/)

```bash
# 1. Clone and install
git clone <repo-url>
cd mental-health-signal-detector
poetry install

# 2. Configure environment
cp .env.example .env
# Edit .env: set DATABASE_URL (defaults to SQLite for local dev)

# 3. Run the API
poetry run uvicorn src.api.main:app --reload

# 4. Run the dashboard (separate terminal)
poetry run streamlit run src/dashboard/app.py
```

### Model Artifacts

Model files are not committed to Git (too large). Download them via:

```bash
poetry run python scripts/download_models.py
```

Or set `GCS_BUCKET` in `.env` and the API will load models from GCS on startup.

---

## Tests

```bash
# Run all tests with coverage report
poetry run pytest

# Run only fast tests (no model artifacts needed)
poetry run pytest tests/api/test_health.py tests/training/
```

The conftest automatically skips model-dependent tests if artifacts are not present.

---

## CI/CD

| Stage | Tool | Trigger |
|---|---|---|
| Lint | ruff (check + format) | push to any branch |
| Pre-commit hooks | pre-commit | push to any branch |
| Tests + Coverage | pytest + pytest-cov | push to any branch |
| Docker build + deploy | GCP Cloud Build | push to `main` |

CD pipeline ([cloudbuild.yaml](cloudbuild.yaml)):
1. Run pytest (test gate)
2. Build Docker image → push to Artifact Registry
3. Deploy to Cloud Run (zero-downtime revision update)

See [docs/mlops_runbook.md](docs/mlops_runbook.md) for rollback and model versioning procedures.

---

## Project Structure

```
mental-health-signal-detector/
├── src/
│   ├── api/          # FastAPI app (main.py, services.py, database.py, schemas.py)
│   ├── dashboard/    # Streamlit app (app.py, pages.py, stats.py)
│   ├── training/     # Model training (train.py, preprocess.py, evaluate.py)
│   ├── data_cleaning/ # Data pipeline (data.py)
│   └── common/       # Shared config, logging, utils
├── tests/            # pytest test suite
├── scripts/          # Utility scripts (model conversion, doc generation)
├── docker/           # Dockerfiles
├── notebooks/        # Exploration and benchmark notebooks
├── docs/             # Project documentation
│   ├── technology_watch.md   # C6 — Veille technologique
│   └── mlops_runbook.md      # C13 — MLOps runbook
└── .github/workflows/ci.yml  # CI pipeline
```

---

## Team

| Member | Role |
|---|---|
| Stanislav Grinchenko | Architecture, FastAPI, CI/CD, coordination |
| Aïmen | ML/NLP ownership |
| Fabrice | Data engineering |
| Thomas | Streamlit dashboard, documentation |
