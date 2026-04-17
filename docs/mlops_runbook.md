# MLOps Runbook — Mental Health Signal Detector
## Competency C13 — CI/CD and MLOps Pipeline

**Project**: RNCP37827BC02 — Bloc 2 — Développeur en Intelligence Artificielle
**Last updated**: April 2026

---

## 1. Architecture Overview

```
GitHub Push to main
        |
        v
  GitHub Actions CI  (.github/workflows/ci.yml)
  ├── ruff lint
  ├── pre-commit hooks
  └── pytest + coverage
        |
        v (on success, same push)
  GitHub Actions CD  (.github/workflows/cd.yml)
  ├── Step 1: Authenticate to GCP (Workload Identity Federation — keyless)
  ├── Step 2: Docker build → tagged :latest + :MODEL_VERSION
  ├── Step 3: Push to Artifact Registry
  ├── Step 4: Deploy new revision to Cloud Run
  └── Step 5: Smoke test GET /health on live URL
        |
        v
  Cloud Run — new revision receives 100% traffic
  (previous revision remains available for rollback)
```

---

## 2. Model Versioning Convention

| Version Tag | Meaning |
|---|---|
| `mental_roberta_v1` | Initial MentalRoBERTa fine-tune (F1=0.96) |
| `mental_roberta_v1.1` | Dataset refresh / retraining without architecture change |
| `mental_roberta_v2` | New architecture or major dataset overhaul |

**How to deploy a new model version:**

1. Train and save the model to `models/mental_roberta_hf/`
2. Update the version tag in `.github/workflows/cd.yml`:
   ```yaml
   env:
     MODEL_VERSION: mental_roberta_v1.1
   ```
3. Push to `main` — GitHub Actions CI runs first, then CD builds, tags, and deploys automatically

The versioned Docker image is retained in Artifact Registry indefinitely, providing a full audit trail of deployed model versions.

---

## 3. Rollback Procedure

### Option A — Cloud Run Revision Rollback (fastest, <2 minutes)

Cloud Run keeps previous revisions. To roll back:

```bash
# List available revisions
gcloud run revisions list --service mental-health-api --region europe-west1

# Send 100% traffic to the previous stable revision
gcloud run services update-traffic mental-health-api \
  --region europe-west1 \
  --to-revisions REVISION_NAME=100
```

This does **not** require a new Docker build — it simply redirects traffic.

### Option B — Redeploy Previous Image Tag

If a revision is no longer available, redeploy from the versioned image in Artifact Registry:

```bash
gcloud run deploy mental-health-api \
  --image europe-west1-docker.pkg.dev/mental-health-signal-detector/mental-health-api/api:mental_roberta_v1 \
  --region europe-west1 \
  --platform managed
```

---

## 4. Model Drift Monitoring

The `GET /stats/drift` endpoint monitors confidence distribution shifts in real time.

### Alerting Thresholds

| Metric | Warning | Critical | Action |
|---|---|---|---|
| `confidence_delta` | \|delta\| > 0.05 | \|delta\| > 0.10 | Investigate dataset shift; retrain |
| `recent_distress_rate` vs baseline | ±10% | ±20% | Check for input distribution change |
| `recent_predictions_count` drop | 50% drop vs 7d avg | 80% drop | Check API health / Cloud Run logs |

### How to Check Drift Manually

```bash
curl https://<api-url>/stats/drift | jq .
```

A `drift_detected: true` response means the 7-day mean confidence has deviated more than 5% from the all-time baseline — this is the trigger to investigate.

---

## 5. Incident Response

### API is down (Cloud Run health check fails)

1. Check Cloud Run logs: `gcloud run services logs read mental-health-api --region europe-west1`
2. Verify the database connection: check `DATABASE_URL` env var is set in Cloud Run service
3. If model files are corrupted: roll back to previous revision (Option A above)

### CI is failing on main

1. Check GitHub Actions logs
2. Common causes: ruff format violation, pre-commit hook failure, test regression
3. Fix on a feature branch — do NOT push directly to main

### Database migration

The `prediction_logs` table schema is managed by SQLAlchemy's `Base.metadata.create_all()`. For schema changes:
1. Add the new column to `PredictionLog` in `src/api/database.py`
2. Write a migration script in `scripts/` using `ALTER TABLE` (SQLite: recreate table; PostgreSQL: direct ALTER)
3. Test locally with `DATABASE_URL=sqlite:///./test.db`

---

## 6. Local Development Setup

```bash
# Start API locally (SQLite)
DATABASE_URL=sqlite:///./predictions.db poetry run uvicorn src.api.main:app --reload

# Run full test suite with coverage
poetry run pytest

# Simulate a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel hopeless today", "model_type": "lr"}'

# Check drift locally (after making some predictions)
curl http://localhost:8000/stats/drift
```

---

## 7. GitHub Actions CD — One-Time GCP Setup (Workload Identity Federation)

The CD pipeline authenticates to GCP using **Workload Identity Federation (WIF)** — no service account key is stored in GitHub secrets. This is the recommended approach per Google's security guidelines.

### Why WIF instead of a JSON key?

Service account JSON keys are long-lived credentials that can be leaked. WIF issues short-lived tokens (1 hour) bound to a specific GitHub repository and branch — they expire automatically and cannot be reused outside GitHub Actions.

### Setup Steps

**Step 1 — Create a Workload Identity Pool**

```bash
gcloud iam workload-identity-pools create "github-pool" \
  --project="mental-health-signal-detector" \
  --location="global" \
  --display-name="GitHub Actions Pool"
```

**Step 2 — Create an OIDC Provider for GitHub**

```bash
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="mental-health-signal-detector" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

**Step 3 — Create a Service Account for CD**

```bash
gcloud iam service-accounts create "github-cd-sa" \
  --project="mental-health-signal-detector" \
  --display-name="GitHub Actions CD Service Account"
```

Grant it the minimum required roles:

```bash
# Push images to Artifact Registry
gcloud projects add-iam-policy-binding mental-health-signal-detector \
  --member="serviceAccount:github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Deploy to Cloud Run
gcloud projects add-iam-policy-binding mental-health-signal-detector \
  --member="serviceAccount:github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com" \
  --role="roles/run.developer"

# Needed to attach the SA to the Cloud Run service
gcloud iam service-accounts add-iam-policy-binding \
  github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com \
  --member="serviceAccount:github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

**Step 4 — Allow GitHub to impersonate the Service Account**

Replace `YOUR_GITHUB_ORG/YOUR_REPO` with the actual repo path (e.g. `mental-health-signal-team/mental-health-signal-detector`):

```bash
# Get the pool's full resource name
POOL=$(gcloud iam workload-identity-pools describe "github-pool" \
  --project="mental-health-signal-detector" \
  --location="global" \
  --format="value(name)")

gcloud iam service-accounts add-iam-policy-binding \
  github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL}/attribute.repository/mental-health-signal-team/mental-health-signal-detector"
```

**Step 5 — Add GitHub Repository Secrets**

Go to GitHub → your repo → Settings → Secrets and variables → Actions → New repository secret.

Add these two secrets:

| Secret name | Value |
|---|---|
| `WIF_PROVIDER` | `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider` |
| `WIF_SERVICE_ACCOUNT` | `github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com` |

To get the project number:
```bash
gcloud projects describe mental-health-signal-detector --format="value(projectNumber)"
```

**Step 6 — Verify**

Push any change to `main`. In GitHub Actions → CD workflow, the "Authenticate to Google Cloud" step should show:

```
Successfully authenticated as github-cd-sa@mental-health-signal-detector.iam.gserviceaccount.com
```

---

## 8. Key Configuration

| Variable | Where set | Default | Purpose |
|---|---|---|---|
| `DATABASE_URL` | Cloud Run env var | `sqlite:///./predictions.db` | DB connection string |
| `MODEL_VERSION` | Cloud Run env var (set by cd.yml) | `mental_roberta_v1` | Deployed model version label |
| `GDRIVE_MODEL_FOLDER_ID` | Cloud Run env var | see config.py | GDrive fallback for model loading |
| `LOG_LEVEL` | Cloud Run env var | `INFO` | Logging verbosity |
