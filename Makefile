SHELL := /bin/bash

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest

IMAGE   = europe-west1-docker.pkg.dev/mental-health-signal-detector/mental-health-api/api:latest
SERVICE = mental-health-api
REGION  = europe-west1

.PHONY: help test test-data-cleaning test-api test-dashboard test-training \
        docker-build docker-deploy cloud-deploy

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Tests:"
	@echo "  make test                 Run all tests"
	@echo "  make test-data-cleaning   Run data cleaning tests"
	@echo "  make test-api             Run API tests"
	@echo "  make test-dashboard       Run dashboard tests"
	@echo "  make test-training        Run training tests"
	@echo ""
	@echo "  Deployment:"
	@echo "  make docker-build         Build image locally (native arch — for local testing)"
	@echo "  make docker-deploy        Build for linux/amd64 locally and deploy to Cloud Run"
	@echo "  make cloud-deploy         Build and deploy entirely on GCP (no local Docker needed)"

# ── Deployment ────────────────────────────────────────────────────────────────

# 1. Build image locally using native architecture (arm64 on Apple Silicon).
#    Use this only for local testing — do NOT push this image to the registry.
docker-build:
	docker build -f docker/Dockerfile.api -t $(IMAGE) .

# 2. Build for linux/amd64 (Cloud Run architecture) on your local machine and
#    push + deploy. Handles the arm64 → amd64 cross-compilation automatically.
#    Requires: docker buildx, gcloud auth configure-docker europe-west1-docker.pkg.dev
docker-deploy:
	docker buildx build \
		--platform linux/amd64 \
		-f docker/Dockerfile.api \
		-t $(IMAGE) \
		--push \
		.
	gcloud run deploy $(SERVICE) \
		--image $(IMAGE) \
		--region $(REGION) \
		--platform managed \
		--memory 4Gi \
		--cpu 2 \
		--timeout 300

# 3. Build and deploy entirely on GCP — no local Docker needed.
#    Cloud Build handles the build on linux/amd64 machines and deploys to Cloud Run.
cloud-deploy:
	gcloud builds submit --config cloudbuild.yaml .

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) tests -q

test-data-cleaning:
	$(PYTEST) tests/data_cleaning -q

test-api:
	$(PYTEST) tests/api -q

test-dashboard:
	$(PYTEST) tests/dashboard -q

test-training:
	$(PYTEST) tests/training -q
