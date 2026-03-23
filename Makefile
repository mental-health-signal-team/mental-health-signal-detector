SHELL := /bin/bash

PYTHON ?= python3
PYTEST ?= $(PYTHON) -m pytest
PLATFORM ?= linux/amd64
DASHBOARD_IMAGE ?= mental-health-dashboard
API_SERVICE ?= mental-health-app
DASHBOARD_SERVICE ?= mental-health-dashboard

.PHONY: help test test-data-cleaning test-api test-dashboard test-training train-xgboost docker-build-api docker-build-dashboard cloud-run-deploy cloud-run-deploy-dashboard docker-run-local deploy-all

help:
	@echo "Available targets:"
	@echo "  make test                    Run all tests"
	@echo "  make test-data-cleaning      Run data cleaning tests"
	@echo "  make test-api                Run API tests"
	@echo "  make test-dashboard          Run dashboard tests"
	@echo "  make test-training           Run training tests"
	@echo "  make train-xgboost           Train only the XGBoost model"
	@echo ""
	@echo "Docker targets:"
	@echo "  make docker-build-api        Build API Docker image"
	@echo "  make docker-build-dashboard  Build Dashboard Docker image"
	@echo "  make docker-run-local        Run both services locally with docker-compose"
	@echo "  make deploy-all              Build, push and deploy API + Dashboard"
	@echo ""
	@echo "Cloud Run targets:"
	@echo "  make cloud-run-deploy            Deploy API to Google Cloud Run"
	@echo "  make cloud-run-deploy-dashboard  Deploy Dashboard to Google Cloud Run"

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

train-xgboost:
	$(PYTHON) -m src.training.train_xgboost

docker-build-api:
	docker build --platform $(PLATFORM) -t $(LOCATION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY)/$(IMAGE) -f docker/Dockerfile.api .

docker-build-dashboard:
	docker build --platform $(PLATFORM) -t $(LOCATION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY)/$(DASHBOARD_IMAGE) -f docker/Dockerfile.dashboard .

docker-run-local:
	docker-compose -f docker/docker-compose.yml up --build

cloud-run-deploy:
	gcloud run deploy $(API_SERVICE) \
		--image=$(LOCATION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY)/$(IMAGE) \
		--platform=managed \
		--region=$(LOCATION) \
		--allow-unauthenticated \
		--timeout=600 \
		--memory=2Gi

cloud-run-deploy-dashboard:
	gcloud run deploy $(DASHBOARD_SERVICE) \
		--image=$(LOCATION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY)/$(DASHBOARD_IMAGE) \
		--platform=managed \
		--region=$(LOCATION) \
		--allow-unauthenticated \
		--set-env-vars=API_URL=$(API_CLOUD_RUN_URL) \
		--timeout=600 \
		--memory=1Gi

deploy-all:
	bash scripts/deploy_all.sh
