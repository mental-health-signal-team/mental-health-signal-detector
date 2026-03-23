#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  # Export variables defined in .env for this script.
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${LOCATION:?Missing LOCATION (example: europe-west1)}"
: "${PROJECT_ID:?Missing PROJECT_ID}"
: "${REPOSITORY:?Missing REPOSITORY}"
: "${IMAGE:?Missing IMAGE}"

PLATFORM="${PLATFORM:-linux/amd64}"
DASHBOARD_IMAGE="${DASHBOARD_IMAGE:-mental-health-dashboard}"
API_SERVICE="${API_SERVICE:-mental-health-app}"
DASHBOARD_SERVICE="${DASHBOARD_SERVICE:-mental-health-dashboard}"

API_IMAGE_URI="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE}"
DASHBOARD_IMAGE_URI="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${DASHBOARD_IMAGE}"

echo "==> Configuring Docker auth for Artifact Registry"
gcloud auth configure-docker "${LOCATION}-docker.pkg.dev" -q

echo "==> Building API image: ${API_IMAGE_URI}"
docker build --platform "${PLATFORM}" -t "${API_IMAGE_URI}" -f docker/Dockerfile.api .

echo "==> Building Dashboard image: ${DASHBOARD_IMAGE_URI}"
docker build --platform "${PLATFORM}" -t "${DASHBOARD_IMAGE_URI}" -f docker/Dockerfile.dashboard .

echo "==> Pushing API image"
docker push "${API_IMAGE_URI}"

echo "==> Pushing Dashboard image"
docker push "${DASHBOARD_IMAGE_URI}"

echo "==> Deploying API service: ${API_SERVICE}"
gcloud run deploy "${API_SERVICE}" \
  --image="${API_IMAGE_URI}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --region="${LOCATION}" \
  --allow-unauthenticated \
  --timeout=600 \
  --memory=2Gi

API_URL="$(gcloud run services describe "${API_SERVICE}" --project="${PROJECT_ID}" --region="${LOCATION}" --format='value(status.url)')"
if [[ -z "${API_URL}" ]]; then
  echo "ERROR: Unable to retrieve API URL after deployment."
  exit 1
fi

echo "==> API URL detected: ${API_URL}"
echo "==> Deploying Dashboard service: ${DASHBOARD_SERVICE}"
gcloud run deploy "${DASHBOARD_SERVICE}" \
  --image="${DASHBOARD_IMAGE_URI}" \
  --project="${PROJECT_ID}" \
  --platform=managed \
  --region="${LOCATION}" \
  --allow-unauthenticated \
  --set-env-vars="API_URL=${API_URL}" \
  --timeout=600 \
  --memory=1Gi

DASHBOARD_URL="$(gcloud run services describe "${DASHBOARD_SERVICE}" --project="${PROJECT_ID}" --region="${LOCATION}" --format='value(status.url)')"

echo ""
echo "Deployment completed successfully."
echo "API URL: ${API_URL}"
echo "Dashboard URL: ${DASHBOARD_URL}"
