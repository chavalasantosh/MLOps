#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export DIABETES_IMAGE="${DIABETES_IMAGE:-diabetes-bentoml:v1}"
export DIABETES_PORT="${DIABETES_PORT:-3001}"
export COMPOSE_PROJECT_NAME="${COMPOSE_PROJECT_NAME:-diabetes-ci}"
cleanup() {
  status=$?
  trap - EXIT
  if (( status != 0 )); then docker compose logs --tail 100 || true; fi
  docker compose down --remove-orphans || true
  exit "$status"
}
trap cleanup EXIT
python3 verify_artifact.py
docker compose config -q
docker compose up -d --no-build --wait --wait-timeout 180
docker compose ps
API_BASE_URL="http://127.0.0.1:${DIABETES_PORT}" python3 test_api.py
