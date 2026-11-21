#!/usr/bin/env bash
# Trusted lab host only. Accept an immutable GHCR manifest digest, never a mutable tag.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
image="${1:-}"
if [[ ! "$image" =~ ^ghcr\.io/[a-z0-9._-]+/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$ ]]; then
  echo 'Usage: bash scripts/deploy.sh ghcr.io/owner/repo@sha256:<64 hex>'; exit 2
fi
export COMPOSE_PROJECT_NAME=diabetes-mlops
export DIABETES_PORT="${DIABETES_PORT:-3001}"
export DIABETES_IMAGE="$image"
docker compose config -q
# Serialize lab deployments. Do not run other Compose commands concurrently.
state="${HOME}/.local/state/diabetes-mlops"
mkdir -p "$state"
exec 9>"$state/deploy.lock"
flock -n 9 || { echo 'Another deployment is running'; exit 1; }
old_container=$(docker compose ps -aq diabetes-api)
old_image=''
if [[ -n "$old_container" ]]; then
  old_image=$(docker inspect --format '{{.Config.Image}}' "$old_container")
fi
# Pull before replacing existing service.
docker pull "$image"
expected_id=$(docker image inspect --format '{{.Id}}' "$image")
verify_live() {
  local container actual
  container=$(docker compose ps -q diabetes-api)
  [[ -n "$container" ]] || return 1
  actual=$(docker inspect --format '{{.Image}}' "$container")
  [[ "$actual" == "$expected_id" ]] || return 1
  API_BASE_URL="http://127.0.0.1:${DIABETES_PORT}" python3 test_api.py
}
if docker compose up -d --no-build --wait --wait-timeout 180 && verify_live; then
  printf '%s
' "$image" > "$state/current-image.tmp"
  mv "$state/current-image.tmp" "$state/current-image"
  if [[ -n "$old_image" && "$old_image" != "$image" ]]; then printf '%s
' "$old_image" > "$state/previous-image"; fi
  docker compose ps
  echo "DEPLOYMENT PASSED: $image"
else
  echo 'Deployment failed; collecting logs and restoring prior container image.' >&2
  docker compose logs --tail 100 || true
  if [[ -n "$old_image" ]]; then
    export DIABETES_IMAGE="$old_image"
    if docker compose up -d --no-build --wait --wait-timeout 180; then
      echo 'Previous image restored and ready. Validate old-version golden fixtures separately.' >&2
    else
      echo 'Rollback failed; inspect Compose logs manually.' >&2
    fi
  else
    docker compose down || true
  fi
  exit 1
fi
