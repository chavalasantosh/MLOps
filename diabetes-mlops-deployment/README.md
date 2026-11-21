# Diabetes MLOps — educational deployment

Public UCI dataset, educational classification only; not a medical device or diagnosis. This repository contains a trained Bento model, synthetic API fixtures, Docker/Compose deployment and GitHub Actions. Training and private experiment files remain on Google Drive.

## Local (optional)
Prerequisites: Linux/amd64 or compatible platform, Docker Engine, Compose v2 with `--wait`, Python 3, bash. From repository root:

```bash
python3 verify_artifact.py
docker build -t diabetes-bentoml:v1 .
bash scripts/smoke.sh
```

## CI/CD
- PR to `main`: `.github/workflows/ci.yml` builds and tests the image.
- Push to `main`: `.github/workflows/cd.yml` builds, tests, then publishes the tested image to GitHub Container Registry. Use the immutable digest printed in the run summary.
- Optional `ENABLE_LAB_AUTO_DEPLOY=true`: dedicated trusted runner label `diabetes-lab` and protected `lab` environment; never enable on a temporary/shared/untrusted host.

## Deploy a published image (no build)
Clone this repository at the commit reported by the successful Actions run. Authenticate only if package is private and the host is trusted. Copy the digest reference from Actions:

```bash
bash scripts/deploy.sh 'ghcr.io/OWNER/REPO@sha256:REPLACE_WITH_REAL_DIGEST'
docker compose ps
API_BASE_URL=http://127.0.0.1:3001 python3 test_api.py
```

The placeholder is intentionally invalid. Replace it with the real lowercase digest reference. Script validates syntax, pulls the image, starts Compose, checks readiness, verifies running image ID and tests golden responses. If update fails, it attempts to restore previous container image. Rollback is not zero-downtime; use old-version fixtures to validate previous behavior. To stop: `docker compose down`. Never expose the educational service publicly.

## Files
`service.py`, `diabetes_model.bentomodel`, `model_manifest.json`, `evaluation_summary.json`, `requirements.txt`, `Dockerfile`, `compose.yaml`, `api_fixtures.json`, `test_api.py`, `verify_artifact.py`, `scripts/`, `.github/workflows/`.

## Data source and documentation
- https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset
- https://mlflow.org/docs/latest/ml/mlflow-3
- https://docs.bentoml.org/en/latest/examples/mlflow.html
- https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images

No monitoring stack is included. Dependency pins cover direct model and serving requirements, not every transitive dependency. Build, smoke tests and digest verification are required before release. Review dependency security and data/model distribution permissions before publishing.
