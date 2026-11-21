---
title: "Diabetes MLOps | Complete Engineering Documentation"
subtitle: "Reproducible lab, systems design, reference source, operations, observability & incident runbooks"
author: "Project technical documentation — reconstructed from source and executed evidence"
date: "21 November 2026"
lang: en
---

# A fully sourced technical manual — expanded edition

**Version:** 2.0 • **Coverage:** 266 notebook cells (115 code cells), 99 guided development steps, GitHub CI/CD, GitHub Container Registry, Ubuntu deployment, BentoML, Prometheus, Grafana and observed test evidence.

**Important honesty note.** The earlier 23-page handbook was a summary, not the complete engineering record. This is an expanded, source-driven manual. The notebook's executable code is reproduced in a complete annex. Immutable GitHub permalinks identify the current deployed source files; local monitoring instructions come from the observed Ubuntu terminal and screenshots. Some training outputs and raw run reports live in private Google Drive and cannot be reproduced as if they were present in the deployment repository.

**Not for medical use:** This educational model must not be used to diagnose or treat anyone. The training dataset and holdout were limited, and the measured scores are not a clinical validation.

## How to use this manual

- **Start with the architecture and Parts I-VI** to understand the entire project in straightforward language.
- **Read Part VII** for exact reproduction procedures, caveats, evidence, and engineering trade-offs.
- **Follow the Part VIII notebook source annex** when you want every original step and Python source cell, without skipping ahead.
- **Use the original `.ipynb` in GitHub for execution and generated charts**: a typeset document cannot replace an interactive Colab notebook.
- **Use the version-locked repository links** for the exact merged source of the deployed v1.1.0 app. Do not rely on latest `main` if it changes in the future.

## Detailed navigation

- **Parts I–II:** Data preparation, training, evaluation, MLflow, BentoML and API contracts.
- **Part III:** Git feature branches, pull requests, CI/CD, GHCR and Ubuntu deployments.
- **Parts IV–VI:** Prometheus/Grafana, troubleshooting, command references and glossary.
- **Part VII:** Source-aligned reproducibility guide, operational runbooks, evidence and limitations.
- **Part VIII:** Every Markdown and Python code cell from the 266-cell guided Colab notebook.
- **Part IX:** The two actual Grafana screenshots and evidence checklist.
- **Part X:** Source-of-truth hierarchy and documentation boundaries.

## System architecture in one diagram

![Project lifecycle: training, release and observability](architecture_diagram.png){width=6.3in}

**Runtime source of truth:** Git revision `5405f94ab490a4fb1fb4c6b33faae5f732f411c8`; deployed container digest `sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e`. Monitoring was added separately on an authorized lab VM.

---

## Foundation handbook (original overview, revised and expanded)

**As observed:** 21 November 2026  
**Project:** [chavalasantosh/MLOps](https://github.com/chavalasantosh/MLOps)  
**Scope:** Learning project from UCI data through model training, experiment management, API serving, Docker, GitHub Actions, GHCR, Ubuntu deployment, Prometheus, and Grafana.

## Contents at a glance

- **Part I — Fundamentals and data science:** purpose, tools, UCI data, cleaning, features, five-model comparison, final SVM evaluation.
- **Part II — Tracking and serving:** MLflow, model registry, BentoML, API requests, Docker image, regression tests.
- **Part III — CI/CD and deployment:** Git branches, pull requests, GitHub Actions, GHCR, immutable digests, Ubuntu deployment and rollback.
- **Part IV — Monitoring and observability:** BentoML metrics, Prometheus, Grafana, counters, traffic graphs, latency and logs.
- **Part V — Operations:** command reference, troubleshooting, limitations, security and next lessons.
- **Part VI — FAQ and glossary:** everyday explanations and source references.

> **Important safety boundary:** The model is an **educational classifier**, not a validated medical device or a diagnostic tool. Its predictions must not be used for patient care or clinical decisions. The public dataset is small; strong test-set metrics do not establish clinical performance.

---

## Quick orientation — read this first

**In one sentence:** We took a public diabetes-risk dataset, cleaned it, trained and compared five models, logged experiments in MLflow, selected an SVM, packaged it with BentoML, tested the API, built a Docker image in GitHub Actions, published that image to GHCR, deployed it on Ubuntu, and monitored live requests with Prometheus and Grafana.

### The whole architecture

```text
UCI diabetes dataset (public ZIP)
    |
    v
Google Colab + Google Drive
    | raw archive -> schema checks -> cleaning -> duplicate handling
    | grouped development/holdout split -> feature engineering
    | five candidate models -> cross-validation -> tuning -> final evaluation
    v
MLflow tracking + registry
    | experiment runs, metrics, model artifacts
    | selected model: Diabetes-Risk-Educational, registry version 1
    v
BentoML model + typed prediction API
    | /predict, /version, /readyz, /metrics
    v
GitHub repository
    | feature branch -> Pull Request -> ci.yml (build + tests)
    | merge into main -> cd.yml (build + tests + publish)
    v
GHCR: immutable Docker image digest
    |
    v
Ubuntu VM: scripts/deploy.sh + Docker Compose
    | running BentoML container on 127.0.0.1:3001
    | /metrics on the same application
    v
Prometheus (every 10 seconds) -> Grafana dashboard
    | request count, rejected inputs, scrape availability, latency
```

### What was actually verified

| Stage | Evidence and current state |
|---|---|
| Guided Colab training | `Diabetes_MLOps.ipynb`, 266 cells including 115 code cells, executed during the lab |
| Selected algorithm | SVM; see `model_manifest.json` and `evaluation_summary.json` |
| Final holdout | 51 examples; F1 = 0.9722; recall = 0.9722; balanced accuracy = 0.9528 |
| Source delivery | Pull Request #1 merged into `main`, commit `5405f94` |
| Pull-request CI | Build and regression tests passed |
| Main-branch delivery | [GitHub Actions run 44887546219](https://github.com/chavalasantosh/MLOps/actions/runs/44887546219) succeeded and published a new digest |
| Latest deployed release | Application v1.1.0; 13/13 regression tests passed; container `healthy` |
| Metrics | BentoML `/metrics` returned HTTP 200 with request counters and request-duration histogram |
| Prometheus | `up{job="diabetes-api"} = 1` and successful prediction count returned `8` at first inspection |
| Grafana | Working dashboard; counts changed from 8 to 16 successes and 6 to 12 rejections after tests |
| Automatic production rollout | **Not enabled**; the optional self-hosted deployment job was skipped |
| Model drift, alert delivery, Kubernetes | **Not yet built** |

---

# Part I — Fundamentals and data science

## 1. Why this project exists

The purpose is to learn the *whole machine-learning lifecycle*, not merely train a model once. A notebook with a high accuracy number is only the start. An MLOps engineer must also know which data and code produced a model, which experiment won, how to package the exact model, how to test and deliver a service, how to recover a known-good release, and how to notice problems after deployment.

Five distinct responsibilities appear in the project:

1. **Data and ML development:** obtain, clean, analyze, transform, train, and evaluate data.
2. **Experiment management:** retain runs, metrics, parameters, model files, and lineage.
3. **Model serving:** expose a repeatable and validated prediction interface.
4. **Delivery and operations:** build, verify, publish, and deploy immutable containers.
5. **Observability:** collect and visualize how the running service behaves.

### A simple analogy

Think of a restaurant: the *recipe* is the Dockerfile; the *prepared meal packaged for delivery* is the Docker image; the *restaurant kitchen currently preparing orders* is the running container; a *quality-control inspector* is CI; a *delivery service* is CD/GHCR; *the manager recording orders* is BentoML metrics; *the accountant collecting records* is Prometheus; and *the wallboard* is Grafana.

## 2. Technologies and what each one does

| Technology | Its job in this project | Beginner translation |
|---|---|---|
| UCI Repository | Public source of early-stage diabetes risk data | Where the original table came from |
| Google Colab | Interactive Python notebook execution | Development laboratory |
| Google Drive | Persistent private run outputs and backups | Filing cabinet for experiment evidence |
| pandas / NumPy | Data processing and numerical work | Prepare and inspect the table |
| scikit-learn | Preprocessing pipelines, training, cross-validation | Train and test candidate algorithms |
| MLflow | Experiment tracking and registered model versions | Model experiment notebook with IDs |
| BentoML | Saved-model loading and typed HTTP API | Makes the model answer requests |
| Pydantic | Validate the exact request format | Rejects wrong or missing fields |
| Docker | Reproducible runnable application image | Package code, dependencies, and model |
| Docker Compose | Run and manage the application container | Reproducible container configuration |
| Git + GitHub | Version history, branches, PRs | Collaboration and code history |
| GitHub Actions | CI and CD automation | Robot that builds and tests changes |
| GHCR | Docker image registry | Warehouse of tested image versions |
| Ubuntu VM | Deployment runtime | Machine hosting the API |
| Prometheus | Scrape/store time-series measurements | Collector and historian |
| Grafana | Query/display time-series measurements | Live monitoring dashboard |

## 3. Source dataset and data lineage

The notebook downloads the **UCI Early Stage Diabetes Risk Prediction** dataset, dataset ID 529, from:

`https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip`

The notebook anticipates a raw shape of approximately **520 rows and 17 columns**: 16 input fields and one target label. It stores the downloaded ZIP without changing its bytes and records a SHA-256 hash, then extracts the expected `diabetes_data_upload.csv` entry. This supports *lineage*: being able to identify exactly which source artifact produced a result.

A **SHA-256 hash** is a long fingerprint of bytes. Changing a file changes its hash with overwhelming probability. It helps verify integrity, although a matching model plus tampered manifest would require stronger trust controls to detect.

The recorded dataset SHA-256 for the released run is:

`7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137`

The notebook creates run-specific directories like:

```text
/content/drive/MyDrive/Diabetes_MLOps_Master/
    raw/<raw-sha256>/                  # immutable downloaded data
    runs/<run-id>/
        extracted/                     # extracted original CSV copy
        processed/                     # cleaned data and split references
        reports/                       # EDA, audit and evaluation outputs
        mlflow_artifacts/              # artifacts saved for tracking
```

**Critical separation:** Private experiment artifacts and the MLflow tracking store are not the same as the GitHub deployment bundle. The repository contains the selected exported model and test fixtures, not the entire Drive working area.

## 4. Data schema — all 16 inputs

The serving API accepts exactly the following fields under the JSON key `input_data`.

| Field | Accepted value |
|---|---|
| `age` | Numeric, 1 through 120 inclusive |
| `gender` | `Male` or `Female` (as in this dataset's limited source schema) |
| `polyuria` | `Yes` or `No` |
| `polydipsia` | `Yes` or `No` |
| `sudden_weight_loss` | `Yes` or `No` |
| `weakness` | `Yes` or `No` |
| `polyphagia` | `Yes` or `No` |
| `genital_thrush` | `Yes` or `No` |
| `visual_blurring` | `Yes` or `No` |
| `itching` | `Yes` or `No` |
| `irritability` | `Yes` or `No` |
| `delayed_healing` | `Yes` or `No` |
| `partial_paresis` | `Yes` or `No` |
| `muscle_stiffness` | `Yes` or `No` |
| `alopecia` | `Yes` or `No` |
| `obesity` | `Yes` or `No` |

The training target is `class`: `Negative -> 0`, `Positive -> 1`. **These are historical dataset labels, not validated individual diagnoses.** The API enforces the dataset's strict categories and forbids extra inputs; those category constraints are limitations of this demonstration rather than a clinically inclusive design.

## 5. Data cleaning and quality controls

The notebook deliberately keeps the original raw data unchanged and works on copies. Its cleaning process:

1. Normalizes column names to lowercase underscore-separated names.
2. Strips whitespace from values and interprets blanks and `?` as missing.
3. Converts `age` to a number and standardizes category capitalization.
4. Validates age range, gender, target classes, and Yes/No symptom values.
5. Quarantines invalid records rather than guessing missing or erroneous values.
6. Removes exact duplicate feature-plus-label rows.
7. Encodes the target as 0/1.
8. Creates feature-profile hashes to keep identical feature profiles together during train/test splitting, even when their labels conflict.
9. Saves `cleaned.csv` and `cleaning_audit.json` for inspection.

**Why avoid duplicates across splits?** If an almost identical profile appears in both training and evaluation, the score may look better than genuinely unseen performance. The grouped split reduces that particular leakage risk.

## 6. Feature engineering, splitting, and exploratory analysis

The 16-field public API stays unchanged. Derived features live *inside the sklearn pipeline*, which also learns the preprocessing parameters. This avoids incorrectly fitting transformations on evaluation data.

The feature transformer includes median imputation and standardization for age, one-hot encoding for categorical variables, a count of `Yes` symptoms (symptom burden), and age buckets. All learned transformations are fitted inside training folds, not globally before cross-validation.

A reproducible seed of **42** was used. The notebook uses `StratifiedGroupKFold` with five folds: one group-separated holdout fold, and a separate five-fold grouped cross-validation on the development portion. It stores split membership and index files for reproducibility.

EDA includes class balance, age and gender distributions, symptom prevalence, correlations, symptom co-occurrence, subgroup summaries, false-positive and false-negative investigation, ROC/PR curves, calibration, and feature importance. Association charts are **descriptive, not evidence of causality**.

## 7. Five candidate models and automated tuning

The five candidate algorithms were:

| Model | Simple intuition |
|---|---|
| Logistic Regression | Combines weighted feature signals into a probability-like score |
| Random Forest | Aggregates many decision trees |
| Gradient Boosting | Builds trees sequentially to correct earlier errors |
| Support Vector Machine (SVM) | Learns a separating boundary between categories |
| K-Nearest Neighbors (KNN) | Compares a new example with nearby known examples |

The development comparison ranks candidate models primarily by **cross-validation F1**, then recall, then fold variability. The notebook logs five baseline models, checks saved-model consistency, applies development-data rules to choose tuning grids, and uses `GridSearchCV` to retrain the candidates. This is *deterministic tuning triggered by analysis*, not an always-running scheduled production retraining service.

After comparing candidates and checking stability, the notebook freezes one winner **before** reading the untouched final holdout. The final selected algorithm in the shipped files is **SVM**.

## 8. Evaluation results — actual shipped values

Read from `diabetes-mlops-deployment/evaluation_summary.json` and `model_manifest.json`:

| Metric | Value | Interpretation |
|---|---:|---|
| Accuracy | 0.9608 | Fraction of all holdout examples predicted correctly |
| Balanced accuracy | 0.9528 | Average class-specific recall |
| Precision | 0.9722 | Of predicted positives, fraction correctly labeled positive |
| Recall | 0.9722 | Of actual positives, fraction recovered |
| F1 | 0.9722 | Harmonic balance of precision and recall |
| ROC-AUC | 0.9815 | Ranking/discrimination summary |
| Average precision | 0.9914 | Precision-recall ranking summary |
| Brier score | 0.0451 | Mean squared probability error; lower is better |
| Holdout rows | 51 | Small final evaluation sample |

The **confusion matrix** is `[[14, 1], [1, 35]]`, with rows ordered actual Negative/Positive and columns predicted Negative/Positive:

| | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 14 | 1 |
| Actual positive | 1 | 35 |

In this holdout, there was **one false positive and one false negative**. The reported approximate 95% grouped-bootstrap ranges are **F1: 0.9290–1.0000**, **recall: 0.9167–1.0000**, based on 500 valid resamples. They reflect uncertainty within this limited sample and method; they do not establish population-level or clinical generalization.

The **educational acceptance gate**, declared before final evaluation, was F1 >= 0.75, recall >= 0.75, and balanced accuracy >= 0.70. The released model passed that educational gate.

---

# Part II — Tracking, registration, packaging, and serving

## 9. MLflow experiments and model registry

**Experiment** = group of related runs. **Run** = one recorded training/evaluation execution. **Artifact** = output file such as a serialized model. **Registry** = named versions of models. **Alias** = movable name referring to a registry version.

The notebook uses a local SQLite tracking database during Colab execution and persists experiment artifacts/backups in Drive. It creates distinct experiment groupings for baselines, audits, tuning, and final validation. It logs parameters, fold metrics, aggregate metrics, model references, and lineage.

Shipped registry references:

```text
Registered model: Diabetes-Risk-Educational
Version:          1
Alias:            candidate
Alias URI:        models:/Diabetes-Risk-Educational@candidate
Algorithm:        SVM
Bento tag:        diabetes_risk_educational:boucwoxfq2zkeasc
```

**Important difference:** `candidate` can later be reassigned to another MLflow registry version. The deployed Bento artifact was exported from the resolved **specific version**, so a later alias change does not silently update an existing Docker container.

### The MLflow compatibility issue encountered in Colab

During the original run, `mlflow==3.5.1` conflicted with the installed SQLAlchemy 2.1.x environment. The working solution was to pin SQLAlchemy to a compatible 2.0 release and restart Colab. The current deployment requirements pin `SQLAlchemy==2.0.58`.

## 10. BentoML service and the model contract

The current `service.py` contains a `@bentoml.service` called `DiabetesClassifier`. It loads the pinned saved model once during initialization and exposes two application methods decorated with `@bentoml.api`:

```python
@bentoml.service
class DiabetesClassifier:
    @bentoml.api
    def predict(self, input_data: PatientInput) -> dict:
        ...

    @bentoml.api
    def version(self) -> dict:
        ...
```

This is a compact illustration of the service class and method signatures. To make code changes, use the full implementation and indentation from the repository's `service.py`.

The current HTTP contract is:

| Endpoint | Method | Purpose | Expected response |
|---|---|---|---|
| `/predict` | POST | Validate features and produce educational classification | 200 + JSON prediction |
| `/version` | POST | Identify deployed application version | 200 + JSON version |
| `/readyz` | GET | Readiness probe | 200 when ready |
| `/metrics` | GET | Prometheus text measurements | 200 + metric samples |

A valid request (from `test_request.json`):

```json
{
  "input_data": {
    "age": 45.0,
    "gender": "Female",
    "polyuria": "Yes",
    "polydipsia": "Yes",
    "sudden_weight_loss": "No",
    "weakness": "Yes",
    "polyphagia": "No",
    "genital_thrush": "No",
    "visual_blurring": "No",
    "itching": "No",
    "irritability": "No",
    "delayed_healing": "No",
    "partial_paresis": "No",
    "muscle_stiffness": "No",
    "alopecia": "No",
    "obesity": "No"
  }
}
```

Expected synthetic fixture response:

```json
{"prediction": 1, "label": "Positive", "model": "SVM-Educational", "registry_version": "1"}
```

**How a request is processed:** BentoML routes the HTTP request -> Pydantic checks the schema -> pandas makes one-row DataFrame -> saved sklearn pipeline transforms inputs and predicts -> API returns JSON.

If age is missing, out of range, or a category is invalid, request validation returns **HTTP 400** in the observed BentoML version. Calling `/predict` with GET returned **405**. These are expected rejected requests, not necessarily system failures.

### Manual checks

```bash
curl -i http://127.0.0.1:3001/readyz
curl -sS -X POST http://127.0.0.1:3001/version \
  -H 'Content-Type: application/json' -d '{}'
# From diabetes-mlops-deployment/:
curl -sS -X POST http://127.0.0.1:3001/predict \
  -H 'Content-Type: application/json' --data-binary @test_request.json
```

The version endpoint returns `application: Diabetes MLOps`, `version: 1.1.0`, `framework: BentoML`, `purpose: Educational`.

## 11. Deployment bundle — what each file means

The deployed code lives under `diabetes-mlops-deployment/` in the GitHub repository. GitHub Actions workflow files live at the **repository root**, not inside that folder.

```text
MLOps/
  .gitattributes
  .github/
    workflows/
      ci.yml                  # pull-request validation
      cd.yml                  # main-branch image delivery
  Diabetes_MLOps.ipynb       # full guided Colab development
  diabetes-mlops-deployment/
    Dockerfile               # instructions to build image
    compose.yaml             # how the application container runs
    requirements.txt         # pinned Python serving packages
    service.py               # BentoML HTTP service
    diabetes_model.bentomodel# exported model archive
    model_manifest.json      # provenance + hashes + evaluation metrics
    evaluation_summary.json  # final holdout evidence
    api_fixtures.json        # three synthetic golden examples
    test_request.json        # one example input
    test_api.py              # 13 API regression tests
    verify_artifact.py       # checksum + gate + fixture check
    scripts/
      smoke.sh               # CI container test and cleanup
      deploy.sh              # deploy exact image + rollback attempt
    README.md
```

**Not tracked in this repository:** the manually created `monitoring/prometheus.yml`, manually created Grafana dashboard JSON, live Prometheus database, or local Grafana Docker volume. These were added on Ubuntu in later lessons. Back them up separately if the VM is ephemeral.

## 12. Dockerfile, image, container, and Compose

A **Dockerfile** is a build recipe. A **Docker image** is its reusable output. A **container** is a running/stopped instance of that image. A **Compose file** records settings for containerized services.

Our `Dockerfile` starts from `python:3.13-slim`, installs pinned requirements, creates a non-root user, verifies the model archive's SHA-256, imports that model into BentoML's store, and starts the BentoML service on container port **3000**. Using an unprivileged application user reduces the impact of some container compromises but is not a complete security boundary.

The current `compose.yaml` specifies:

```yaml
name: diabetes-mlops
services:
  diabetes-api:
    image: ${DIABETES_IMAGE:-diabetes-bentoml:v1}
    pull_policy: never
    ports:
      - '127.0.0.1:${DIABETES_PORT:-3001}:3000'
    restart: unless-stopped
    mem_limit: 1g
    cpus: 1.0
    healthcheck:
      test: ['CMD', 'python', '-c', "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/readyz', timeout=5)"]
      interval: 10s
      timeout: 8s
      retries: 6
      start_period: 60s
```

The mapping `127.0.0.1:3001:3000` means **host localhost port 3001** forwards to **container port 3000**. The API is intentionally not exposed on a public network interface. `pull_policy: never` means Docker Compose expects the named image to already exist locally; `deploy.sh` runs `docker pull` first.

The healthcheck asks the API for `/readyz`. A healthy container demonstrates readiness; it does **not** prove prediction quality, low latency, or zero errors.

### Optional manual Docker lab

From inside `diabetes-mlops-deployment/` on a compatible trusted machine:

```bash
python3 verify_artifact.py
bash -n scripts/smoke.sh scripts/deploy.sh
docker build -t diabetes-bentoml:v1 .
docker image ls
docker history diabetes-bentoml:v1
bash scripts/smoke.sh
```

The smoke script starts an isolated Compose project, waits for health, runs all regression tests, and removes the test container afterward. Do not use `scripts/smoke.sh` on the live deployment as an ad-hoc substitute for deployed-service testing.

## 13. The 13 API regression tests

The original release had 12 tests. The v1.1.0 feature added a thirteenth test for `/version`.

| Test | Why it matters |
|---|---|
| API readiness | Server started and can accept requests |
| Three synthetic golden predictions | Served outputs match the frozen model's expected outputs |
| Negative age | Rejects out-of-range age |
| Age above 120 | Rejects out-of-range age |
| Unknown symptom | Rejects invalid Yes/No category |
| Unknown gender | Enforces current API categories |
| Extra field | No unexpected input accepted |
| Missing age | Required field enforced |
| GET `/predict` | Wrong method returns 405 |
| Repeated prediction consistency | Five repeated calls match the expected output |
| Application version endpoint | `/version` returns the exact v1.1.0 metadata |

**What 13/13 means:** the currently defined API checks passed. It does not guarantee freedom from vulnerabilities, robustness under high load, or clinical validity.

Run the tests against the currently deployed API:

```bash
cd ~/MLOps/diabetes-mlops-deployment
API_BASE_URL=http://127.0.0.1:3001 python3 test_api.py
# Expected for the v1.1.0 source + image: RESULT: 13/13 tests passed
```

Each test run adds **8 successful prediction calls** (three initial golden calls + five repeated calls), **6 deliberately invalid prediction calls**, one wrong-method call, and the `/version` call, plus readiness checks. This explains the jumps observed in Grafana.

---

# Part III — Git, CI/CD, registry, and Ubuntu deployment

## 14. Git workflow — feature branch to main

This was the exact development lifecycle we practiced:

```text
main (stable code)
   |
   +-- create branch: api-version
           |
           +-- add /version to service.py
           +-- add version regression test to test_api.py
           +-- commit and push branch
           +-- open Pull Request #1
           +-- ci.yml checks build + API tests
           +-- merge Pull Request after successful checks
   |
   +-- main receives merge commit 5405f94
           +-- cd.yml rebuilds and tests merged revision
           +-- publishes the tested image to GHCR
```

Common Windows CMD commands:

```bat
cd C:\Users\SCHAVALA\Downloads\mlops
git switch main
git pull --ff-only origin main
git switch -c api-version
rem Edit the source and test files.
git status
git diff
py -m py_compile diabetes-mlops-deployment\service.py diabetes-mlops-deployment\test_api.py
git add diabetes-mlops-deployment\service.py diabetes-mlops-deployment\test_api.py
git commit -m "Add application version endpoint and regression test"
git push -u origin api-version
```

After merging on GitHub, update the local default branch:

```bat
git switch main
git pull --ff-only origin main
```

### What `on:` does in GitHub Actions

This YAML section decides **when** a workflow starts; `jobs:` defines **what it does**.

`ci.yml`:

```yaml
name: Pull request CI
on:
  pull_request:
    branches: [main]
```

`cd.yml`:

```yaml
name: Main CI and GHCR delivery
on:
  push:
    branches: [main]
  workflow_dispatch:
```

- `pull_request` targeting `main`: runs **pre-merge CI**.
- `push` to `main`: runs **post-merge delivery**, also after direct pushes.
- `workflow_dispatch`: gives the manual **Run workflow** option.

**Subtle but important:** The workflow triggers do not, by themselves, prevent merging a failing PR. That requires required status checks/branch protection or repository ruleset enforcement, with availability and bypass behavior depending on repository settings.

### CI steps actually implemented

1. Checkout the source.
2. Verify model artifact/checksum and Python/shell syntax.
3. Validate Compose configuration.
4. Build the Docker image.
5. Inspect image layers.
6. Run `scripts/smoke.sh` against the image.
7. On `main` only, authenticate with temporary GitHub token and publish the **already tested** image.

`ci.yml` uses read-only contents permission. `cd.yml` grants package-write permission to its publish job. Its optional deploy job is configured for a special **trusted self-hosted runner**, but the job was **skipped** in our demonstrated runs.

## 15. GHCR: image tags vs immutable digests

GitHub Container Registry (GHCR) is where the tested Docker images are stored. The repository is `ghcr.io/chavalasantosh/mlops`.

**Tag** = human-friendly pointer, e.g. `sha-5405f94...`, tied to a commit. Tags can potentially be repointed. **Digest** = SHA-256 reference identifying the exact published manifest content. Our deployment script requires a digest to avoid accidentally pulling a different image under a reused tag.

Two verified published releases:

| Application | Source commit | Published image digest |
|---|---|---|
| Initial API (12 tests) | `da2424761d816136412c028f186dba994049d3e6` | `sha256:79ebe5a13f50a96ffa5c140d01cb84c439221bd2b6eacc7bfb9ff54c7f8b1b67` |
| v1.1.0 API (13 tests) | `5405f94ab490a4fb1fb4c6b33faae5f732f411c8` | `sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e` |

Find the complete digest in **GitHub Actions -> successful Main CI and GHCR delivery run -> publish summary -> Published tested image**. The Packages/Versions view can show additional platform-specific manifests and attestations; do not substitute an unrelated digest entry.

Our CD workflow extracts the digest from the completed `docker push` and reports an exact image reference in the run summary:

```bash
repo="ghcr.io/${GITHUB_REPOSITORY,,}"
# ... docker push "$tag" | tee push.log
digest=$(sed -nE 's/.*digest: (sha256:[0-9a-f]{64}).*/\1/p' push.log | tail -1)
ref="$repo@$digest"
echo "image=$ref" >> "$GITHUB_OUTPUT"
```

## 16. Deploy the tested v1.1.0 image to Ubuntu

Prerequisites: a compatible Linux/amd64 trusted Ubuntu host, working Docker daemon, Docker Compose, Bash, Python 3, network access to GHCR, and enough free disk. Ephemeral training VMs can disappear; don't assume state survives a reset.

From a **fresh** Ubuntu environment:

```bash
git clone https://github.com/chavalasantosh/MLOps.git ~/MLOps
cd ~/MLOps
git checkout 5405f94ab490a4fb1fb4c6b33faae5f732f411c8
cd diabetes-mlops-deployment

docker --version
docker compose version
python3 --version
df -h /

bash scripts/deploy.sh 'ghcr.io/chavalasantosh/mlops@sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e'
```

If your clone lives under `~/mlops-automation` rather than `~/MLOps`, use your actual path. **The script is in `diabetes-mlops-deployment/scripts/`, not directly under the GitHub repository root.**

The deployment script:

1. Checks that the argument looks like a lowercase immutable GHCR digest reference.
2. Sets the Compose project name, port, and image variables.
3. Uses `flock` so another copy of the deployment script cannot operate concurrently.
4. Records the image of the currently running Compose service if one exists.
5. Pulls the requested image *before* replacing the existing service.
6. Starts/recreates the Compose service and waits until healthy.
7. Verifies that the live container has the expected image ID.
8. Executes the matching `test_api.py` suite.
9. On success, writes current/previous image references and reports `DEPLOYMENT PASSED`.
10. On a failed deployment, attempts to restore the old image and waits for health.

This is **automated scripted deployment**, not automatically triggered deployment from GitHub onto the temporary Ubuntu VM. The latter is intentionally disabled.

### Verify the live service

```bash
cd ~/MLOps/diabetes-mlops-deployment
docker compose ps
docker ps
docker stats --no-stream
curl -sS -i http://127.0.0.1:3001/readyz
curl -sS -X POST http://127.0.0.1:3001/version \
  -H 'Content-Type: application/json' -d '{}'
python3 test_api.py
```

Expected v1.1.0 behavior: container **healthy**, `/readyz` HTTP 200, `/version` reports `1.1.0`, and `RESULT: 13/13 tests passed`.

### Practical rollback concept

Rollback means redeploying the previous **known-good digest**. For our two schema-compatible releases, use the corresponding code checkout and tests; the old image has only 12 tests and no `/version` endpoint. An example manual rollback to the initial release:

```bash
cd ~/MLOps
git checkout da2424761d816136412c028f186dba994049d3e6
cd diabetes-mlops-deployment
bash scripts/deploy.sh 'ghcr.io/chavalasantosh/mlops@sha256:79ebe5a13f50a96ffa5c140d01cb84c439221bd2b6eacc7bfb9ff54c7f8b1b67'
```

To return to v1.1.0, checkout commit `5405f94...` and redeploy the `ab3da87...` digest. **Rollback causes a brief application interruption** in this single-container configuration; it does not revert external data, databases, or downstream effects. The failure handler restores a prior container image and readiness but does not automatically re-run old-version golden fixtures. Check those separately.

---

# Part IV — Monitoring and observability

## 17. Three questions monitoring answers

1. **Is it available?** Example: `GET /readyz` returns HTTP 200; Prometheus target `up` equals 1.
2. **How much work is it doing?** Example: number of successful and rejected `/predict` calls.
3. **How fast is it?** Example: request handling duration in milliseconds and how it changes over time.

A service may be **healthy but slow**, or **fast but inaccurate**. Container health, API performance, and model predictive quality are separate concerns.

The monitoring installation performed in the Ubuntu lessons was **manual and local to that VM**. It is not currently deployed by GitHub Actions or the repository's Compose file.

## 18. BentoML already exposes `/metrics`

The v1.1.0 BentoML service exposes Prometheus-compatible text at:

```bash
curl -sS -i http://127.0.0.1:3001/metrics | sed -n '1,40p'
```

The observed output included:

```text
bentoml_service_request_total{endpoint="/predict",http_response_code="200",...} 8.0
bentoml_service_request_total{endpoint="/predict",http_response_code="400",...} 6.0
bentoml_service_request_total{endpoint="/predict",http_response_code="405",...} 1.0
bentoml_service_request_total{endpoint="/version",http_response_code="200",...} 2.0
```

Each metric has a **name**, optional **labels** describing what is counted, and a **value**. The `/predict` 200 value `8` means eight successful prediction requests recorded by this process since its counters started. The `400` values were from intentionally invalid regression tests; they do **not** necessarily mean real users encountered a bug.

### Three metric types you met

| Type | Meaning | Example |
|---|---|---|
| Counter | Accumulated events; normally increases until process restart | `bentoml_service_request_total` |
| Gauge | Value that can rise or fall | `bentoml_service_request_in_progress` |
| Histogram | Groups observed durations into buckets and also records sum/count | `bentoml_service_request_duration_seconds_*` |

Histogram labels such as `le="0.05"` mean **less than or equal to 0.05 seconds**, i.e., 50 milliseconds. Buckets are cumulative: a request in the 50 ms bucket also appears in all larger buckets. Do not add all bucket values together to count requests.

## 19. Prometheus — automatic metrics collection

BentoML provides the numbers, but it does not by itself build a historical time-series store. Prometheus **scrapes** (fetches) `/metrics` at configured intervals and stores samples with timestamps.

Create this file on Ubuntu:

```bash
cd ~/MLOps/diabetes-mlops-deployment
mkdir -p monitoring
cat > monitoring/prometheus.yml <<'EOF_PROM'
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: "diabetes-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["diabetes-api:3000"]
EOF_PROM
```

Why `diabetes-api:3000` instead of `localhost:3001`? Both Prometheus and BentoML are containers on the **same Docker network**. Within that network, Docker can resolve the Compose service name `diabetes-api` and reach its internal port 3000. `localhost` inside Prometheus would refer to Prometheus itself.

Prometheus was started with:

```bash
docker run -d \
  --name diabetes-prometheus \
  --network diabetes-mlops_default \
  -p 127.0.0.1:9090:9090 \
  -v "$PWD/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:v3.15.0 \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time=1d \
  --storage.tsdb.retention.size=200MB
```

**Caveat:** That learning command did not create a persistent Prometheus data volume, and the container was not started with a restart policy. A deleted/recreated container can lose local metrics history, and a host restart may require manual recovery. This is acceptable for an ephemeral lab but not production persistence.

Verify:

```bash
curl -sS http://127.0.0.1:9090/-/ready
curl -sSG --data-urlencode 'query=up{job="diabetes-api"}' \
  http://127.0.0.1:9090/api/v1/query | python3 -m json.tool
```

`up = 1` = Prometheus successfully scraped that target most recently. `up = 0` = scrape failed. This is a **scrape-health** metric; it does not prove the classifier's predictions are correct.

### First PromQL query

```promql
sum(bentoml_service_request_total{endpoint="/predict",http_response_code="200"})
```

Read it in English: “Find the counter of HTTP 200 requests to `/predict` and add matching series.” The screenshot first showed `8`, then the next regression run increased this to `16`.

## 20. Grafana — a window onto Prometheus

Grafana visualizes stored data; it does not replace Prometheus as the metrics database in this setup.

The lab created a persistent Grafana Docker volume and an admin password file:

```bash
docker volume create diabetes-grafana-data
umask 077
GRAFANA_ADMIN_PASS="$(openssl rand -hex 16)"
printf '%s\n' "$GRAFANA_ADMIN_PASS" > ~/grafana-lab-password.txt
```

The lab started Grafana with:

```bash
docker run -d \
  --name diabetes-grafana \
  --network diabetes-mlops_default \
  -p 127.0.0.1:3002:3000 \
  -v diabetes-grafana-data:/var/lib/grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASS" \
  grafana/grafana:latest
```

A `latest` image tag was convenient for the lesson but is not reproducible enough for a production deployment. Pin a reviewed image version/digest in a future hardening pass. Grafana uses **host 3002 -> container 3000**, separately from BentoML's **host 3001 -> container 3000**.

Check Grafana health:

```bash
curl -sS -i http://127.0.0.1:3002/api/health
# Expect HTTP 200 and JSON containing "database":"ok" once ready.
```

An empty response piped into `python3 -m json.tool` produces `Expecting value: line 1 column 1`; inspect the HTTP response with `curl -sS -i` rather than assuming the service is broken. Check `docker logs --tail 40 diabetes-grafana` if needed.

### Login and data source

Grafana username: **`admin`**. The initially configured password is kept privately in `~/grafana-lab-password.txt` (unless you changed it after login). **Never paste passwords into chats, commits, or dashboard JSON files.**

Inside Grafana: **Connections -> Data sources -> Add data source -> Prometheus**. Configure URL:

`http://diabetes-prometheus:9090`

Set **Save & test**. The existing lab also configured this data source through the Grafana API using UID `diabetes-prometheus`. Local dashboard URL:

`http://127.0.0.1:3002/d/diabetes-mlops-monitor`

Since the Ubuntu VM is remote, use an authorized secure tunnel/port-forward to open Grafana from another computer. Do not publish port 3002 to the internet merely to access the dashboard.

### The four dashboard panels

| Panel | Meaning | Query concept |
|---|---|---|
| Successful Predictions | Cumulative successful `/predict` requests | Counter filtered by endpoint and 200 |
| Rejected Predictions | Cumulative invalid `/predict` requests | Counter filtered by endpoint and 400 |
| API Metrics Available | Whether Prometheus can scrape the API | `up{job="diabetes-api"}` |
| New panel (traffic graph; recommended title: Prediction Requests Over Time) | Change in cumulative request counts per HTTP status | Counter grouped by response code |

The screenshot verified **8 -> 16 successful requests**, **6 -> 12 rejected requests**, **up = 1**, and a new staircase graph. The rejected calls were deliberately generated by `test_api.py`.

Useful PromQL for the traffic graph:

```promql
sum by (http_response_code) (
  bentoml_service_request_total{endpoint="/predict"}
)
```

This displays **running totals over time**, not requests per second. For clearer charts, ensure the time-series panel does not stack statuses unless you deliberately want a stacked view; stacked chart height can otherwise be mistaken for one status series.

## 21. Latency — how long a prediction takes

**Latency** means elapsed time to respond to a request. The BentoML duration metric measures server-side handling; it is not a full remote client round-trip including all network delays.

- `bentoml_service_request_duration_seconds_sum`: total recorded request-handling seconds for matching series.
- `bentoml_service_request_duration_seconds_count`: number of observations.
- Average = total time / count.
- Seconds -> milliseconds: multiply by **1,000**.

From one observed VM scrape: the total successful prediction time was **0.31956829 seconds** for **8** successful predictions. So the average was about **39.95 ms** for those eight requests.

The Grafana **Stat** panel query for a *cumulative* average is:

```promql
1000 *
sum(bentoml_service_request_duration_seconds_sum{endpoint="/predict",http_response_code="200"})
/
sum(bentoml_service_request_duration_seconds_count{endpoint="/predict",http_response_code="200"})
```

Select unit **milliseconds (ms)** and use an **instant query** for the current average. If no successful requests have happened, the denominator can be zero and the panel may display no meaningful value.

### More useful production-style latency: last 5 minutes

A cumulative average can hide a *recent slowdown*. To estimate the average among recent requests, use rates of the counter-style sum and count metrics:

```promql
1000 *
sum(rate(bentoml_service_request_duration_seconds_sum{endpoint="/predict",http_response_code="200"}[5m]))
/
sum(rate(bentoml_service_request_duration_seconds_count{endpoint="/predict",http_response_code="200"}[5m]))
```

If there were no successful calls during the five-minute window, the result may be empty. For a histogram-based approximate 95th percentile:

```promql
1000 * histogram_quantile(
  0.95,
  sum by (le) (
    rate(bentoml_service_request_duration_seconds_bucket{endpoint="/predict",http_response_code="200"}[5m])
  )
)
```

**P95** means roughly 95% of observed durations fall at or below that latency threshold, as estimated from histogram buckets. With sparse traffic or coarse buckets, it can be unstable. This is not the same as the average.

### Traffic and errors per minute

```promql
# Successful prediction requests during the last five minutes
sum(increase(bentoml_service_request_total{endpoint="/predict",http_response_code="200"}[5m]))

# Server-side 5xx failures per second, averaged over the last five minutes
sum(rate(bentoml_service_request_total{endpoint="/predict",http_response_code=~"5.."}[5m]))
```

Prometheus rate and increase account for normal counter resets on process restarts. A 400 response often represents bad client input, not a server outage; do not blindly count all 4xx responses as application failures.

## 22. Generate test traffic and watch the dashboard change

From the running VM with the matching v1.1.0 source:

```bash
cd ~/MLOps/diabetes-mlops-deployment
python3 test_api.py
```

The regression suite produces eight successful `/predict` calls and six rejected `/predict` calls, among other checks. Wait for the next **10-second Prometheus scrape** and Grafana refresh. If the service and counters did not restart and no other prediction requests were sent, the originally displayed `8` and `6` become `16` and `12`.

**Why not instantly?** BentoML increments counters as work happens; Prometheus only learns new values at scrape time; Grafana reads them on its next refresh. The dashboard is *near-real-time*, not a live packet trace.

### What a staircase chart tells you

A cumulative counter graph stays flat when no new requests arrive and steps upward after requests. It can jump back toward zero if the application's metrics process restarts. A falling raw counter after restart is not proof of canceled requests.

## 23. Logs, metrics, traces — related but different

| Signal | Question it answers | Example from this project |
|---|---|---|
| Logs | What happened during one event? | Request path, status, validation traceback |
| Metrics | How much / how often / how fast? | 16 successes, 12 invalid calls, ~40 ms sample mean |
| Traces | Which operations occurred within one request? | BentoML log fields include trace and span identifiers |

The lab did **not** implement end-to-end trace collection and visualization. A log containing a Pydantic `ValidationError` for missing age was generated by an intentional invalid-input test, which returned HTTP 400. It is not evidence of a crashed service.

Helpful commands:

```bash
docker compose ps
docker compose logs --tail=50 diabetes-api
docker logs --tail=50 diabetes-prometheus
docker logs --tail=50 diabetes-grafana
docker stats --no-stream
```

`docker logs` without a container name returns `docker logs requires 1 argument`; use the exact container name or Compose service logs.

## 24. Networking, security, and temporary-VM behavior

Three containers were running in the demonstrated lab:

| Container | Host port | Internal connectivity |
|---|---|---|
| `diabetes-mlops-diabetes-api-1` | 127.0.0.1:3001 | Service name `diabetes-api:3000` |
| `diabetes-prometheus` | 127.0.0.1:9090 | Scrapes `diabetes-api:3000` |
| `diabetes-grafana` | 127.0.0.1:3002 | Queries `diabetes-prometheus:9090` |

Both monitoring containers were attached to the Compose-created `diabetes-mlops_default` network. **If you run `docker compose down`, Compose may remove that network**, interrupting communication for the separately created Prometheus and Grafana containers. For long-lived setups, define all services and a persistent dedicated network in a reviewed Compose configuration.

All published host ports in this lab are bound to **127.0.0.1**, limiting exposure to local access. Protect access via secure SSH port forwarding or an authorized lab proxy. Do not store real patient data, GitHub tokens, or weak Grafana credentials on shared temporary machines.

Use `df -h /` before pulling large images; a previous VM had only about **4.5–5.7 GB** available on its 29 GB root filesystem. Avoid `docker system prune -a` blindly: it could remove the old image required for rollback.

---

# Part V — Operations handbook

## 25. Day-to-day command cheat sheet

### Source and environment

```bash
pwd
git log -1 --oneline
git status
docker --version
docker compose version
python3 --version
df -h /
```

### API and containers

```bash
cd ~/MLOps/diabetes-mlops-deployment
docker compose ps
docker compose logs --tail=30 diabetes-api
docker stats --no-stream
curl -sS -i http://127.0.0.1:3001/readyz
curl -sS -X POST http://127.0.0.1:3001/version -H 'Content-Type: application/json' -d '{}'
python3 test_api.py
```

### Prometheus

```bash
curl -sS http://127.0.0.1:9090/-/ready
curl -sSG --data-urlencode 'query=up{job="diabetes-api"}' \
 http://127.0.0.1:9090/api/v1/query | python3 -m json.tool
curl -sSG --data-urlencode 'query=sum(bentoml_service_request_total{endpoint="/predict",http_response_code="200"})' \
 http://127.0.0.1:9090/api/v1/query | python3 -m json.tool
```

### Grafana

```bash
curl -sS -i http://127.0.0.1:3002/api/health
docker logs --tail=40 diabetes-grafana
# Dashboard, on the VM or a secure port-forward:
# http://127.0.0.1:3002/d/diabetes-mlops-monitor
```

## 26. Troubleshooting by symptom

| Symptom | Likely cause | First safe check |
|---|---|---|
| `scripts/deploy.sh: No such file` | Terminal is in repository root rather than deployment subfolder | `pwd; ls; cd diabetes-mlops-deployment` |
| `docker: command not found` | Docker not installed/on PATH | `docker --version` |
| Docker daemon unavailable | Daemon stopped or access missing | `docker info` |
| Image pull fails | Auth/private package, network, missing image | `docker pull '<exact-digest>'` on trusted host |
| Compose says image not found | `pull_policy: never` plus image not pulled | Use `deploy.sh`, which pulls first |
| `/readyz` not 200 | Service not ready or crashed | `docker compose ps; docker compose logs --tail=100 diabetes-api` |
| 400 for age/symptom | Invalid request data, often intentional in tests | Inspect request body and Pydantic field error |
| 405 for GET `/predict` | Endpoint requires POST | `POST /predict` with JSON |
| `docker logs` needs argument | No container named | `docker compose logs diabetes-api` |
| Prometheus returns `up = 0` | Unable to scrape service | Check target address, Docker network, and BentoML `/metrics` |
| Prometheus query empty | No scrape yet, wrong labels, or no matching series | Check `up`, inspect raw metric names, allow a scrape interval |
| Grafana JSON parser error | Empty/non-JSON HTTP response | `curl -sS -i .../api/health`; check logs |
| Grafana cannot query Prometheus | Wrong data-source URL or network | Use `http://diabetes-prometheus:9090`; inspect network |
| Grafana shows no data | Data source, time range, or query mismatch | Check source health, Grafana time range, query labels |
| Grafana counters remain flat | No new requests, scrape delay, or no refresh | Run tests; inspect `/metrics`; wait scrape + refresh |
| Graph jumps down | Counter reset after restart | Use `rate()`/`increase()` for windowed trends |
| Prometheus/Grafana disappear after VM reset | Temporary VM or standalone container persistence limitations | Recreate lab, or use dedicated persistent host |
| Colab SQLAlchemy import error | MLflow/SQLAlchemy version incompatibility encountered earlier | Check pinned compatible SQLAlchemy and restart Colab |
| CI workflow not appearing | YAML not in repository root `.github/workflows/` | Move workflow files to root, commit, push |
| GitHub Actions Node warning | Old pinned checkout action targets deprecated Node runtime | Update to reviewed supported checkout action in maintenance PR |

## 27. Security, reliability, and what remains to harden

This project is an excellent **educational deployment**, but not a medical or enterprise production service. Before a serious operational rollout, review at least the following:

- **Clinical and data governance:** representativeness, consent/privacy where applicable, clinical evaluation, subgroup performance, bias, decision impact, and labeling quality.
- **Reproducibility:** pin image digests (including Grafana), resolve transitive Python dependencies, archive an SBOM, and protect model artifact lineage.
- **Secrets:** use a secrets manager or protected environment, no committed passwords, restrict token permissions, use trusted runners only.
- **Network security:** TLS, auth, least privilege, firewalls, audited external access; avoid publishing inference or metrics endpoints openly.
- **Supply chain:** dependency vulnerability checks, image scanning, provenance/attestations, signature verification, trusted model deserialization.
- **Reliability:** persistent monitoring storage, restart policies for all services, capacity tests, availability objectives, alerts, backups, recovery drills, and load balancing if needed.
- **Model operations:** input schema drift, prediction distribution drift, delayed ground-truth feedback, actual performance monitoring, retraining policy and approval gates.
- **CD controls:** branch protection and required checks, protected deployment environments, explicit rollout/rollback strategy, staging, and approval for production.

### What has NOT been implemented yet

- Full automatic deployment of a merged commit onto the temporary Ubuntu VM.
- Managed or persistent monitoring stack in the repository's Compose configuration.
- Production notification delivery (email, Slack, PagerDuty, etc.).
- Formal alert rules for error rate, latency, scrape failures, and capacity.
- Continuous data drift monitoring or verified real-world prediction-quality monitoring.
- Scheduled ingestion/retraining/approval/redeployment with fresh datasets.
- Kubernetes deployment, horizontal autoscaling, distributed tracing back end.
- Production compliance or clinical validation.

Do not describe these as complete merely because the core lab is working.

## 28. Recommended next lessons

1. **Grafana average latency and P95 panels:** learn cumulative versus windowed queries, and why traffic volume matters.
2. **Basic alert rules:** create a controlled alert on Prometheus target loss or synthetic slow responses, without notifying external parties yet.
3. **Persistent monitoring Compose stack:** define the API, Prometheus, and Grafana together; add retention volumes, pinned images, safe restart behavior, and provisioning.
4. **Capacity and reliability:** load-test carefully, understand CPU/memory constraints and timeout behavior.
5. **ML monitoring:** separate application monitoring from data drift, prediction drift, and accuracy with ground truth.
6. **Kubernetes fundamentals:** pods, deployments, services, readiness probes, rolling updates, ConfigMaps, Secrets, and observability integration.

---

# Part VI — FAQ and glossary

## 29. Frequently asked questions

**Why do we need Docker if we already have Python code?**  
Docker packages the serving environment and model dependencies so a new compatible machine can run the same tested image. It improves reproducibility but does not make all hosts identical or eliminate security issues.

**Why do we publish to GHCR?**  
A successful CI run can store the tested image once. Other machines pull the same immutable image instead of rebuilding it locally.

**Why use a digest rather than a tag?**  
A digest identifies the exact published content. A mutable tag can later point elsewhere.

**Why run CI both before and after merging?**  
Before merge, catch problems in a proposed change; after merge, test and publish the actual merged revision. Both can be useful because the merged state may differ.

**Did the PR automatically deploy to my Ubuntu VM?**  
No. The post-merge workflow automatically published the image to GHCR; we ran `scripts/deploy.sh` manually on Ubuntu. The optional self-hosted deployment job was skipped.

**What is the difference between `/readyz` and `/metrics`?**  
`/readyz` answers “can the service accept requests?”; `/metrics` reports observed activity and performance counters.

**Is `up = 1` the same as 100% model accuracy?**  
No. `up = 1` only means Prometheus successfully scraped the target. It says nothing about predictive accuracy.

**Why did Grafana's success count increase by eight after running the 13 tests?**  
The thirteen tests are *test cases*, not thirteen successful predictions. Three cases sent one golden prediction each, and another consistency case sent five: eight successful prediction calls total.

**Why does Grafana show six rejected requests?**  
Our tests intentionally send invalid ages, categories, an extra field, and a missing field. Rejecting them is the intended behavior.

**Is latency 40 ms good?**  
It was the average of a few successful requests on one lab VM, not a statistically stable service-level benchmark. “Good” requires a target tied to application needs and enough measurements under representative load.

**Can I run Grafana without Prometheus?**  
Grafana supports many data sources, but the dashboard in this project queries Prometheus. Removing Prometheus breaks these panels unless you supply an alternate compatible source.

**Will my monitoring data remain after this training VM disappears?**  
Not reliably. Grafana has a local Docker volume in the current host, while Prometheus was started without a persistent data volume. Temporary VM teardown can remove both.

**Is retraining already continuously automated in production?**  
No. The notebook includes deterministic tuning/retraining operations during a training run. A persistent retraining schedule/pipeline with fresh data and deployment gates has not been implemented.

## 30. Glossary

| Term | In plain English |
|---|---|
| Artifact | A saved output of training/building, such as a model file |
| Bento model | Model packaged into the BentoML model store |
| CI | Automatic checking/building/testing of code changes |
| CD | Automatic delivery of validated builds and, when configured, deployment |
| Container | Running or stopped instance of an image |
| Counter | Metric accumulating events over time, normally resets when process restarts |
| Cross-validation | Repeated train/validation splits to estimate development performance |
| Data leakage | Evaluation uses information that would not be available in genuine future predictions |
| Digest | Content-based identifier for a Docker image manifest |
| EDA | Exploratory Data Analysis: inspecting patterns and data quality |
| Endpoint | HTTP path such as `/predict` or `/metrics` |
| F1 score | Single metric balancing precision and recall |
| Feature engineering | Transforming inputs into useful representations |
| Gauge | Metric that can increase or decrease |
| GHCR | GitHub Container Registry |
| Golden fixture | Synthetic test input plus expected result |
| Grafana | Visual dashboards for measurements stored in data sources |
| Healthcheck | Automatic readiness/liveness style probe of a running service |
| Histogram | Counts observations in duration/value buckets |
| Holdout | Data reserved for evaluation and not used for training/tuning |
| Immutable image | Docker image referenced by its content digest |
| Latency | Duration required to serve a request |
| MLflow experiment | Collection of related training/evaluation runs |
| Model registry | Named versions of saved models |
| Observability | Understanding running systems through metrics, logs, and traces |
| P95 | Approximate 95th percentile; most requests are faster than this boundary |
| Prometheus | Collects and stores timestamped numerical metrics |
| PromQL | Query language for Prometheus metrics |
| Pull request | Proposed code change asking to merge one branch into another |
| Rate | Per-second change of a counter over a time interval |
| Recall | Of actual positives, proportion correctly predicted positive |
| Rollback | Return to a previously known-good release |
| Scrape | Periodic collection of metrics from an endpoint |
| SHA-256 | Cryptographic hash used as a file or image-content fingerprint |
| SVM | Support Vector Machine classifier |
| Time series | Values associated with timestamps |

---

## 31. Source of truth and references

**Project files (actual implementations):**

- [GitHub repository](https://github.com/chavalasantosh/MLOps)
- [Guided Google Colab notebook](https://github.com/chavalasantosh/MLOps/blob/main/Diabetes_MLOps.ipynb)
- [Current BentoML service](https://github.com/chavalasantosh/MLOps/blob/main/diabetes-mlops-deployment/service.py)
- [Current API regression tests](https://github.com/chavalasantosh/MLOps/blob/main/diabetes-mlops-deployment/test_api.py)
- [Compose configuration](https://github.com/chavalasantosh/MLOps/blob/main/diabetes-mlops-deployment/compose.yaml)
- [Deployment automation](https://github.com/chavalasantosh/MLOps/blob/main/diabetes-mlops-deployment/scripts/deploy.sh)
- [Pull-request workflow](https://github.com/chavalasantosh/MLOps/blob/main/.github/workflows/ci.yml)
- [Main CI/CD workflow](https://github.com/chavalasantosh/MLOps/blob/main/.github/workflows/cd.yml)
- [Model manifest](https://github.com/chavalasantosh/MLOps/blob/main/diabetes-mlops-deployment/model_manifest.json)
- [Evaluation summary](https://github.com/chavalasantosh/MLOps/blob/main/diabetes-mlops-deployment/evaluation_summary.json)
- [PR #1](https://github.com/chavalasantosh/MLOps/pull/1)
- [Successful post-merge delivery run](https://github.com/chavalasantosh/MLOps/actions/runs/44887546219)

**Platform documentation:**

- [UCI dataset page](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)
- [MLflow](https://mlflow.org/docs/latest/)
- [BentoML](https://docs.bentoml.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [GitHub Actions](https://docs.github.com/actions)
- [GitHub Container Registry](https://docs.github.com/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
- [Prometheus](https://prometheus.io/docs/)
- [Grafana](https://grafana.com/docs/grafana/latest/)

### Documentation provenance and limits

This handbook was assembled from the actual repository's current `main`-branch notebook, code, manifests, tests, Compose configuration, and GitHub workflows, plus the completed command outputs and dashboard screenshots shared during the learning sessions. The original private Colab/Drive tracking databases and all saved EDA plots were **not** independently inspected for this handbook. Exact training results quoted here come from the committed final evaluation artifacts. Monitoring configuration and dashboards were created manually on Ubuntu and were confirmed through user logs/screenshots, not a committed infrastructure definition.

**End of handbook.**



---

# Part VII — Exact reproduction and engineering details

## 32. Reproduction contract: what “end to end” actually means

A complete reproduction has **two different paths**, and this distinction matters. The **training path** starts with the UCI archive in Colab, obtains a clean and group-aware split, compares and tunes five algorithms, tracks them in MLflow, freezes a candidate, runs the holdout exactly once, registers the selected model, and exports a BentoML-compatible artifact. The **delivery path** begins with the committed, already-exported model artifact: GitHub Actions validates its checksum, builds the BentoML Docker image, starts it, runs golden API tests, and publishes an exact image digest to GHCR. A target Ubuntu host then pulls that digest and validates the running service. The **observability path** (built later on the Ubuntu machine) adds Prometheus and Grafana to an already-running service.

These paths intentionally do not do everything on every push: **CI does not retrain on each commit**. The training notebook is the source of the model artifact, and the deployment repository carries a pinned export. “Automatic tuning” refers to notebook-driven policy and GridSearchCV, not an online service that continuously learns from live patients. “Continuous delivery” means an image is automatically made available for release; the optional unattended host deployment is **not enabled** in this recorded lab.

### A reproducible release must pin all of these

| Item | Actual recorded source of truth | Why it matters |
|---|---|---|
| Training data | UCI source URL and dataset SHA-256 | A different CSV can change metrics and model behavior. |
| Data cleaning | Notebook and cleaning audit | Quarantine and deduplication change what is modeled. |
| Splits | Notebook run's split indices in Google Drive | Same dataset with a different split may give different scores. |
| Training pipeline | Fitted scikit-learn preprocessing and SVM | Prevents accidentally serving different transformations. |
| Experiment | MLflow run ID/model URI | Traceability from trained artifact back to evaluation. |
| Registry | `Diabetes-Risk-Educational`, version `1`, alias `candidate` | Registry alias is a movable pointer; version is pinned. |
| Serving artifact | `diabetes_model.bentomodel` + `model_manifest.json` | The exported file is verified against a recorded hash. |
| Git revision | Deployment commit SHA | Establishes exactly which code was reviewed and built. |
| Docker image | GHCR reference `@sha256:...` | Pulls the exact published image rather than a mutable tag. |
| API contract | BentoML service + `api_fixtures.json` | Detects accidental changes in prediction and response structure. |
| Monitoring | Local Ubuntu config + Docker runtime | Grafana/Prometheus were added after deployment, not by the CI/CD job. |

## 33. Training environment and storage structure — exact steps

### Preparing Colab

Open `Diabetes_MLOps.ipynb` in Google Colab. Use a compatible runtime and execute cells sequentially; cells depend on variables and paths created earlier. Training and private experiment outputs are stored in Google Drive, but SQLite tracking metadata is kept on a local filesystem during execution, with explicit backups. Never assume a Colab runtime will preserve `/content` after the session ends.

The reference notebook uses these paths and constants:

```python
ROOT = Path('/content/drive/MyDrive/Diabetes_MLOps_Master')
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
RUN = ROOT / 'runs' / RUN_ID
RAW = ROOT / 'raw'
EXTRACTED = RUN / 'extracted'
PROCESSED = RUN / 'processed'
REPORTS = RUN / 'reports'
ARTIFACTS = RUN / 'mlflow_artifacts'
LOCAL = Path('/content/diabetes_mlops') / RUN_ID
DEPLOY = LOCAL / 'deployment'
SEED = 42
N_JOBS = 2
MIN_F1, MIN_RECALL, MIN_BALANCED_ACCURACY = .75, .75, .70
```

The thresholds are **educational release gates**, not clinical thresholds. They were selected before examining the final holdout. Re-run the full notebook to reproduce training; cloning the deployment folder is not a substitute for the original training process.

### Data lineage and cleaning

The notebook downloads an archive from `https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip`, stores the exact archive under a content-addressed `raw/<hash>` directory, then extracts the named `diabetes_data_upload.csv`. The run records the original checksums and creates an independent working copy. It validates **age 1–120**, the source gender categories, the two target labels, and Yes/No symptom categories; invalid rows go into `quarantine.csv`. It removes exact duplicates and groups identical feature profiles to keep them in the same split.

**Leakage prevention:** training-fit operations (imputers, encoders, scalers) belong in a `Pipeline`/`ColumnTransformer` fitted on training folds only. Feature-group overlaps are explicitly asserted absent across development/holdout and cross-validation folds. Target-aware EDA is conducted on the development set, not used to optimize against the protected holdout.

### Feature representation

The public API accepts sixteen raw fields. Inside the fitted pipeline the notebook transforms those raw fields by: median-imputing/scaling age, one-hot encoding categorical inputs, creating a count of Yes-symptoms, and deriving age bins. These engineered features are internal. Clients must send the original 16 fields; they must **not** invent or supply derived features.

### Candidate models and choice

The five candidates are Logistic Regression, Random Forest, Gradient Boosting, SVM and K-nearest neighbors. The notebook evaluates comparable grouped CV folds, logs baselines to MLflow, calculates out-of-fold diagnostics, derives deterministic tuning policies and searches hyperparameter grids. Its selection ranking emphasizes development F1, recall, and lower variability; the candidate is frozen before final holdout evaluation. The shipped winner is **SVM**. Training all five models and selecting SVM is supported by the notebook and released manifest; do not claim that the SVM will be best on another dataset.

## 34. The final evaluation: interpret, do not oversell

The released artifact contains this observed holdout confusion matrix:

| Actual / Predicted | Predicted negative | Predicted positive |
|---|---:|---:|
| Actual negative | 14 | 1 |
| Actual positive | 1 | 35 |

Interpretation: among 15 observed negative examples, 14 were classified negative and one incorrectly positive. Among 36 observed positive examples, 35 were classified positive and one incorrectly negative. These **51 examples are a small holdout**; they do not establish clinical effectiveness or fairness.

| Metric (released artifact) | Value | Plain-English question |
|---|---:|---|
| Accuracy | 0.96078 | How often was the class correct overall? |
| Balanced accuracy | 0.95278 | How well did it recognize both classes when balanced? |
| Precision | 0.97222 | When it predicted positive, how often was it correct? |
| Recall | 0.97222 | Of actual positives, how many were caught? |
| F1 | 0.97222 | Balance between precision and recall? |
| ROC-AUC | 0.98148 | How well do scores rank positives above negatives? |
| Average precision | 0.99139 | Precision/recall ranking summary? |
| Brier score | 0.04507 | How close were predicted probabilities to labels, on average? Lower is better. |

A bootstrap with 500 valid resamples estimated F1 and recall intervals, but the sample is too small and too narrow to support medical use. The MLflow “gate_passed” flag means the project met its **predeclared educational tests**, nothing more.

## 35. MLflow: tracking, registry, and BentoML handoff

**Experiment:** a named group of model-training runs. **Run:** a single execution with parameters, fold metrics, artifacts, and a run ID. **Model URI:** tells MLflow where a logged model can be loaded. **Registered model/version:** a governed name and numbered instance for a model artifact. **Alias:** a human-friendly pointer that can be reassigned to another version.

In this lab, the selected model is recorded as:

```text
Registered model: Diabetes-Risk-Educational
Registry version: 1
Registry alias: candidate
Alias URI: models:/Diabetes-Risk-Educational@candidate
Underlying source run ID: 353a2db8e4fb4276b496740555365db3
Underlying source model URI: models:/m-8bc9a3a4b7c1455b844c7a63bd8e8b33
Bento model tag: diabetes_risk_educational:boucwoxfq2zkeasc
```

The alias is useful for identifying a current candidate, but the export uses a resolved registry **version**, then imports it into BentoML with an explicit tag. This avoids later alias movement silently changing the running model.

The training notebook uses an MLflow 3.5.1 environment and encountered a SQLAlchemy 2.1 incompatibility. The lab recovered with a compatible SQLAlchemy 2.0.x version and a runtime restart; deployment requirements now pin `SQLAlchemy==2.0.58`. A version mismatch between the training and serving environments can break serialized sklearn models. Compatibility tests in Docker are therefore essential.

## 36. The deployed HTTP API — detailed reference

BentoML starts inside the container using:

```text
bentoml serve service:DiabetesClassifier --host 0.0.0.0 --port 3000
```

Docker publishes only `127.0.0.1:3001` on the Ubuntu host. Endpoints:

| Method | Endpoint | Meaning |
|---|---|---|
| `POST` | `/predict` | Parse 16 required fields, preprocess with fitted pipeline, return a class and metadata. |
| `POST` | `/version` | Return application metadata (`1.1.0`, `BentoML`, educational). |
| `GET` | `/readyz` | Health/readiness check; HTTP 200 when ready. |
| `GET` | `/metrics` | Prometheus exposition format for counts, durations, in-progress requests. |

A valid request can be sent using the repository's test file:

```bash
cd ~/MLOps/diabetes-mlops-deployment
curl -sS -i -X POST http://127.0.0.1:3001/predict \
  -H 'Content-Type: application/json' \
  --data-binary @test_request.json
```

A synthetic expected prediction example looks like:

```json
{"prediction":1,"label":"Positive","model":"SVM-Educational","registry_version":"1"}
```

**Input validation:** age outside 1–120, unknown Yes/No values, wrong gender category, omitted fields, or extra fields must be rejected as client errors. The API's built-in validation is Pydantic, and BentoML may render a stack trace in application logs while still returning HTTP 400. This is **expected for the negative tests**; a server-side HTTP 500 is not an acceptable substitute.

### Exactly what the 13 regression checks cover

| # | Test | Purpose |
|---:|---|---|
| 1 | Readiness | The deployed application is accepting requests. |
| 2–4 | Three synthetic golden predictions | Frozen inputs produce exact expected response objects. |
| 5 | Negative age | Rejects invalid numerical values. |
| 6 | Age >120 | Rejects out-of-range input. |
| 7 | Unknown symptom | Rejects values other than Yes/No. |
| 8 | Unknown gender | Rejects values outside source schema. |
| 9 | Extra field | Rejects unexpected input rather than silently ignoring it. |
| 10 | Missing age | Rejects omission of mandatory fields. |
| 11 | GET on `/predict` | Rejects the incorrect HTTP method with 405. |
| 12 | Five repeated predictions | Detects inconsistent results for a fixed input. |
| 13 | Version endpoint | Returns exactly the approved v1.1.0 metadata. |

The count of **13 tests** does not mean 13 HTTP requests. One test executes multiple requests (five repeats), and readiness may poll multiple times. The 13/13 is a count of **assertion groups**, not of all request calls.

## 37. Deployment structure and why each file exists

```text
MLOps/
├── .gitattributes
├── .github/
│   └── workflows/
│       ├── ci.yml                    # pull_request -> build and tests
│       └── cd.yml                    # push main -> build, tests, GHCR
├── Diabetes_MLOps.ipynb             # training + model-export laboratory
└── diabetes-mlops-deployment/
    ├── service.py                    # BentoML API and strict schema
    ├── diabetes_model.bentomodel     # exported, pinned artifact
    ├── model_manifest.json           # hashes, registry lineage, model metrics
    ├── evaluation_summary.json       # selected-model evaluation evidence
    ├── requirements.txt              # serving dependency pins
    ├── Dockerfile                    # image build recipe
    ├── compose.yaml                  # app startup / health / resource limits
    ├── api_fixtures.json             # synthetic golden test vectors
    ├── test_api.py                   # 13 grouped integration tests
    ├── test_request.json            # a synthetic request
    ├── verify_artifact.py           # model hash and fixture checks
    ├── scripts/
    │   ├── smoke.sh                  # temporary CI container + tests
    │   └── deploy.sh                 # digest deployment + attempted rollback
    ├── .dockerignore                 # exclude build-time noise/secrets
    ├── .gitignore                    # exclude local generated state
    └── README.md
```

**Operational files added later, on the Ubuntu VM (not committed to `main` in the evidence):**

```text
diabetes-mlops-deployment/monitoring/prometheus.yml
# Grafana dashboard JSON was created locally and imported using Grafana API.
# Prometheus and Grafana Docker containers were created with docker run.
```

A fresh Git clone contains the checked-in training and deployment files, but does **not** automatically recreate the three-container monitoring setup. Instructions for that follow in chapter 41.

## 38. Docker: build layers, runtime, and port mappings

| Docker concept | Actual configuration | Consequence |
|---|---|---|
| Base image | `python:3.13-slim` | Contains the Python runtime for BentoML. |
| Workdir | `/app` | Relative runtime paths resolve consistently. |
| `requirements.txt` copied before app code | Cached dependency layer | Changing `service.py` may not force all pip packages to reinstall. |
| Service user | `appuser` UID 10001 | Reduces risk compared with running as root. |
| Model import | `bentoml models import` | Image contains the exported model in BentoML's model store. |
| Container port | `3000` | BentoML listens inside Docker. |
| Host port | `127.0.0.1:3001` | Not publicly listening on the Ubuntu interface. |
| Healthcheck | Python urllib on `/readyz` | Docker reports `(healthy)` after successful checks. |
| Resource constraints | 1 GiB RAM, 1 CPU | Safety budget for the educational VM. |

`docker build` creates an **image** (reusable template). `docker run` or `docker compose up` creates a **container** (running instance). The image is not itself a running process. Docker Compose describes services, networking, ports, and health behavior. The application can be tested using either a manual local image (`diabetes-bentoml:v1`) or the exact GHCR digest.

Commands from the deployment directory:

```bash
docker compose config -q              # check configuration
docker compose ps                     # list application service
curl -i http://127.0.0.1:3001/readyz # readiness
docker compose logs --tail=50 diabetes-api
docker stats --no-stream
```

`docker logs` **requires** a container ID or name; `docker compose logs diabetes-api` uses the service name and is easier in this project. `docker ps` lists running containers; `docker ps -a` also includes stopped containers.

## 39. Branching, PR quality gate, CI and CD — complete behavior

```text
feature branch: api-version
  └─ push branch
      └─ open Pull Request #1 against main
          └─ GitHub Actions ci.yml (pull_request)
              ├─ checkout
              ├─ verify model artifact + Python/shell syntax + Compose
              ├─ docker build
              ├─ docker history and docker images
              └─ scripts/smoke.sh: start/test/cleanup
                  └─ 13/13 PASS
                      └─ merge PR into main
                          └─ GitHub Actions cd.yml (push main)
                              ├─ verify artifact + code
                              ├─ build Docker image
                              ├─ same smoke test suite
                              └─ publish immutable GHCR artifact
                                  └─ manual scripted deployment on Ubuntu
```

The two trigger excerpts are intentionally small:

```yaml
# .github/workflows/ci.yml
on:
  pull_request:
    branches: [main]
```

```yaml
# .github/workflows/cd.yml
on:
  push:
    branches: [main]
  workflow_dispatch:
```

A PR created or updated against `main` runs the first pipeline. A PR merge writes new commit(s) to `main`, triggering the second pipeline. A direct push to main also triggers CD. `workflow_dispatch` enables manual action runs. Neither YAML trigger, **by itself**, prevents merging a failed PR: configure required status checks and branch protections/rulesets separately if the repository plan supports them.

### Verified GitHub execution evidence

| Event | GitHub reference | Result |
|---|---|---|
| Original delivery | Commit `da2424761d816136412c028f186dba994049d3e6` | CI/CD job published first image; 12/12 tests in post-build smoke test. |
| Feature change | PR `https://github.com/chavalasantosh/MLOps/pull/1` | Added `/version` and a regression test; merged successfully. |
| Post-merge main | Commit `5405f94ab490a4fb1fb4c6b33faae5f732f411c8` | Post-merge delivery succeeded; 13/13 tests passed. |
| Release 1 image | `ghcr.io/chavalasantosh/mlops@sha256:79ebe5a13f50a96ffa5c140d01cb84c439221bd2b6eacc7bfb9ff54c7f8b1b67` | Pinned original tested image. |
| Release 2 image | `ghcr.io/chavalasantosh/mlops@sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e` | Pinned updated tested image. |

### GHCR digest vs commit tag

The **commit SHA** identifies the Git source revision; the **OCI image digest** identifies the exact published container manifest. Tags such as `sha-5405f94...` are readable names, but a digest reference is the safest way to ask Docker to fetch precisely what passed tests.

```bash
# Latest educational release; requires a compatible Docker host.
docker pull 'ghcr.io/chavalasantosh/mlops@sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e'
```

The Actions publishing step captures `docker push` output, extracts the registry-reported `sha256:...`, emits it as a job output, and prints the full reference in **Published tested image** on the workflow run summary. The GHCR Versions UI may show additional untagged platform-specific or attestation manifests; those are not automatically separate application releases.

## 40. Ubuntu deployment and recovery runbook

### 40.1 Fresh authorized Ubuntu host

```bash
# Run on your own authorized lab VM; Git is needed.
git clone https://github.com/chavalasantosh/MLOps.git
cd MLOps
git checkout 5405f94ab490a4fb1fb4c6b33faae5f732f411c8
cd diabetes-mlops-deployment
docker --version
docker compose version
python3 --version
df -h /
```

It is expected to see a **detached HEAD** message because the checkout intentionally selects a specific immutable commit rather than an active development branch. Verify Docker daemon access (`docker info`), not only client version. The example VM was root; on a normal trusted machine use an authorized Docker user instead of indiscriminate sudo.

### 40.2 Deploy without rebuilding

```bash
bash scripts/deploy.sh 'ghcr.io/chavalasantosh/mlops@sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e'
docker compose ps
curl -sS -i http://127.0.0.1:3001/readyz
curl -sS -X POST http://127.0.0.1:3001/version \
  -H 'Content-Type: application/json' -d '{}'
```

The script validates the image reference pattern; takes a `flock` lock to avoid simultaneous deployments on the same host; records the old container's image reference; pulls the new image **before** replacement; starts Compose with `--wait`; compares the running image ID; and executes API tests against loopback. On verification failure it attempts to recreate the previous image, but the old image must remain accessible and the script may not restore the previous application's complete configuration or data. A replacement is **not zero downtime**.

### 40.3 Manual educational rollback

To roll back in this lab, check out the matching **old test fixtures** and deploy the old digest. The old release expects 12 checks and has no `/version` endpoint. Then restore the new revision and new digest for normal lab operation.

```bash
cd ~/MLOps
git checkout da2424761d816136412c028f186dba994049d3e6
cd diabetes-mlops-deployment
bash scripts/deploy.sh 'ghcr.io/chavalasantosh/mlops@sha256:79ebe5a13f50a96ffa5c140d01cb84c439221bd2b6eacc7bfb9ff54c7f8b1b67'

cd ~/MLOps
git checkout 5405f94ab490a4fb1fb4c6b33faae5f732f411c8
cd diabetes-mlops-deployment
bash scripts/deploy.sh 'ghcr.io/chavalasantosh/mlops@sha256:ab3da87f94e172d3227e9012b2a59725c66bb71b63a7910780a62bd71fe9640e'
```

No automated rollback drill was evidenced in the supplied terminal logs. The successful v1.0 and v1.1.0 deployment checks **do not prove** that a failed-deployment rollback path has been exercised.

## 41. The complete monitoring stack — reproduced exactly

Monitoring was installed after the image was published. The deployment repository's `compose.yaml` launches only BentoML. Prometheus and Grafana were added with `docker run` and share the `diabetes-mlops_default` Docker network.

### 41.1 Verify BentoML metrics

```bash
curl -sS -i http://127.0.0.1:3001/metrics | head -40
```

In the observed application, metrics include:

```text
bentoml_service_request_total
bentoml_service_request_duration_seconds_sum
bentoml_service_request_duration_seconds_count
bentoml_service_request_duration_seconds_bucket
bentoml_service_request_in_progress
bentoml_service_last_request_timestamp_seconds
```

### 41.2 Configure Prometheus

```bash
cd ~/MLOps/diabetes-mlops-deployment
mkdir -p monitoring
cat > monitoring/prometheus.yml <<'EOF'
global:
  scrape_interval: 10s
scrape_configs:
  - job_name: "diabetes-api"
    metrics_path: /metrics
    static_configs:
      - targets: ["diabetes-api:3000"]
EOF
```

Docker's internal DNS resolves `diabetes-api` when Prometheus joins the same network. The target is port **3000 inside the API container**, not localhost:3001 from the host. Prometheus `scrape_interval: 10s` means it asks for a fresh metrics snapshot every ten seconds; it is **not** a guarantee that Grafana polls the application itself.

```bash
docker run -d \
  --name diabetes-prometheus \
  --network diabetes-mlops_default \
  -p 127.0.0.1:9090:9090 \
  -v "$PWD/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:v3.15.0 \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time=1d \
  --storage.tsdb.retention.size=200MB
```

Prometheus was run with a one-day/200 MB retention limit and a container filesystem backing the metrics store, **without a named volume**. The metrics history can be lost if the container is deleted/recreated. Use a dedicated volume for persistent environments.

```bash
curl -sS http://127.0.0.1:9090/-/ready
curl -sSG --data-urlencode 'query=up{job="diabetes-api"}' \
  http://127.0.0.1:9090/api/v1/query | python3 -m json.tool
```

A result of `"1"` means Prometheus **successfully scraped** the target at its last evaluation, not necessarily that the service is semantically correct or that predictions are accurate.

### 41.3 Grafana installation

```bash
docker volume create diabetes-grafana-data
umask 077
GRAFANA_ADMIN_PASS="$(openssl rand -hex 16)"
printf '%s\n' "$GRAFANA_ADMIN_PASS" > ~/grafana-lab-password.txt

docker run -d \
  --name diabetes-grafana \
  --network diabetes-mlops_default \
  -p 127.0.0.1:3002:3000 \
  -v diabetes-grafana-data:/var/lib/grafana \
  -e "GF_SECURITY_ADMIN_PASSWORD=$GRAFANA_ADMIN_PASS" \
  grafana/grafana:latest
curl -sS -i http://127.0.0.1:3002/api/health
```

`grafana/grafana:latest` was used during the lab, but **a pinned version or digest is more reproducible** for long-term use. Grafana persists its data in the named Docker volume. The initial admin password was saved with restrictive file permissions; never paste or commit it. Note that setting the initial password at first start does not necessarily change a password already stored in the Grafana DB.

### 41.4 Connect the data source

Grafana connects to Prometheus using the internal address `http://diabetes-prometheus:9090`. The user completed this using the Grafana data-source API, authenticating as `admin` with the saved password, or equivalently using the Grafana UI. Example read-only verification:

```bash
# Keep the Grafana password private.
curl -sS -u "admin:$(cat ~/grafana-lab-password.txt)" \
  http://127.0.0.1:3002/api/datasources/uid/diabetes-prometheus/health \
  | python3 -m json.tool
```

**Accessing Grafana from Windows:** The port is intentionally bound to the Ubuntu VM's `127.0.0.1:3002`. Use an approved SSH tunnel or the lab's authorized secure forwarding mechanism. Do **not** expose the admin UI directly to the public internet for convenience.

### 41.5 Dashboard panels that actually worked

| Panel | Data source query | Meaning |
|---|---|---|
| Successful Predictions | `sum(bentoml_service_request_total{endpoint="/predict",http_response_code="200"})` | Cumulative successful prediction HTTP responses since metrics startup. |
| Rejected Predictions | `sum(bentoml_service_request_total{endpoint="/predict",http_response_code="400"})` | Cumulative invalid prediction requests that returned 400. |
| API Metrics Available | `up{job="diabetes-api"}` | Whether Prometheus' last scrape succeeded. |
| Prediction Requests Over Time | `sum by (http_response_code) (bentoml_service_request_total{endpoint="/predict"})` | Cumulative request counts split by HTTP status, shown as time series. |

The screenshots demonstrated 8 successful/6 rejected requests, then 16 successful/12 rejected after rerunning `python3 test_api.py`, with `up=1`. This proves the data chain, but is generated **synthetic test traffic**, not production demand.

### 41.6 Latency: both the easy and operationally useful formulas

**Cumulative average** since the currently exposed metrics process began:

```promql
1000 *
sum(bentoml_service_request_duration_seconds_sum{endpoint="/predict",http_response_code="200"})
/
sum(bentoml_service_request_duration_seconds_count{endpoint="/predict",http_response_code="200"})
```

This divides aggregate duration by request count, then multiplies by 1000 because the metric measures **seconds** and dashboards often display **milliseconds**. The sample at one point was ~0.319568 seconds over 8 requests = ~39.95 ms average.

**Rolling five-minute average**, better for a recently slow API:

```promql
1000 *
sum(rate(bentoml_service_request_duration_seconds_sum{endpoint="/predict",http_response_code="200"}[5m]))
/
sum(rate(bentoml_service_request_duration_seconds_count{endpoint="/predict",http_response_code="200"}[5m]))
```

**Approximate 95th percentile** using histograms, subject to sufficient data:

```promql
1000 * histogram_quantile(
  0.95,
  sum by (le) (
    rate(bentoml_service_request_duration_seconds_bucket{endpoint="/predict",http_response_code="200"}[5m])
  )
)
```

With only a handful of requests, rolling rates and quantiles are unstable or temporarily empty. Averages can hide outliers, so production teams also watch p95/p99 and error rates.

### 41.7 Traffic and error rates

```promql
# Successful predictions per second, over the previous 5 minutes
sum(rate(bentoml_service_request_total{endpoint="/predict",http_response_code="200"}[5m]))

# Total prediction requests over the last 5 minutes
sum(increase(bentoml_service_request_total{endpoint="/predict"}[5m]))

# Internal server errors per second over 5 minutes
sum(rate(bentoml_service_request_total{endpoint="/predict",http_response_code=~"5.."}[5m]))
```

A rising **400** count in this lab is expected because negative regression tests deliberately send invalid age, missing fields, and incorrect categories. A sudden rise in **500** responses during genuine traffic deserves investigation. An **HTTP 405** means the caller used an unsupported request method. None of these metrics independently measures clinical model accuracy or data drift.

## 42. Operational troubleshooting table (observed issues vs potential issues)

| Symptom | Meaning | First action | Evidence status |
|---|---|---|---|
| `scripts/deploy.sh: No such file` | In the repository root rather than nested deployment folder | `cd ~/MLOps/diabetes-mlops-deployment; ls scripts` | Actually observed and resolved. |
| `docker logs` needs an argument | CLI expects a container name/ID | `docker compose logs --tail=50 diabetes-api` | Actually observed. |
| `json.tool` says `Expecting value...` after Grafana health | Empty or non-JSON response; often startup or network problem | `curl -sS -i --max-time 10 http://127.0.0.1:3002/api/health` and logs | Actually observed; health later succeeded. |
| `FallbackAsyncAdaptedQueuePool` in Colab | SQLAlchemy/MLflow environment incompatibility | Pin compatible SQLAlchemy; restart notebook runtime; rerun imports | Actually observed and resolved. |
| `predict_proba` array shape mismatch | Comparing 2-column probabilities to 1-column positive-class vector | Use `predict_proba(X)[:,1]` for positive-class probability | Actually observed and resolved. |
| Workflow missing in Actions | `.github/workflows` placed below a nested directory | Move workflow YAML to repository root `.github/workflows/` | Actually observed and resolved. |
| Windows `CRLF` breaks `.sh` scripts | Linux interprets `bash\r` or unexpected characters | `.gitattributes` with `*.sh text eol=lf`; `git add --renormalize .` | Mitigation implemented; potential failure mode. |
| `up=0` for Prometheus | Target scrape failed | Check Docker network, `diabetes-api:3000`, `/metrics`, Prometheus logs | Potential failure mode. |
| API readiness not 200 | Service startup or health problem | `docker compose ps; docker compose logs --tail=100 diabetes-api` | Potential failure mode. |
| Version endpoint not found after upgrade | Old image still running or incorrect source/image pairing | Inspect running image digest; execute matching release's tests | Potential failure mode. |
| No data in a rate/quantile panel | Too few samples in selected window | Generate authorized synthetic traffic; choose reasonable time window | Potential failure mode. |
| `docker pull` fails | Private package, wrong digest, registry rate/network issue | Confirm package visibility, image summary, authorized login | Potential failure mode. |

**Security note:** The negative input tests can cause server-side validation tracebacks to be logged even though request handling correctly returns HTTP 400. Do not mistake every `ERROR` stack trace for a service crash. Correlate logs with HTTP status, container health, and test results.

## 43. Reliability limits and production engineering backlog

This project demonstrates the lifecycle, but is **not a production-ready clinical or high-availability platform**. Critical distinctions:

1. **Not a medical diagnostic model.** Dataset size, collection bias, representativeness, demographic limitations, consent/regulatory context, and limited holdout performance prohibit clinical use.
2. **No continuously running training pipeline.** Retraining is deterministic notebook logic, not an automated data-triggered registry-to-production pipeline.
3. **No automatic target host deployment on the temporary VM.** The `deploy` job exists but is skipped because `ENABLE_LAB_AUTO_DEPLOY` is not enabled and a dedicated persistent runner is not configured.
4. **No zero-downtime rollout.** Docker Compose replaces the app container; there may be a brief outage.
5. **No exercised failure-injection rollback.** Manual known-good digest redeployment is understood, but recovery under scripted failure has not been tested in the provided evidence.
6. **No database/state migration flow for the API.** This simple classifier is effectively stateless; future stateful services require backups and schema rollback planning.
7. **No durable Prometheus metrics volume in the lab.** Grafana persists dashboards, but Prometheus history was not mounted to a named volume.
8. **No persistent real alert delivery, on-call or SLO definitions** observed. Green dashboard counters alone do not constitute incident response.
9. **No live feature drift, prediction drift, delayed-label accuracy checks, or fairness monitoring.** Request latency and count do not tell us whether predictions remain accurate.
10. **No Kubernetes cluster, service mesh, horizontal autoscaling or multizone deployment** in this phase.
11. **Partially pinned environment.** Direct dependencies are pinned; transitive library and base-image supply-chain risks require more work. `grafana/grafana:latest` is mutable. GitHub's old pinned checkout SHA emitted a Node.js deprecation warning, despite pipeline success.
12. **Supply chain hardening is incomplete.** Add controlled base-image updates, SBOM, vulnerability scanning, trusted build attestation and tested artifact provenance.
13. **Public sample content requires review.** The stored trained model comes from public UCI data, but distribution permissions, licensing, privacy and use restrictions still require review before reusing outside a lab.

### Suggested future milestones (not falsely marked complete)

- Use stable, persistent lab infrastructure and persistent monitoring storage.
- Add Grafana alert rules plus a verified notification channel; test an intentional alert.
- Define request-rate, error-rate, p95 latency and availability SLOs.
- Monitor model inputs, feature distribution, drift and delayed ground-truth performance.
- Add load tests, resource profiling, security scans and role-based access.
- Learn Kubernetes deployments, Services, probes, ConfigMaps, Secrets and HPA.
- Practice failing deployments with controlled rollback, without using real medical data.
- Automate reproducible monitoring via version-controlled Compose configuration or Helm charts.

## 44. Interview-ready explanation of this exact project

A useful two-minute summary:

> I built an end-to-end educational MLOps pipeline around the UCI early-stage diabetes dataset. I used Google Colab and Python to preserve data lineage, clean and validate inputs, prevent duplicate-profile leakage, engineer features and compare five sklearn algorithms using grouped cross-validation. I used MLflow for baseline and tuned experiment runs, audits, and model registration. The selected SVM passed a predeclared educational holdout gate, and I exported the exact registered artifact to BentoML. I built a typed inference service with strict Pydantic input validation, synthetic golden test fixtures, a checksum verifier and 13 regression test groups. GitHub Actions validates and tests pull requests; after a merge to main it builds and tests a Docker image, publishes it to GHCR by immutable digest, and an Ubuntu deployment script pulls and verifies that image. I then added Prometheus metrics scraping and Grafana visualizations, verified that request counters changed from 8 to 16 after a controlled test run, and began learning latency metrics. It is a lab demonstration rather than a medical device or fully automated high-availability production system.

**Questions you should be able to answer:** Why freeze the holdout? Why use group-aware splitting? Why not retrain on every Git push? Why verify the exact Docker digest? What does `/readyz` verify versus `/metrics`? Why can HTTP 400 logs include Pydantic errors? Why does Prometheus `up=1` not mean the model is clinically accurate? Why use a rolling latency average instead of a lifetime average?

## 45. Source links and evidence provenance

The current source code is at `https://github.com/chavalasantosh/MLOps`. For a version-locked reference use the merge commit `5405f94ab490a4fb1fb4c6b33faae5f732f411c8` in URLs, not `main`, which can change. Primary files:

- Notebook: `https://github.com/chavalasantosh/MLOps/blob/5405f94ab490a4fb1fb4c6b33faae5f732f411c8/Diabetes_MLOps.ipynb`
- BentoML service: `https://github.com/chavalasantosh/MLOps/blob/5405f94ab490a4fb1fb4c6b33faae5f732f411c8/diabetes-mlops-deployment/service.py`
- Pinned release evidence: `model_manifest.json`, `evaluation_summary.json`, `api_fixtures.json` in the same folder.
- API tests: `diabetes-mlops-deployment/test_api.py`
- Deployment scripts: `diabetes-mlops-deployment/scripts/smoke.sh` and `scripts/deploy.sh`
- CI: `.github/workflows/ci.yml`; CD: `.github/workflows/cd.yml` at the repository root.
- Source PR: `https://github.com/chavalasantosh/MLOps/pull/1`
- First release: `https://github.com/chavalasantosh/MLOps/actions/runs/44886156011`
- Post-merge release: `https://github.com/chavalasantosh/MLOps/actions/runs/44887546219`
- UCI source: `https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset`

**Scope note:** The monitoring section is based on the terminal output and Grafana screenshots supplied during the conversation. It is not a statement that those files are already part of the tracked GitHub repository. The training notebook source in the following annex is preserved from the attached notebook; code cells should be run in their Colab environment and order, not pasted randomly into Ubuntu.

---

# Part VIII — Guided notebook source annex: all notebook cells

This annex reconstructs the **original sequential learning notebook** without truncating its executable Python. It retains its tutorial Markdown and runnable code blocks. Notebook output images, interactive Colab widgets and transient rich displays are not reproduced here, because they depend on execution and environment. For those, use the original `.ipynb` linked in Chapter 45. The attached notebook is the authoritative executable artifact; this annex is a reading and source-inspection reference.



<!-- Notebook original cell 000: markdown -->

## Diabetes MLOps — Guided Master Lab (step-by-step edition)
**Google Colab → Google Drive → MLflow → BentoML → Docker → GitHub Actions → GHCR → Ubuntu**

This is the expanded learning edition of the previous notebook. Keep it open alongside Colab. Each small code cell has a purpose and expected result. Run **one cell at a time** with Shift+Enter. Read the output before moving on; do not use Run all on your first attempt. All five model families and all deployment files remain included.

**Where commands run:** Python cells run in Colab. Markdown terminal blocks marked **Ubuntu** run in a trusted Ubuntu shell, and **GitHub** steps run in a web browser. No credentials in notebook cells. If a cell fails, stop and inspect the error rather than skipping assertions. A fresh Colab session creates a fresh run ID. Mount Drive again after a reset. The notebook does not claim any external deployment has already passed.

**Scope:** educational model; no diagnosis, no real patient input. CI, image publishing and deployment only. No monitoring, Prometheus, Grafana or Kubernetes yet.

**Learning objective:** understand each intermediate data table, each logged model, every Docker layer and every workflow job; then reproduce the deployment on a clean VM without manual rebuilding.


<!-- Notebook original cell 001: markdown -->

### Contents

- 01 · Install modules
- 02 · Drive + run configuration
- 03 · Data collection and immutable raw storage
- 04 · Data engineering — schema, cleaning and integrity
- 05 · Feature engineering and structural EDA
- 06 · Model initialization, data split and split analysis
- 07 · Development EDA and statistical analysis
- 08 · Baseline model development and metrics
- 09 · Advanced development analysis (out-of-fold)
- 10 · MLflow initialization and five baseline experiment runs
- 11 · Load saved models, independently recompute metrics and compare with MLflow
- 12 · Adaptive retraining — analysis-driven, deterministic policy
- 13 · Consistency, model comparison and freeze final candidate
- 14 · Final evaluation — one frozen candidate, one holdout
- 15 · MLflow Registry and alias
- 16 · BentoML init and registry model import
- 17 · Generate a typed BentoML service in Colab
- 18 · Create deterministic API fixtures and automated tests
- 19 · Start the BentoML server in Colab, send requests and analyze responses
- 20 · Preserve the model and generate serving requirements
- 21 · Docker — layers, image, container and Compose
- 22 · Optional actual Docker build/run in Colab
- 23 · Continuous integration and continuous delivery
- 24 · Deployment mode selection: ephemeral THM VM vs persistent host
- 25 · Final package, integrity checks and backup
- 26 · GitHub upload and CI/CD execution (external steps)
- 27 · Fresh VM deployment (no manual Docker build)
- 28 · Completion checklist and release decision
- References and boundaries


<!-- Notebook original cell 002: markdown -->

### 01 · Install modules
These are deliberately selected, compatible-generation lab versions, not a claim that they are the latest. The serving requirements are generated from the actually imported environment. If Colab reports a previously loaded package changed, restart the session and run from the top. The installer is idempotent for the pinned versions. Use Python 3.10–3.13; do not silently change sklearn between training and serving.


<!-- Notebook original cell 003: markdown -->

#### Step 001 — Check Python and select packages

**What and why:** sys.executable points to the actual notebook interpreter. Pin MLflow, BentoML and sklearn; the rest have compatible ranges.

**Checkpoint:** Python is 3.10–3.13; an assertion means stop and use a supported runtime.


**Executable Python notebook cell 004**

```python
import sys, subprocess
from importlib import metadata
assert (3, 10) <= sys.version_info[:2] <= (3, 13), sys.version
packages = ["mlflow==3.5.1", "bentoml==1.4.27", "scikit-learn==1.6.1",
            "pandas>=2.2,<3", "numpy>=1.26,<3", "scipy>=1.12,<2",
            "matplotlib>=3.8", "seaborn>=0.13", "pydantic>=2.7,<3", "requests>=2.31"]
```


<!-- Notebook original cell 005: markdown -->

#### Step 002 — Install modules

**What and why:** Run pip through the same interpreter, rather than an unrelated terminal Python.

**Checkpoint:** Installation complete; if Colab reports already imported modules changed, restart once.


**Executable Python notebook cell 006**

```python
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
print("Installation complete. Restart the runtime if Colab reports loaded packages changed.")
# Inspect, rather than assuming all unrelated preinstalled Colab packages agree.
```


<!-- Notebook original cell 007: markdown -->

#### Step 003 — Inspect dependency health

**What and why:** pip check detects inconsistent installed requirements, including unrelated Colab packages.

**Checkpoint:** Review conflicts; do not proceed with a conflict affecting the model or serving libraries.


**Executable Python notebook cell 008**

```python
check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
print(check.stdout[:3000], check.stderr[:1000])
```


<!-- Notebook original cell 009: markdown -->

### 02 · Drive + run configuration
Raw is content-addressed and never rewritten. Processed copies, charts, model artifacts and audit reports are versioned by a run ID. MLflow SQLite lives on local disk (not mounted Drive); use the backup function to create a consistent snapshot on Drive. Artifacts are written to a persistent directory. This is a single-user lab, not a multi-user tracking server.


<!-- Notebook original cell 010: markdown -->

#### Step 004 — Import filesystem tools and mount Drive

**What and why:** Drive is persistent; /content is temporary. Path objects avoid hard-coded shell paths.

**Checkpoint:** Choose your Google account and grant access only to your own Drive.


**Executable Python notebook cell 011**

```python
from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, shutil, sqlite3, uuid, platform, time, zipfile, io, re
from collections import Counter
from IPython.display import display

drive.mount('/content/drive')
```


<!-- Notebook original cell 012: markdown -->

#### Step 005 — Create raw, run, and deployment directories

**What and why:** Each execution gets a separate ID. RAW is shared and content-addressed; all processed artifacts are per-run.

**Checkpoint:** The directories exist; run this once per experiment.


**Executable Python notebook cell 013**

```python
ROOT = Path('/content/drive/MyDrive/Diabetes_MLOps_Master')
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
RUN = ROOT / 'runs' / RUN_ID
RAW = ROOT / 'raw'
EXTRACTED = RUN / 'extracted'
PROCESSED = RUN / 'processed'
REPORTS = RUN / 'reports'
ARTIFACTS = RUN / 'mlflow_artifacts'
LOCAL = Path('/content/diabetes_mlops') / RUN_ID
DEPLOY = LOCAL / 'deployment'
for p in [RAW, EXTRACTED, PROCESSED, REPORTS, ARTIFACTS, LOCAL, DEPLOY]:
    p.mkdir(parents=True, exist_ok=True)
```


<!-- Notebook original cell 014: markdown -->

#### Step 006 — Set reproducibility and quality requirements

**What and why:** Set random seed and educational validation gates before looking at test performance. The Bento store is isolated.

**Checkpoint:** Do not relax the thresholds after viewing holdout results.


**Executable Python notebook cell 015**

```python
SEED = 42
N_JOBS = 2
MIN_F1, MIN_RECALL, MIN_BALANCED_ACCURACY = .75, .75, .70  # educational gates, set BEFORE holdout
os.environ['BENTOML_HOME'] = str(LOCAL / 'bentoml')
os.environ['BENTOML_DO_NOT_TRACK'] = 'True'
os.environ['GIT_TERMINAL_PROMPT'] = '0'
```


<!-- Notebook original cell 016: markdown -->

#### Step 007 — Define reusable storage helpers

**What and why:** SHA-256 detects changes; write_json preserves structured metadata; immutable_write refuses to overwrite different bytes.

**Checkpoint:** Printed run paths identify exactly where your work will be saved.


**Executable Python notebook cell 017**

```python
def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

def immutable_write(path, data):
    path = Path(path)
    if path.exists():
        assert path.read_bytes() == data, f'Existing raw file differs: {path}'
    else:
        with path.open('xb') as f:
            f.write(data)

print('Run ID:', RUN_ID, '\nDrive:', RUN, '\nLocal deployment:', DEPLOY)
```


<!-- Notebook original cell 018: markdown -->

#### Step 008 — Import analysis and machine learning modules

**What and why:** Pandas handles tables, NumPy arrays, matplotlib/seaborn plots, and sklearn provides pipelines and models.

**Checkpoint:** Imports work and sklearn reports 1.6.1.


**Executable Python notebook cell 019**

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import sklearn, requests
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix,
    classification_report, make_scorer, roc_curve, precision_recall_curve)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from importlib import metadata
assert sklearn.__version__ == '1.6.1', 'Restart runtime before continuing.'
np.random.seed(SEED)
sns.set_theme(style='whitegrid')
```


<!-- Notebook original cell 020: markdown -->

#### Step 009 — Capture actual training and serving versions

**What and why:** An environment report is important when moving serialized estimators from Colab to Docker.

**Checkpoint:** Inspect VERSIONS and environment.json; pip freeze is private.


**Executable Python notebook cell 021**

```python
CORE = ['mlflow','bentoml','scikit-learn','numpy','pandas','scipy','pydantic','joblib','threadpoolctl','cloudpickle']
VERSIONS = {p: metadata.version(p) for p in CORE}
MODEL_PINS = [f'{p}=={VERSIONS[p]}' for p in CORE if p not in {'bentoml','pydantic'}]
SERVING_PINS = [f'{p}=={v}' for p,v in VERSIONS.items()]
write_json(REPORTS / 'environment.json', {'python':sys.version,'platform':platform.platform(),'packages':VERSIONS})
( REPORTS / 'training-pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
print(VERSIONS)
```


<!-- Notebook original cell 022: markdown -->

#### Step 010 — Create a chart-saving helper

**What and why:** All charts are displayed in Colab and saved in the persistent reports directory.

**Checkpoint:** Function definition completes without output.


**Executable Python notebook cell 023**

```python
def save_plot(name):
    plt.tight_layout()
    plt.savefig(REPORTS / f'{name}.png', dpi=140, bbox_inches='tight')
    plt.show()
    plt.close()
```


<!-- Notebook original cell 024: markdown -->

### 03 · Data collection and immutable raw storage
Source: [UCI Early Stage Diabetes Risk Prediction, ID 529](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset). The archive is stored exactly as downloaded with a SHA-256 identity. The checksum detects local mutation; it is not an independently published authenticity signature. Read a named CSV member instead of extracting arbitrary archive paths. Do not commit raw records to GitHub.


<!-- Notebook original cell 025: markdown -->

#### Step 011 — Download public UCI dataset

**What and why:** requests retrieves the ZIP over HTTPS and raises an error for unsuccessful responses. A size limit rejects unexpectedly large data.

**Checkpoint:** HTTP request succeeds; if it fails, check connectivity rather than using unrelated data.


**Executable Python notebook cell 026**

```python
DATA_URL = 'https://archive.ics.uci.edu/static/public/529/early+stage+diabetes+risk+prediction+dataset.zip'
response = requests.get(DATA_URL, timeout=(15, 90))
response.raise_for_status()
blob = response.content
assert len(blob) < 10_000_000, 'Unexpected archive size'
```


<!-- Notebook original cell 027: markdown -->

#### Step 012 — Preserve original ZIP with a content hash

**What and why:** Identical downloads reuse a raw directory. A changed archive receives a new hash.

**Checkpoint:** Original bytes are stored under raw/<SHA256>/uci_529.zip.


**Executable Python notebook cell 028**

```python
raw_hash = hashlib.sha256(blob).hexdigest()
raw_dir = RAW / raw_hash
raw_dir.mkdir(exist_ok=True)
raw_zip = raw_dir / 'uci_529.zip'
immutable_write(raw_zip, blob)
```


<!-- Notebook original cell 029: markdown -->

#### Step 013 — Read the named CSV and preserve the original

**What and why:** Select the expected member rather than extracting arbitrary paths. Save original CSV separately.

**Checkpoint:** One expected CSV member is found.


**Executable Python notebook cell 030**

```python
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    members = [m for m in z.infolist() if Path(m.filename).name.lower() == 'diabetes_data_upload.csv']
    assert len(members) == 1 and members[0].file_size < 10_000_000
    csv_bytes = z.read(members[0])
raw_csv = raw_dir / 'diabetes_data_upload.csv'
immutable_write(raw_csv, csv_bytes)
```


<!-- Notebook original cell 031: markdown -->

#### Step 014 — Create an independent working copy and lineage record

**What and why:** The extraction copy is used for analysis. The recorded source, retrieval date and hashes tie later results to the source.

**Checkpoint:** The two CSV hashes match.


**Executable Python notebook cell 032**

```python
extract_csv = EXTRACTED / raw_csv.name
shutil.copy2(raw_csv, extract_csv)
assert sha256_file(raw_csv) == sha256_file(extract_csv)
manifest = {'source':DATA_URL,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
            'archive_sha256':raw_hash,'csv_sha256':sha256_file(raw_csv), 'raw_path':str(raw_csv)}
write_json(RUN / 'data_manifest.json', manifest)
```


<!-- Notebook original cell 033: markdown -->

#### Step 015 — Load the copied data into pandas

**What and why:** Load all columns as strings first to inspect source values without premature coercion.

**Checkpoint:** Typically 520 rows and 17 columns; inspect the displayed head.


**Executable Python notebook cell 034**

```python
raw_df = pd.read_csv(extract_csv, dtype='string')
print('Raw shape:', raw_df.shape, 'SHA256:', manifest['csv_sha256'])
display(raw_df.head())
```


<!-- Notebook original cell 035: markdown -->

#### Step 016 — Inspect the working data before cleaning

**What and why:** Use info, head, missing counts and exact-duplicate counts to understand the starting shape; the raw copy is read-only by convention.

**Checkpoint:** 17 columns, source categories and any missing/duplicate counts are shown.


**Executable Python notebook cell 036**

```python
print('Working file:', extract_csv)
print('Shape:', raw_df.shape)
raw_df.info()
display(raw_df.head(5))
display(pd.DataFrame({'missing':raw_df.isna().sum(),'distinct':raw_df.nunique()}))
print('Exact source duplicates:', raw_df.duplicated().sum())
```


<!-- Notebook original cell 037: markdown -->

### 04 · Data engineering — schema, cleaning and integrity
The raw file stays untouched. Normalize column names, whitespace, categorical values and target. Invalid rows are quarantined (not guessed). Exact duplicate feature+label rows are removed. Identical features with conflicting labels are reported, kept in the same split group, and never allowed across train and test. This is a feature-profile group, not a verified patient ID; true patient-level leakage cannot be ruled out.


<!-- Notebook original cell 038: markdown -->

#### Step 017 — Declare schema and normalize column names

**What and why:** Define the 16 expected fields, copy the raw DataFrame, and enforce the schema.

**Checkpoint:** All fields present; no missing or unexpected columns.


**Executable Python notebook cell 039**

```python
FEATURES = ['age','gender','polyuria','polydipsia','sudden_weight_loss','weakness','polyphagia',
 'genital_thrush','visual_blurring','itching','irritability','delayed_healing','partial_paresis',
 'muscle_stiffness','alopecia','obesity']
SYMPTOMS = FEATURES[2:]
TARGET = 'class'
df = raw_df.copy(deep=True)
df.columns = [re.sub(r'[^a-z0-9]+','_',str(c).strip().lower()).strip('_') for c in df.columns]
assert df.columns.is_unique and set(df.columns) == set(FEATURES+[TARGET]), df.columns.tolist()
df = df[FEATURES+[TARGET]].copy()
```


<!-- Notebook original cell 040: markdown -->

#### Step 018 — Normalize values and data types

**What and why:** Strip whitespace, map blanks and ? to missing, coerce age, and standardize categorical spelling.

**Checkpoint:** Age becomes numeric; categorical values use canonical case.


**Executable Python notebook cell 041**

```python
for c in df.columns:
    df[c] = df[c].str.strip().replace({'?':pd.NA,'':pd.NA})
df['age'] = pd.to_numeric(df['age'], errors='coerce').astype(float)
for c in FEATURES[1:]+[TARGET]:
    df[c] = df[c].str.lower().str.capitalize()
```


<!-- Notebook original cell 042: markdown -->

#### Step 019 — Validate values and quarantine bad rows

**What and why:** Check age, gender, label and symptom domains. Invalid records are saved privately; no fabricated labels or clinical values.

**Checkpoint:** Quarantine file exists; inspect the count in the audit below.


**Executable Python notebook cell 043**

```python
valid = df['age'].between(1,120) & df['gender'].isin(['Male','Female']) & df[TARGET].isin(['Positive','Negative'])
for c in SYMPTOMS:
    valid &= df[c].isin(['Yes','No'])
invalid = df.loc[~valid].copy()
invalid.to_csv(PROCESSED/'quarantine.csv',index=False)
clean = df.loc[valid].copy()
```


<!-- Notebook original cell 044: markdown -->

#### Step 020 — Deduplicate and encode the target

**What and why:** Identical feature and label records are removed. Negative becomes 0 and Positive becomes 1.

**Checkpoint:** Both classes must remain.


**Executable Python notebook cell 045**

```python
exact_duplicates = int(clean.duplicated().sum())
clean = clean.drop_duplicates().reset_index(drop=True)
for c in FEATURES[1:]: clean[c] = clean[c].astype(str)
clean[TARGET] = clean[TARGET].map({'Negative':0,'Positive':1}).astype(int)
# Stable profile hash; independent of target, and portable across sessions.
```


<!-- Notebook original cell 046: markdown -->

#### Step 021 — Identify potentially leaking duplicate profiles

**What and why:** Hash only the feature values so conflicting labels on identical profiles stay together during splits.

**Checkpoint:** The hash is a profile identifier, not proof of patient identity.


**Executable Python notebook cell 047**

```python
clean['group'] = [hashlib.sha256(json.dumps(list(row),separators=(',',':')).encode()).hexdigest()
                  for row in clean[FEATURES].itertuples(index=False,name=None)]
conflicts = clean.groupby('group')[TARGET].nunique().gt(1)
```


<!-- Notebook original cell 048: markdown -->

#### Step 022 — Save cleaned dataset and data quality report

**What and why:** Record raw, rejected, duplicate and retained row counts and verify original file integrity.

**Checkpoint:** cleaned.csv and cleaning_audit.json are saved; hashes still match.


**Executable Python notebook cell 049**

```python
audit = {'raw_rows':len(raw_df),'quarantined':len(invalid),'exact_duplicates_removed':exact_duplicates,
         'clean_rows':len(clean),'conflicting_profiles':int(conflicts.sum()),'unique_profiles':int(clean.group.nunique())}
assert len(clean) > 100 and clean[TARGET].nunique()==2
clean.to_csv(PROCESSED/'cleaned.csv',index=False)
write_json(REPORTS/'cleaning_audit.json',audit)
assert sha256_file(raw_zip)==raw_hash and sha256_file(raw_csv)==manifest['csv_sha256']
display(pd.Series(audit,name='count'))
display(clean[FEATURES].describe(include='all').T)
```


<!-- Notebook original cell 050: markdown -->

#### Step 023 — Confirm the cleaning accounting

**What and why:** Every removed row must be explained; inspect the cleaned schema and private quarantined records if needed.

**Checkpoint:** Accounting identity is True; cleaned DataFrame contains group plus feature and label columns.


**Executable Python notebook cell 051**

```python
assert audit['raw_rows'] == audit['quarantined'] + audit['exact_duplicates_removed'] + audit['clean_rows']
print('Row accounting verified')
display(clean.head(3))
print('Clean types:'); print(clean.dtypes)
```


<!-- Notebook original cell 052: markdown -->

### 05 · Feature engineering and structural EDA
We preserve the 16-field API contract. Two derived features are calculated inside the sklearn pipeline: symptom burden (Yes count) and age buckets. Scaling and encoding are fitted separately in each training fold; no preprocessing is fitted on holdout. Structural checks can use the complete file; target-conditioned EDA below uses development data only, after the split.


<!-- Notebook original cell 053: markdown -->

#### Step 024 — Profile data structure before fitting

**What and why:** Check dtypes, unique counts and missingness; the structural age histogram is for data validation, not model selection.

**Checkpoint:** No unexpected nulls after quarantine; schema report is saved.


**Executable Python notebook cell 054**

```python
# Structural analysis only; target labels are not plotted before splitting.
structure = pd.DataFrame({'dtype':clean[FEATURES].dtypes.astype(str),
                          'missing':clean[FEATURES].isna().sum(), 'unique':clean[FEATURES].nunique()})
display(structure)
structure.to_csv(REPORTS/'schema_profile.csv')
plt.figure(figsize=(9,4)); plt.hist(clean['age'],bins=15); plt.title('Structural age distribution'); save_plot('age_structure')
```


<!-- Notebook original cell 055: markdown -->

#### Step 025 — Define feature engineering inside a pipeline

**What and why:** Scale age, one-hot encode categories, count Yes symptoms, and derive age buckets. Learned parameters are fit inside each training fold.

**Checkpoint:** Function is defined; no transformer has seen the holdout yet.


**Executable Python notebook cell 056**

```python
def make_preprocessor():
    numeric = Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())])
    categoricals = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    symptom_count = Pipeline([('encode',OrdinalEncoder(categories=[['No','Yes']]*len(SYMPTOMS),
                           handle_unknown='use_encoded_value',unknown_value=-1)),
                           ('count',FunctionTransformer(np.sum,kw_args={'axis':1,'keepdims':True})),
                           ('scale',StandardScaler())])
    age_bins = KBinsDiscretizer(n_bins=4,encode='onehot-dense',strategy='uniform')
    return ColumnTransformer([('age',numeric,['age']),('categories',categoricals,FEATURES[1:]),
                              ('symptom_burden',symptom_count,SYMPTOMS),('age_buckets',age_bins,['age'])])
```


<!-- Notebook original cell 057: markdown -->

### 06 · Model initialization, data split and split analysis
Five families: logistic regression, random forest, gradient boosting, support vector machine and K-nearest neighbors. Reserve one fold as an approximately 20% holdout, then build 5-fold stratified group CV on the remaining development records. Every model gets the same CV splits. The holdout is not used to tune, compare or pick a model. Do not rerun selection after seeing holdout outcomes; a failed gate requires fresh independent evaluation data.


<!-- Notebook original cell 058: markdown -->

#### Step 026 — Initialize five candidate estimators

**What and why:** Start with logistic regression, random forest, gradient boosting, SVM and KNN. clone prevents reusing fitted state.

**Checkpoint:** Five unfitted model templates are available.


**Executable Python notebook cell 059**

```python
MODELS = {
 'LogisticRegression':LogisticRegression(max_iter=1500,random_state=SEED),
 'RandomForest':RandomForestClassifier(n_estimators=160,random_state=SEED,n_jobs=1),
 'GradientBoosting':GradientBoostingClassifier(random_state=SEED),
 'SVM':SVC(probability=True,random_state=SEED),
 'KNN':KNeighborsClassifier(n_neighbors=7),
}
def make_pipeline(estimator):
    return Pipeline([('features',make_preprocessor()),('model',clone(estimator))])
```


<!-- Notebook original cell 060: markdown -->

#### Step 027 — Separate features, labels and groups

**What and why:** X holds only serving inputs, y is the target, and groups are used only to keep duplicates together.

**Checkpoint:** No target or group hash appears in model inputs.


**Executable Python notebook cell 061**

```python
X = clean[FEATURES].copy(); y = clean[TARGET].copy(); groups = clean['group'].copy()
```


<!-- Notebook original cell 062: markdown -->

#### Step 028 — Create a protected holdout

**What and why:** One fold of a stratified group split is held out; the other four become the development set.

**Checkpoint:** Both subsets contain both classes; zero shared group hashes.


**Executable Python notebook cell 063**

```python
outer = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
dev_idx,test_idx = next(outer.split(X,y,groups))
X_dev,y_dev,g_dev = [a.iloc[dev_idx].reset_index(drop=True) for a in (X,y,groups)]
X_test,y_test,g_test = [a.iloc[test_idx].reset_index(drop=True) for a in (X,y,groups)]
assert not set(g_dev)&set(g_test)
assert set(y_dev)=={0,1} and set(y_test)=={0,1}
```


<!-- Notebook original cell 064: markdown -->

#### Step 029 — Make five repeatable development folds

**What and why:** Every candidate uses the exact same grouped validation folds.

**Checkpoint:** No fold shares a feature group across train and validation.


**Executable Python notebook cell 065**

```python
cv = list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED+1).split(X_dev,y_dev,g_dev))
for tr,va in cv:
    assert not set(g_dev.iloc[tr])&set(g_dev.iloc[va])
    assert set(y_dev.iloc[tr])==set(y_dev.iloc[va])=={0,1}
```


<!-- Notebook original cell 066: markdown -->

#### Step 030 — Analyze and persist split membership

**What and why:** Compare sample sizes and class proportions, and save split indices for experiment lineage.

**Checkpoint:** Approximately 80/20 split; verify exact counts shown, not assumed.


**Executable Python notebook cell 067**

```python
split_report = pd.DataFrame([{'split':'development','rows':len(y_dev),'positive_rate':y_dev.mean()},
                            {'split':'holdout','rows':len(y_test),'positive_rate':y_test.mean()}])
display(split_report)
# Persist split membership and folds for reproducibility, only in Drive.
clean.assign(split=np.where(np.isin(np.arange(len(clean)),dev_idx),'development','holdout')).to_csv(PROCESSED/'split_membership.csv',index=False)
write_json(PROCESSED/'split_indices.json',{'dev':dev_idx.tolist(),'holdout':test_idx.tolist(),
           'cv':[{'train':a.tolist(),'validation':b.tolist()} for a,b in cv]})
split_report.to_csv(REPORTS/'split_summary.csv',index=False)
```


<!-- Notebook original cell 068: markdown -->

#### Step 031 — Inspect feature engineering on one training fold

**What and why:** Fit a disposable transformer on the first CV training fold, then transform a few validation rows. This is a demonstration only; each actual model refits its own transformer.

**Checkpoint:** Feature matrix has more columns than raw inputs, and validation data is transformed without fitting.


**Executable Python notebook cell 069**

```python
tr_demo, va_demo = cv[0]
pre_demo = make_preprocessor()
pre_demo.fit(X_dev.iloc[tr_demo])
engineered_demo = pre_demo.transform(X_dev.iloc[va_demo[:5]])
print('Input shape:', X_dev.iloc[va_demo[:5]].shape)
print('Output shape:', engineered_demo.shape)
display(pd.DataFrame(engineered_demo).round(3))
display(pd.DataFrame({'transformer':[name for name, _, _ in pre_demo.transformers_],
                      'output_columns':[str(pre_demo.output_indices_[name]) for name, _, _ in pre_demo.transformers_]}))
```


<!-- Notebook original cell 070: markdown -->

### 07 · Development EDA and statistical analysis
Look at class balance, missingness, age/gender distribution, symptom prevalence, associations and engineered burden. Associations are descriptive, not causal. Avoid interpreting a small subgroup as a clinical result. Raw symptoms are retained for the service; the engineered features are produced automatically at inference.


<!-- Notebook original cell 071: markdown -->

#### Step 032 — Explore class, age, gender and symptom burden

**What and why:** Use only development records for target-aware EDA. Study the distributions before model evaluation.

**Checkpoint:** Four plots appear; look for class imbalance and overlapping distributions.


**Executable Python notebook cell 072**

```python
eda = X_dev.copy(); eda['target']=y_dev
eda['symptom_burden']=eda[SYMPTOMS].eq('Yes').sum(axis=1)
eda['age_band']=pd.cut(eda.age,[0,30,45,60,120],include_lowest=True)
fig,axes=plt.subplots(2,2,figsize=(12,8))
sns.countplot(data=eda,x='target',ax=axes[0,0]); axes[0,0].set_title('Development class balance')
axes[0,1].hist([eda.loc[eda.target==c,'age'] for c in [0,1]],bins=15,stacked=True,label=['Negative','Positive']); axes[0,1].legend(); axes[0,1].set_title('Age by class')
sns.countplot(data=eda,x='gender',hue='target',ax=axes[1,0]); axes[1,0].set_title('Gender by class')
sns.boxplot(data=eda,x='target',y='symptom_burden',ax=axes[1,1]); axes[1,1].set_title('Symptom burden')
save_plot('development_eda')
```


<!-- Notebook original cell 073: markdown -->

#### Step 033 — Measure symptom associations by class

**What and why:** Compute Yes proportion separately for positive and negative records and sort by difference.

**Checkpoint:** Heatmap and CSV show which symptoms are associated with the label; not causal effects.


**Executable Python notebook cell 074**

```python
prevalence = pd.DataFrame({c:eda.groupby('target')[c].apply(lambda x:x.eq('Yes').mean()) for c in SYMPTOMS}).T
prevalence.columns=['negative','positive']; prevalence['difference']=prevalence.positive-prevalence.negative
prevalence=prevalence.sort_values('difference',ascending=False)
display(prevalence)
prevalence.to_csv(REPORTS/'symptom_prevalence.csv')
plt.figure(figsize=(8,7)); sns.heatmap(prevalence[['negative','positive']],annot=True,fmt='.2f',cmap='Blues'); save_plot('symptom_prevalence')
```


<!-- Notebook original cell 075: markdown -->

#### Step 034 — Explore feature correlations

**What and why:** Spearman rank association on binary encodings and age helps identify redundant or related variables.

**Checkpoint:** Interpret sign and magnitude without treating correlation as causation.


**Executable Python notebook cell 076**

```python
encoded = pd.concat([eda[['age','target','symptom_burden']],eda[SYMPTOMS].eq('Yes').astype(int)],axis=1)
plt.figure(figsize=(12,10)); sns.heatmap(encoded.corr(method='spearman'),cmap='coolwarm',center=0); save_plot('spearman_associations')
```


<!-- Notebook original cell 077: markdown -->

#### Step 035 — Compare development subgroups

**What and why:** Age band and gender summaries show counts and observed positive proportions.

**Checkpoint:** Small subgroups are uncertain; do not infer demographic risk in the population.


**Executable Python notebook cell 078**

```python
groups_eda=eda.groupby(['age_band','gender'],observed=True).target.agg(['size','mean']).rename(columns={'mean':'positive_rate'})
display(groups_eda); groups_eda.to_csv(REPORTS/'development_subgroups.csv')
```


<!-- Notebook original cell 079: markdown -->

#### Step 036 — Run a more detailed development-data analysis

**What and why:** Compare distributions by class and counts of Yes answers; compute symptom co-occurrence. This is observational EDA, not a clinical risk calculator.

**Checkpoint:** Tables and charts reveal associations and sparse feature combinations.


**Executable Python notebook cell 080**

```python
display(eda.groupby('target').age.describe())
display(pd.crosstab(eda.age_band, eda.target, margins=True))
counts = eda[SYMPTOMS].eq('Yes').sum().sort_values()
plt.figure(figsize=(8,6)); counts.plot.barh(); plt.xlabel('Yes count in development set'); save_plot('symptom_counts')
flags = eda[SYMPTOMS].eq('Yes').astype(int)
cooccurrence = flags.T @ flags
plt.figure(figsize=(10,8)); sns.heatmap(cooccurrence, cmap='Blues'); plt.title('Symptom co-occurrence'); save_plot('symptom_cooccurrence')
```


<!-- Notebook original cell 081: markdown -->

### 08 · Baseline model development and metrics
Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and average precision. Primary rank is mean validation F1, then recall, then lower variability. Recall is important here because false negatives are relevant in a screening-style exercise. The default decision rule is each estimator's `predict`; probabilities are diagnostic only, not clinically calibrated risks. Train scores diagnose potential overfit; validation scores drive decisions.


<!-- Notebook original cell 082: markdown -->

#### Step 037 — Define metrics and reusable CV evaluator

**What and why:** Accuracy measures overall correctness; balanced accuracy averages class recalls; F1 balances precision and recall. Save each fold and train-validation gap.

**Checkpoint:** Functions defined; training has not started in this cell.


**Executable Python notebook cell 083**

```python
SCORING = {'accuracy':'accuracy','balanced_accuracy':'balanced_accuracy',
           'precision':make_scorer(precision_score,zero_division=0),
           'recall':make_scorer(recall_score,zero_division=0),
           'f1':make_scorer(f1_score,zero_division=0),'roc_auc':'roc_auc','average_precision':'average_precision'}

def cv_summary(name, pipe, splits=cv):
    res = cross_validate(pipe,X_dev,y_dev,cv=splits,scoring=SCORING,return_train_score=True,
                         n_jobs=N_JOBS,error_score='raise')
    out={'model':name}
    for key in SCORING:
        out[f'cv_{key}_mean']=float(np.mean(res[f'test_{key}']))
        out[f'cv_{key}_std']=float(np.std(res[f'test_{key}'],ddof=1))
        out[f'train_{key}_mean']=float(np.mean(res[f'train_{key}']))
    out['f1_gap']=out['train_f1_mean']-out['cv_f1_mean']
    out['fit_seconds_mean']=float(np.mean(res['fit_time']))
    folds = pd.DataFrame({k:v for k,v in res.items()}); folds.insert(0,'fold',range(1,len(folds)+1))
    return out,folds

def rank(rows):
    return pd.DataFrame(rows).sort_values(['cv_f1_mean','cv_recall_mean','cv_f1_std','model'],
                                          ascending=[False,False,True,True]).reset_index(drop=True)
```


<!-- Notebook original cell 084: markdown -->

#### Step 038 — Train and evaluate all five models

**What and why:** For every estimator, fit and validate on five folds, then fit a development-only copy for saving.

**Checkpoint:** Five model names with CV F1 and recall; results are computed, never hardcoded.


**Executable Python notebook cell 085**

```python
baseline_rows=[]; baseline_folds={}; baseline_models={}

def train_one_baseline(name):
    est = MODELS[name]
    pipe=make_pipeline(est)
    row,folds=cv_summary(name,pipe)
    baseline_rows.append(row); baseline_folds[name]=folds
    baseline_models[name]=pipe.fit(X_dev,y_dev)
    folds.to_csv(REPORTS/f'baseline_{name}_folds.csv',index=False)
    print(name, 'CV F1:',round(row['cv_f1_mean'],4),'Recall:',round(row['cv_recall_mean'],4))
```


<!-- Notebook original cell 086: markdown -->

##### Train baseline: LogisticRegression
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 087**

```python
train_one_baseline('LogisticRegression')
```


<!-- Notebook original cell 088: markdown -->

##### Train baseline: RandomForest
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 089**

```python
train_one_baseline('RandomForest')
```


<!-- Notebook original cell 090: markdown -->

##### Train baseline: GradientBoosting
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 091**

```python
train_one_baseline('GradientBoosting')
```


<!-- Notebook original cell 092: markdown -->

##### Train baseline: SVM
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 093**

```python
train_one_baseline('SVM')
```


<!-- Notebook original cell 094: markdown -->

##### Train baseline: KNN
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 095**

```python
train_one_baseline('KNN')
```


<!-- Notebook original cell 096: markdown -->

#### Step 039 — Rank baseline models

**What and why:** Rank by F1, then recall, then lower fold variability. Plot mean F1 with fold standard deviation.

**Checkpoint:** A five-row ranked table; bars are fold variability, not confidence intervals.


**Executable Python notebook cell 097**

```python
baseline_table=rank(baseline_rows); display(baseline_table.round(4)); baseline_table.to_csv(REPORTS/'baseline_comparison.csv',index=False)
plt.figure(figsize=(10,4)); plt.bar(baseline_table.model,baseline_table.cv_f1_mean,yerr=baseline_table.cv_f1_std,capsize=4)
plt.ylim(0,1.08); plt.xticks(rotation=20); plt.ylabel('CV F1 ± fold SD'); save_plot('baseline_f1')
```


<!-- Notebook original cell 098: markdown -->

#### Step 040 — Compare all five baseline metrics side by side

**What and why:** Look beyond accuracy: false-negative-sensitive recall, precision, F1, ROC-AUC and train-CV gap describe different behavior.

**Checkpoint:** Five rows, no hardcoded winning algorithm.


**Executable Python notebook cell 099**

```python
cols=['model','cv_accuracy_mean','cv_balanced_accuracy_mean','cv_precision_mean','cv_recall_mean','cv_f1_mean','cv_roc_auc_mean','cv_f1_std','f1_gap']
display(baseline_table[cols].round(4))
ax=baseline_table.set_index('model')[['cv_precision_mean','cv_recall_mean','cv_f1_mean']].plot.bar(figsize=(11,4),ylim=(0,1.05))
ax.set_ylabel('Mean CV score'); save_plot('baseline_metric_comparison')
```


<!-- Notebook original cell 100: markdown -->

### 09 · Advanced development analysis (out-of-fold)
Each row is predicted by a model not fitted on that row or its duplicate profile. Analyze errors, probability discrimination and calibration. SVM probability estimates and the hard predictions may disagree; evaluate both intentionally. These are development diagnostics, not an unbiased final estimate after selection.


<!-- Notebook original cell 101: markdown -->

#### Step 041 — Define advanced classification metrics

**What and why:** Add ROC-AUC and average precision for ranking, and Brier loss for probabilistic accuracy.

**Checkpoint:** Metric function returns a dictionary.


**Executable Python notebook cell 102**

```python
def metrics(y_true,pred,prob):
    return {'accuracy':accuracy_score(y_true,pred),'balanced_accuracy':balanced_accuracy_score(y_true,pred),
            'precision':precision_score(y_true,pred,zero_division=0),'recall':recall_score(y_true,pred,zero_division=0),
            'f1':f1_score(y_true,pred,zero_division=0),'roc_auc':roc_auc_score(y_true,prob),
            'average_precision':average_precision_score(y_true,prob),'brier':brier_score_loss(y_true,prob)}
```


<!-- Notebook original cell 103: markdown -->

#### Step 042 — Define out-of-fold error and calibration analysis

**What and why:** Obtain predictions on unseen validation folds, classify FP/FN, draw ROC/PR/calibration charts, and inspect subgroups.

**Checkpoint:** Function defined; no holdout values used.


**Executable Python notebook cell 104**

```python
def analyze_oof(name,pipe,prefix):
    p=cross_val_predict(pipe,X_dev,y_dev,cv=cv,method='predict',n_jobs=N_JOBS)
    prob=cross_val_predict(pipe,X_dev,y_dev,cv=cv,method='predict_proba',n_jobs=N_JOBS)[:,1]
    vals=metrics(y_dev,p,prob)
    errors=X_dev.copy(); errors['actual']=y_dev; errors['predicted']=p; errors['score']=prob
    errors['error_type']=np.select([(y_dev==1)&(p==0),(y_dev==0)&(p==1)],['FN','FP'],default='correct')
    errors.to_csv(PROCESSED/f'{prefix}_oof_predictions.csv',index=False)
    print(name,vals); print(classification_report(y_dev,p,zero_division=0))
    fig,ax=plt.subplots(2,2,figsize=(10,8))
    sns.heatmap(confusion_matrix(y_dev,p,labels=[0,1]),annot=True,fmt='d',ax=ax[0,0]); ax[0,0].set(title='OOF confusion matrix',xlabel='Predicted',ylabel='Actual')
    fpr,tpr,_=roc_curve(y_dev,prob); ax[0,1].plot(fpr,tpr); ax[0,1].plot([0,1],[0,1],'--'); ax[0,1].set(title='ROC',xlabel='False positive rate',ylabel='True positive rate')
    prec,rec,_=precision_recall_curve(y_dev,prob); ax[1,0].plot(rec,prec); ax[1,0].axhline(y_dev.mean(),linestyle='--'); ax[1,0].set(title='Precision-recall',xlabel='Recall',ylabel='Precision')
    true,mean=calibration_curve(y_dev,prob,n_bins=5,strategy='quantile'); ax[1,1].plot(mean,true,'o-'); ax[1,1].plot([0,1],[0,1],'--'); ax[1,1].set(title='Calibration (diagnostic)',xlabel='Mean score',ylabel='Observed proportion')
    save_plot(prefix+'_diagnostics')
    subgroup=[]
    for field, labels in [('gender',X_dev.gender),('age_band',pd.cut(X_dev.age,[0,30,45,60,120],include_lowest=True))]:
        for label in labels.unique():
            mask=(labels==label).to_numpy(); yt=y_dev.to_numpy()[mask]; yp=p[mask]
            subgroup.append({'field':field,'group':str(label),'n':len(yt),'positives':int(yt.sum()),'false_negatives':int(((yt==1)&(yp==0)).sum()),
                             'recall':recall_score(yt,yp,zero_division=0) if yt.sum() else np.nan,
                             'f1':f1_score(yt,yp,zero_division=0) if yt.sum() else np.nan})
    subgroup=pd.DataFrame(subgroup); display(subgroup); subgroup.to_csv(REPORTS/f'{prefix}_subgroups.csv',index=False)
    display(errors.error_type.value_counts()); return vals,errors
```


<!-- Notebook original cell 105: markdown -->

#### Step 043 — Investigate the top baseline

**What and why:** Review false negatives, false positives and estimated calibration before selecting a tuning strategy.

**Checkpoint:** Confusion matrix, classification report, curves and subgroup tables appear.


**Executable Python notebook cell 106**

```python
baseline_winner=baseline_table.iloc[0].model
base_oof,base_errors=analyze_oof(baseline_winner,baseline_models[baseline_winner],'baseline_winner')
```


<!-- Notebook original cell 107: markdown -->

### 10 · MLflow initialization and five baseline experiment runs
An MLflow **experiment** groups **runs**. Each model has one baseline run storing fitted model, hyperparameters, fold metrics, aggregate metrics and data lineage. We additionally log metrics to a separate metrics-audit experiment (five runs). Use MLflow 3 returned `model_uri`, not an assumed run artifact path. We explicitly declare model dependencies to avoid recording unrelated Colab packages. Only local trusted own model serialization is used.


<!-- Notebook original cell 108: markdown -->

#### Step 044 — Initialize MLflow tracking and registry

**What and why:** Use local SQLite for metadata, Drive for artifacts, and one client to query runs and registry.

**Checkpoint:** MLflow version and tracking URI are valid.


**Executable Python notebook cell 109**

```python
import mlflow, mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
assert mlflow.__version__=='3.5.1'
DB=LOCAL/'mlflow.db'
TRACKING_URI=f'sqlite:///{DB}'
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)
os.environ['MLFLOW_TRACKING_URI']=TRACKING_URI
client=MlflowClient()
```


<!-- Notebook original cell 110: markdown -->

#### Step 045 — Create separate experiment groups

**What and why:** An experiment is a collection of runs, not a single model. Create baseline, audit, tuning and final-validation groups.

**Checkpoint:** Four experiment IDs are created.


**Executable Python notebook cell 111**

```python
def experiment(label):
    name=f'Diabetes/{RUN_ID}/{label}'
    existing=client.get_experiment_by_name(name)
    return existing.experiment_id if existing else client.create_experiment(name,artifact_location=(ARTIFACTS/label).as_uri())
BASE_EXP=experiment('01-baseline-models')
AUDIT_EXP=experiment('02-baseline-metrics-audit')
TUNE_EXP=experiment('03-adaptive-retraining')
FINAL_EXP=experiment('04-final-validation')
```


<!-- Notebook original cell 112: markdown -->

#### Step 046 — Define SQLite backup

**What and why:** Use SQLite backup instead of copying an active DB; Drive snapshot and artifact paths support recovery.

**Checkpoint:** Backup helper defined.


**Executable Python notebook cell 113**

```python
def checkpoint_mlflow():
    target=RUN/'mlflow.sqlite'
    tmp=LOCAL/'mlflow-backup.sqlite'
    with sqlite3.connect(DB) as src, sqlite3.connect(tmp) as dest:
        src.backup(dest)
    shutil.copy2(tmp,target)
    write_json(RUN/'mlflow_restore.json',{'original_tracking_uri':TRACKING_URI,'backup':str(target),'artifact_root':str(ARTIFACTS),
               'python':sys.version,'restore_note':'Mount Drive at same path; copy SQLite backup to local disk; set tracking and registry URIs to copied DB.'})
    print('MLflow database checkpoint:',target)

# Synthetic contract example, no raw observation stored as model input example.
```


<!-- Notebook original cell 114: markdown -->

#### Step 047 — Create a non-patient schema example

**What and why:** The synthetic profile demonstrates the raw serving contract without publishing actual dataset rows.

**Checkpoint:** One 16-column DataFrame.


**Executable Python notebook cell 115**

```python
fixture_record={'age':45.0,'gender':'Female',**{s:'No' for s in SYMPTOMS}}
fixture_record.update(polyuria='Yes',polydipsia='Yes',weakness='Yes')
fixture_frame=pd.DataFrame([fixture_record],columns=FEATURES)
fixture_frame['age']=fixture_frame.age.astype('float64')
```


<!-- Notebook original cell 116: markdown -->

#### Step 048 — Define MLflow model and metrics logging

**What and why:** Log parameters, aggregate metrics, fold values, data lineage and model signature as one model run.

**Checkpoint:** Helper defined; inspect the distinction between run ID and model URI.


**Executable Python notebook cell 117**

```python
def log_model_run(exp,stage,name,pipe,row,folds,extra=None):
    with mlflow.start_run(experiment_id=exp,run_name=f'{stage}-{name}') as run:
        mlflow.set_tags({'stage':stage,'model_family':name,'dataset_sha256':manifest['csv_sha256'],
                         'split_id':sha256_file(PROCESSED/'split_indices.json'),'purpose':'educational'})
        mlflow.log_params({k:str(v) for k,v in pipe.get_params().items() if k.startswith('model__')})
        mlflow.log_metrics({k:float(v) for k,v in row.items() if k!='model'})
        for _,fold in folds.iterrows():
            mlflow.log_metrics({k:float(v) for k,v in fold.items() if k.startswith(('test_','train_'))},step=int(fold['fold']))
        mlflow.log_dict(manifest,'lineage/data_manifest.json')
        mlflow.log_dict({'folds':len(cv),'seed':SEED,'grouping':'raw feature-profile SHA256'},'lineage/evaluation.json')
        if extra: mlflow.log_dict(extra,'analysis/retraining_policy.json')
        info=mlflow.sklearn.log_model(pipe,name='classifier',signature=infer_signature(fixture_frame,pipe.predict(fixture_frame)),
                                     input_example=fixture_frame,pip_requirements=MODEL_PINS)
        return {'run_id':run.info.run_id,'model_uri':info.model_uri}
```


<!-- Notebook original cell 118: markdown -->

#### Step 049 — Record five baseline model runs

**What and why:** Each trained pipeline is serialized to MLflow with its own run and URI.

**Checkpoint:** Five model references and a SQLite checkpoint.


**Executable Python notebook cell 119**

```python
baseline_refs={}

def log_one_baseline(name):
    row = next(r for r in baseline_rows if r["model"] == name)
    name=row['model']; baseline_refs[name]=log_model_run(BASE_EXP,'baseline',name,baseline_models[name],row,baseline_folds[name])
```


<!-- Notebook original cell 120: markdown -->

##### Log baseline: LogisticRegression
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 121**

```python
log_one_baseline('LogisticRegression')
print(baseline_refs['LogisticRegression'])
```


<!-- Notebook original cell 122: markdown -->

##### Log baseline: RandomForest
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 123**

```python
log_one_baseline('RandomForest')
print(baseline_refs['RandomForest'])
```


<!-- Notebook original cell 124: markdown -->

##### Log baseline: GradientBoosting
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 125**

```python
log_one_baseline('GradientBoosting')
print(baseline_refs['GradientBoosting'])
```


<!-- Notebook original cell 126: markdown -->

##### Log baseline: SVM
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 127**

```python
log_one_baseline('SVM')
print(baseline_refs['SVM'])
```


<!-- Notebook original cell 128: markdown -->

##### Log baseline: KNN
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 129**

```python
log_one_baseline('KNN')
print(baseline_refs['KNN'])
```


<!-- Notebook original cell 130: markdown -->

##### Verify completion of all five model runs


**Executable Python notebook cell 131**

```python
print(baseline_refs)
checkpoint_mlflow()
```


<!-- Notebook original cell 132: markdown -->

#### Step 050 — Inspect MLflow runs directly

**What and why:** Query tracking metadata, not your in-memory metric table. A run ID identifies an execution; the model URI identifies a loadable model.

**Checkpoint:** Five baseline run entries.


**Executable Python notebook cell 133**

```python
runs_view = mlflow.search_runs(experiment_ids=[BASE_EXP])
display(runs_view[['run_id','status','tags.model_family','metrics.cv_f1_mean']])
assert len(baseline_refs) == 5
```


<!-- Notebook original cell 134: markdown -->

### 11 · Load saved models, independently recompute metrics and compare with MLflow
Load each logged model using the returned URI. Verify hard predictions and probabilities against the in-memory fitted model. Recompute CV from cloned loaded estimators and compare stored means with a strict tolerance; this is reproducibility evidence, not additional independent validation. Metrics audit runs link to the source model run.


<!-- Notebook original cell 135: markdown -->

#### Step 051 — Define saved-model consistency audit

**What and why:** Load serialized models, compare labels and probabilities on development inputs, and recompute cross-validation from a fresh clone.

**Checkpoint:** Helper enforces deterministic metrics within tolerance.


**Executable Python notebook cell 136**

```python
def reload_and_audit(refs, fitted, rows, exp=None):
    audits=[]; loaded={}; lookup={r['model']:r for r in rows}
    for name,ref in refs.items():
        p=mlflow.sklearn.load_model(ref['model_uri']); loaded[name]=p
        hard=bool(np.array_equal(p.predict(X_dev),fitted[name].predict(X_dev)))
        prob=bool(np.allclose(p.predict_proba(X_dev),fitted[name].predict_proba(X_dev),rtol=1e-8,atol=1e-10))
        fresh,_=cv_summary(name,clone(p))
        saved=client.get_run(ref['run_id']).data.metrics
        keys=[f'cv_{key}_mean' for key in SCORING]
        diff=max(abs(fresh[k]-saved[k]) for k in keys)
        row={'model':name,'hard_equal':hard,'probability_equal':prob,'max_cv_metric_delta':diff,
             'recorded_f1':saved['cv_f1_mean'],'recomputed_f1':fresh['cv_f1_mean']}
        audits.append(row); assert hard and prob and diff<1e-7, row
        if exp is not None:
            with mlflow.start_run(experiment_id=exp,run_name=f'audit-{name}'):
                mlflow.set_tags({'source_run_id':ref['run_id'],'source_model_uri':ref['model_uri']})
                mlflow.log_metrics({k:float(v) for k,v in row.items() if k!='model'})
                mlflow.log_metrics({f'recomputed_{k}':fresh[k] for k in keys})
    return loaded,pd.DataFrame(audits)
```


<!-- Notebook original cell 137: markdown -->

#### Step 052 — Reload and log five independent metrics audits

**What and why:** Audit runs reference the original model runs. Query MLflow itself to compare saved metrics.

**Checkpoint:** Five audit rows, hard_equal and probability_equal True, metric deltas near zero.


**Executable Python notebook cell 138**

```python
loaded_baseline,baseline_audit=reload_and_audit(baseline_refs,baseline_models,baseline_rows,AUDIT_EXP)
display(baseline_audit); baseline_audit.to_csv(REPORTS/'baseline_reload_audit.csv',index=False)
recorded=mlflow.search_runs(experiment_ids=[BASE_EXP],order_by=['metrics.cv_f1_mean DESC'])
display(recorded[['run_id','tags.model_family','metrics.cv_f1_mean','metrics.cv_recall_mean','metrics.cv_f1_std']])
checkpoint_mlflow()
```


<!-- Notebook original cell 139: markdown -->

### 12 · Adaptive retraining — analysis-driven, deterministic policy
A rules-based automation (not an AI diagnosis): look at F1 train/validation gap, development recall, class ratio and OOF false negatives. If models show overfit, include stronger regularization. If recall or class balance is weak, search cost-sensitive weights where supported. Search bounded hyperparameter grids for all five families. Compare baseline against the tuned candidate on the same development folds; retain baseline parameters if tuning is worse. Retain the entire preprocessing pipeline. The holdout remains untouched.


<!-- Notebook original cell 140: markdown -->

#### Step 053 — Derive automated retraining rules

**What and why:** Use development class balance, missed positives and train-validation F1 gaps to choose search spaces; policy is deterministic, not an LLM.

**Checkpoint:** Five policy rows with parameter grids. No holdout inspection.


**Executable Python notebook cell 141**

```python
rows_by_name={r['model']:r for r in baseline_rows}
imbalance = min(y_dev.mean(),1-y_dev.mean()) < .40
need_recall = (base_errors.error_type.eq('FN').sum() > 0 and (base_oof['recall']<.90 or imbalance))
policy={}
grids={}
for name,row in rows_by_name.items():
    overfit=row['f1_gap']>.08
    weights=[None,'balanced'] if (need_recall or imbalance or row['cv_recall_mean']<.90) else [None]
    policy[name]={'overfit_gap':row['f1_gap'],'strong_regularization':overfit,'class_weight_options':weights,
                  'development_imbalance':bool(imbalance),'recall_priority':bool(need_recall)}
    if name=='LogisticRegression': grid={'model__C':[.03,.1,1] if overfit else [.3,1,3,10], 'model__class_weight':weights}
    elif name=='RandomForest': grid={'model__n_estimators':[120,200],'model__max_depth':[4,8,12] if overfit else [6,12,None], 'model__min_samples_leaf':[2,4] if overfit else [1,2], 'model__class_weight':weights}
    elif name=='GradientBoosting': grid={'model__n_estimators':[60,120], 'model__max_depth':[1,2] if overfit else [2,3], 'model__learning_rate':[.05,.1]}
    elif name=='SVM': grid={'model__C':[.1,1,3] if overfit else [1,3,10],'model__gamma':['scale',.03],'model__class_weight':weights}
    else: grid={'model__n_neighbors':[9,13,17] if overfit else [3,5,7,11],'model__weights':['uniform','distance'],'model__p':[1,2]}
    grids[name]=grid; policy[name]['parameter_grid']=grid
write_json(REPORTS/'retraining_policy.json',policy)
display(pd.DataFrame(policy).T)
```


<!-- Notebook original cell 142: markdown -->

#### Step 054 — Tune and retrain five models

**What and why:** GridSearchCV evaluates candidate hyperparameters using the same development folds. Retain baseline parameters if the grid result is worse by the stated rule.

**Checkpoint:** Five trained models and five new MLflow runs; this can be slow.


**Executable Python notebook cell 143**

```python
tuned_rows=[]; tuned_models={}; tuned_refs={}; tuned_folds={}

def tune_one_model(name):
    est = MODELS[name]
    search=GridSearchCV(make_pipeline(est),grids[name],cv=cv,scoring=SCORING,refit='f1',
                        n_jobs=N_JOBS,error_score='raise',return_train_score=True)
    search.fit(X_dev,y_dev)
    pd.DataFrame(search.cv_results_).to_csv(REPORTS/f'grid_{name}.csv',index=False)
    candidate=clone(search.best_estimator_)
    row,folds=cv_summary(name,candidate)
    baseline=rows_by_name[name]
    # F1 is primary; recall is deterministic tie-breaker.
    if (row['cv_f1_mean'],row['cv_recall_mean']) < (baseline['cv_f1_mean'],baseline['cv_recall_mean']):
        candidate=make_pipeline(est); row,folds=cv_summary(name,candidate); decision='retain_baseline_parameters'
    else: decision='use_grid_parameters'
    pipe=candidate.fit(X_dev,y_dev)
    tuned_models[name]=pipe; tuned_rows.append(row); tuned_folds[name]=folds
    folds.to_csv(REPORTS/f'retrained_{name}_folds.csv',index=False)
    rationale={**policy[name],'decision':decision,'grid_best_params':search.best_params_,
               'baseline_f1':baseline['cv_f1_mean'],'chosen_f1':row['cv_f1_mean']}
    tuned_refs[name]=log_model_run(TUNE_EXP,'retrained',name,pipe,row,folds,rationale)
    print(name,decision,round(row['cv_f1_mean'],4))
```


<!-- Notebook original cell 144: markdown -->

##### Tune and log: LogisticRegression
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 145**

```python
tune_one_model('LogisticRegression')
```


<!-- Notebook original cell 146: markdown -->

##### Tune and log: RandomForest
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 147**

```python
tune_one_model('RandomForest')
```


<!-- Notebook original cell 148: markdown -->

##### Tune and log: GradientBoosting
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 149**

```python
tune_one_model('GradientBoosting')
```


<!-- Notebook original cell 150: markdown -->

##### Tune and log: SVM
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 151**

```python
tune_one_model('SVM')
```


<!-- Notebook original cell 152: markdown -->

##### Tune and log: KNN
Run this model independently. Inspect the printed result or logged model reference before the next model. If the cell fails, fix the error rather than proceeding. Avoid rerunning a successful cell because it can duplicate runs or list entries.


**Executable Python notebook cell 153**

```python
tune_one_model('KNN')
```


<!-- Notebook original cell 154: markdown -->

#### Step 055 — Compare before and after retraining

**What and why:** Join each model family and calculate metric differences. Selection CV scores are exploratory after tuning and may be optimistic.

**Checkpoint:** Five-row before/after table and comparison plot.


**Executable Python notebook cell 155**

```python
tuned_table=rank(tuned_rows)
comparison=baseline_table.merge(tuned_table,on='model',suffixes=('_baseline','_retrained'))
for k in ['cv_f1_mean','cv_recall_mean','cv_roc_auc_mean','cv_balanced_accuracy_mean']:
    comparison[k+'_delta']=comparison[k+'_retrained']-comparison[k+'_baseline']
comparison.to_csv(REPORTS/'before_after_retraining.csv',index=False)
display(comparison[['model','cv_f1_mean_baseline','cv_f1_mean_retrained','cv_f1_mean_delta','cv_recall_mean_delta','cv_roc_auc_mean_delta']].round(4))
plt.figure(figsize=(10,4)); x=np.arange(len(comparison)); plt.bar(x-.18,comparison.cv_f1_mean_baseline,.36,label='baseline'); plt.bar(x+.18,comparison.cv_f1_mean_retrained,.36,label='retrained'); plt.xticks(x,comparison.model,rotation=20); plt.legend(); plt.ylim(0,1.05); save_plot('baseline_vs_retraining')
checkpoint_mlflow()
```


<!-- Notebook original cell 156: markdown -->

### 13 · Consistency, model comparison and freeze final candidate
Reload all five retrained models and compare predictions and recorded metrics. Check repeated group CV on the same development dataset with alternative split seeds. This estimates sensitivity to splits, not independent test performance, because hyperparameters were selected on development data. Compute permutation importance on held-out CV folds (raw features, not training rows). Correlated symptoms may share importance. The final winner is frozen by the predefined ranking, not by the test set.


<!-- Notebook original cell 157: markdown -->

#### Step 056 — Reload and verify retrained models

**What and why:** Repeat serialized prediction and metric consistency checks and query MLflow results.

**Checkpoint:** Five reloaded models agree with their fitted copies.


**Executable Python notebook cell 158**

```python
loaded_tuned,tuned_audit=reload_and_audit(tuned_refs,tuned_models,tuned_rows)
tuned_audit.to_csv(REPORTS/'retrained_reload_audit.csv',index=False); display(tuned_audit)
recorded_tuned=mlflow.search_runs(experiment_ids=[TUNE_EXP],order_by=['metrics.cv_f1_mean DESC'])
display(recorded_tuned[['run_id','tags.model_family','metrics.cv_f1_mean','metrics.cv_recall_mean','metrics.f1_gap']])
display(tuned_table[['model','cv_f1_mean','cv_recall_mean','cv_f1_std','cv_roc_auc_mean','f1_gap']].round(4))
```


<!-- Notebook original cell 159: markdown -->

#### Step 057 — Select the development winner

**What and why:** Select only from the ranked development table and keep its saved model reference.

**Checkpoint:** Winner has a model URI and run ID.


**Executable Python notebook cell 160**

```python
winner=tuned_table.iloc[0].model
winner_model=loaded_tuned[winner]
winner_ref=tuned_refs[winner]
```


<!-- Notebook original cell 161: markdown -->

#### Step 058 — Check model stability across three seeds

**What and why:** Repeat grouped cross-validation with different fold assignments to assess sensitivity.

**Checkpoint:** Three rows of F1, recall and variability; results are not independent test estimates.


**Executable Python notebook cell 162**

```python
repeat_rows=[]
for seed in [123,456,789]:
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(X_dev,y_dev,g_dev))
    assert all(not set(g_dev.iloc[a])&set(g_dev.iloc[b]) for a,b in splits)
    r,_=cv_summary(winner,clone(winner_model),splits); repeat_rows.append({'seed':seed,**r})
repeat_table=pd.DataFrame(repeat_rows); display(repeat_table[['seed','cv_f1_mean','cv_recall_mean','cv_f1_std']]); repeat_table.to_csv(REPORTS/'winner_repeat_cv.csv',index=False)
```


<!-- Notebook original cell 163: markdown -->

#### Step 059 — Review retrained winner errors

**What and why:** Repeat OOF diagnostic plots and inspect subgroup error counts.

**Checkpoint:** Compare with the baseline diagnostics; no guaranteed improvement.


**Executable Python notebook cell 164**

```python
tuned_oof,tuned_errors=analyze_oof(winner,winner_model,'retrained_winner')
```


<!-- Notebook original cell 165: markdown -->

#### Step 060 — Estimate cross-validated permutation importance

**What and why:** Shuffle one raw feature at a time on validation folds; measure the F1 decrease.

**Checkpoint:** Feature importance plot; negative or correlated contributions need cautious interpretation.


**Executable Python notebook cell 166**

```python
imp=[]
for tr,va in cv:
    fold_model=clone(winner_model).fit(X_dev.iloc[tr],y_dev.iloc[tr])
    result=permutation_importance(fold_model,X_dev.iloc[va],y_dev.iloc[va],scoring='f1',n_repeats=5,random_state=SEED,n_jobs=N_JOBS)
    imp.append(result.importances_mean)
importance=pd.DataFrame({'feature':FEATURES,'mean_cv_permutation':np.mean(imp,axis=0),'fold_sd':np.std(imp,axis=0,ddof=1)}).sort_values('mean_cv_permutation')
importance.to_csv(REPORTS/'feature_importance.csv',index=False)
plt.figure(figsize=(8,6)); plt.barh(importance.feature,importance.mean_cv_permutation,xerr=importance.fold_sd); plt.xlabel('Decrease in CV F1 when shuffled'); save_plot('permutation_importance')
```


<!-- Notebook original cell 167: markdown -->

#### Step 061 — Freeze the candidate before final evaluation

**What and why:** Record the winner, model URI, dataset hash and predeclared thresholds.

**Checkpoint:** FROZEN printed; do not change candidate based on the holdout.


**Executable Python notebook cell 168**

```python
freeze={'winner':winner,**winner_ref,'selection':'development CV F1, recall, fold SD','thresholds':{'f1':MIN_F1,'recall':MIN_RECALL,'balanced_accuracy':MIN_BALANCED_ACCURACY},'data_hash':manifest['csv_sha256']}
freeze_path=RUN/'frozen_candidate.json'
if freeze_path.exists(): assert json.loads(freeze_path.read_text())==freeze, 'Candidate already frozen for this run; do not reselect after holdout.'
else: write_json(freeze_path,freeze)
print('FROZEN:',freeze)
```


<!-- Notebook original cell 169: markdown -->

### 14 · Final evaluation — one frozen candidate, one holdout
Only now score the untouched holdout. Generate metrics, errors, a group-bootstrap interval and model card. Group bootstrap resamples feature profiles rather than individual duplicate rows. Small sample intervals are illustrative. The gate blocks registration and deployment if metrics fail. Do not tune on this test set or lower thresholds after seeing its results. A later improvement needs new independent data for final evaluation.


<!-- Notebook original cell 170: markdown -->

#### Step 062 — Evaluate the frozen candidate on the holdout

**What and why:** Calculate final predictions and probabilities once for the chosen pipeline.

**Checkpoint:** This is the first final evaluation, not tuning data.


**Executable Python notebook cell 171**

```python
assert json.loads(freeze_path.read_text())==freeze
pred=winner_model.predict(X_test); prob=winner_model.predict_proba(X_test)[:,1]
holdout=metrics(y_test,pred,prob)
# Bootstrap by feature-profile group; stratification is not forced, omit single-class samples.
```


<!-- Notebook original cell 172: markdown -->

#### Step 063 — Estimate group-bootstrap uncertainty

**What and why:** Resample feature groups to estimate approximate 95% intervals for F1 and recall.

**Checkpoint:** Interpret intervals as limited by small, nonrepresentative data.


**Executable Python notebook cell 173**

```python
rng=np.random.default_rng(SEED); labels=g_test.unique(); group_idx={g:np.flatnonzero(g_test.to_numpy()==g) for g in labels}
boot=[]
for _ in range(500):
    idx=np.concatenate([group_idx[g] for g in rng.choice(labels,len(labels),replace=True)])
    yt=y_test.to_numpy()[idx]
    if len(np.unique(yt))==2: boot.append([f1_score(yt,pred[idx],zero_division=0),recall_score(yt,pred[idx],zero_division=0)])
interval=np.quantile(np.asarray(boot),[.025,.975],axis=0)
```


<!-- Notebook original cell 174: markdown -->

#### Step 064 — Apply the educational acceptance gate

**What and why:** Compare holdout metrics with thresholds set at the start, and log the final evaluation.

**Checkpoint:** Gate True is needed to continue. False means stop; do not retune on the same holdout.


**Executable Python notebook cell 175**

```python
gate=bool(holdout['f1']>=MIN_F1 and holdout['recall']>=MIN_RECALL and holdout['balanced_accuracy']>=MIN_BALANCED_ACCURACY)
final_report={'candidate':freeze,'metrics':holdout,'holdout_rows':len(y_test),'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]).tolist(),
              'bootstrap_95_percent':{'f1':interval[:,0].tolist(),'recall':interval[:,1].tolist(),'valid_resamples':len(boot)},'gate_passed':gate}
write_json(REPORTS/'holdout_report.json',final_report)
with mlflow.start_run(experiment_id=FINAL_EXP,run_name=f'holdout-{winner}'):
    mlflow.set_tags({'source_run_id':winner_ref['run_id'],'source_model_uri':winner_ref['model_uri'],'gate_passed':str(gate)})
    mlflow.log_metrics({f'holdout_{k}':float(v) for k,v in holdout.items()})
    mlflow.log_dict(final_report,'evaluation/holdout_report.json')
checkpoint_mlflow(); display(pd.Series(holdout)); print('Gate:',gate); print('95% bootstrap intervals:',interval)
plt.figure(figsize=(4,3)); sns.heatmap(confusion_matrix(y_test,pred,labels=[0,1]),annot=True,fmt='d'); plt.xlabel('Predicted'); plt.ylabel('Actual'); save_plot('holdout_confusion')
assert gate, 'Educational evaluation gate failed. Stop deployment; do not tune on this holdout.'
```


<!-- Notebook original cell 176: markdown -->

### 15 · MLflow Registry and alias
Register the frozen passing model as a version, assign `candidate`, and verify that alias loading reproduces the selected model. Registry alias is mutable; the Bento export is pinned to the resolved model version and Bento tag. This notebook uses one isolated registry per run, so it does not accidentally repoint a shared production model.


<!-- Notebook original cell 177: markdown -->

#### Step 065 — Register the approved model version

**What and why:** A registry entry names a versioned model; tag it with purpose and data lineage.

**Checkpoint:** A registered version is returned.


**Executable Python notebook cell 178**

```python
assert gate
REGISTRY_NAME='Diabetes-Risk-Educational'
version=mlflow.register_model(winner_ref['model_uri'],REGISTRY_NAME,await_registration_for=120)
client.set_model_version_tag(REGISTRY_NAME,version.version,'dataset_sha256',manifest['csv_sha256'])
client.set_model_version_tag(REGISTRY_NAME,version.version,'purpose','educational-only')
```


<!-- Notebook original cell 179: markdown -->

#### Step 066 — Assign and resolve the candidate alias

**What and why:** An alias is a movable pointer. Verify it resolves to the exact intended version.

**Checkpoint:** The resolved version matches.


**Executable Python notebook cell 180**

```python
client.set_registered_model_alias(REGISTRY_NAME,'candidate',version.version)
alias_uri=f'models:/{REGISTRY_NAME}@candidate'
resolved=client.get_model_version_by_alias(REGISTRY_NAME,'candidate')
assert str(resolved.version)==str(version.version)
```


<!-- Notebook original cell 181: markdown -->

#### Step 067 — Load by alias and compare predictions

**What and why:** Verify the registry model has identical predictions and scores to the frozen candidate.

**Checkpoint:** Registry metadata and checkpoint are saved.


**Executable Python notebook cell 182**

```python
registry_model=mlflow.sklearn.load_model(alias_uri)
assert np.array_equal(registry_model.predict(X_test),pred)
assert np.allclose(registry_model.predict_proba(X_test),prob)
registry_meta={'registered_name':REGISTRY_NAME,'version':str(version.version),'alias':'candidate','model_uri':alias_uri,'source':winner_ref}
write_json(REPORTS/'registry.json',registry_meta); checkpoint_mlflow(); print(registry_meta)
```


<!-- Notebook original cell 183: markdown -->

#### Optional MLflow UI (local only)
The UI is optional. Colab does not expose localhost directly in the ordinary browser; the notebook can display an iframe through Colab's supported proxy, but do not expose MLflow publicly. The stored metrics tables above work without a UI. Stop the UI before copying/restoring the database for other use.


<!-- Notebook original cell 184: markdown -->

#### Step 068 — Optional: open MLflow UI

**What and why:** Set RUN_MLFLOW_UI to True only if you want to inspect local runs via Colab. Keep the service bound to localhost.

**Checkpoint:** If disabled, no output is expected; MLflow logging still works.


**Executable Python notebook cell 185**

```python
RUN_MLFLOW_UI=False
if RUN_MLFLOW_UI:
    ui_log=open(LOCAL/'mlflow_ui.log','w')
    ui_proc=subprocess.Popen(['mlflow','ui','--backend-store-uri',TRACKING_URI,'--host','127.0.0.1','--port','5000'],stdout=ui_log,stderr=subprocess.STDOUT)
    from google.colab import output
    output.serve_kernel_port_as_window(5000)
```


<!-- Notebook original cell 186: markdown -->

### 16 · BentoML init and registry model import
BentoML was installed in section 01; verify or install if missing. Download the exact registry version to a local MLflow model directory, then import into BentoML. The exported Bento model contains its own copied MLflow model artifacts and does not need access to the tracking database at serving time. Test hard prediction consistency before creating the API.


<!-- Notebook original cell 187: markdown -->

#### Step 069 — Check and import BentoML

**What and why:** Install only if absent, confirm version, and enforce the educational gate.

**Checkpoint:** Correct BentoML version.


**Executable Python notebook cell 188**

```python
import importlib.util
if importlib.util.find_spec('bentoml') is None:
    subprocess.run([sys.executable,'-m','pip','install','bentoml==1.4.27'],check=True)
import bentoml, bentoml.mlflow
assert metadata.version('bentoml')=='1.4.27'
assert gate
```


<!-- Notebook original cell 189: markdown -->

#### Step 070 — Import the exact registry version into BentoML

**What and why:** Download the registered model to local disk and import it as a Bento model with its own tag. Use version, not a mutable alias.

**Checkpoint:** BENTO_TAG identifies the exported artifact.


**Executable Python notebook cell 190**

```python
pinned_uri=f'models:/{REGISTRY_NAME}/{version.version}'
mlflow_dir=mlflow.artifacts.download_artifacts(artifact_uri=pinned_uri,dst_path=str(LOCAL))
bento_model=bentoml.mlflow.import_model('diabetes_risk_educational',model_uri=str(mlflow_dir),
                         labels={'model_family':winner,'registry_version':str(version.version)})
BENTO_TAG=str(bento_model.tag)
```


<!-- Notebook original cell 191: markdown -->

#### Step 071 — Verify the imported model

**What and why:** Load Bento model and compare its output with the original registry model.

**Checkpoint:** Prediction equality assertions pass and model store is displayed.


**Executable Python notebook cell 192**

```python
serving_model=bentoml.mlflow.load_model(BENTO_TAG)
assert np.array_equal(np.asarray(serving_model.predict(X_test)),pred)
assert np.array_equal(np.asarray(serving_model.predict(fixture_frame)),registry_model.predict(fixture_frame))
print('Bento tag:',BENTO_TAG); print('Fixture prediction:',serving_model.predict(fixture_frame))
display(pd.DataFrame({'tag':[str(m.tag) for m in bentoml.models.list()]}))
```


<!-- Notebook original cell 193: markdown -->

### 17 · Generate a typed BentoML service in Colab
The API accepts the original 16 fields with strict categorical choices and bounded age. It applies preprocessing and the selected classifier together. The model reference is an immutable Bento tag, not `latest`. Service responses include the selected family and registry version. Inference remains local and is not clinical advice.


<!-- Notebook original cell 194: markdown -->

#### Step 072 — Write the entire BentoML service

**What and why:** Read the multiline Python template: Pydantic validates raw inputs, BentoModel resolves the artifact, and predict wraps a single record in a DataFrame.

**Checkpoint:** service.py is generated, syntax checked and printed.


**Executable Python notebook cell 195**

```python
service_template = """import bentoml
import bentoml.mlflow
import pandas as pd
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict

YesNo = Literal["Yes", "No"]

class PatientInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float = Field(ge=1, le=120, allow_inf_nan=False)
    gender: Literal["Male", "Female"]
    polyuria: YesNo
    polydipsia: YesNo
    sudden_weight_loss: YesNo
    weakness: YesNo
    polyphagia: YesNo
    genital_thrush: YesNo
    visual_blurring: YesNo
    itching: YesNo
    irritability: YesNo
    delayed_healing: YesNo
    partial_paresis: YesNo
    muscle_stiffness: YesNo
    alopecia: YesNo
    obesity: YesNo

@bentoml.service
class DiabetesClassifier:
    model_ref = bentoml.models.BentoModel(__TAG__)

    def __init__(self):
        self.model = bentoml.mlflow.load_model(self.model_ref)

    @bentoml.api
    def predict(self, input_data: PatientInput) -> dict:
        frame = pd.DataFrame([input_data.model_dump()])
        frame["age"] = frame["age"].astype("float64")
        result = int(self.model.predict(frame)[0])
        return {"prediction": result, "label": "Positive" if result == 1 else "Negative",
                "model": __MODEL_LABEL__, "registry_version": __VERSION__}
"""
assert gate
service_code=(service_template.replace('__TAG__',repr(BENTO_TAG))
              .replace('__MODEL_LABEL__',repr(winner+'-Educational'))
              .replace('__VERSION__',repr(str(version.version))))
compile(service_code,'service.py','exec')
(DEPLOY/'service.py').write_text(service_code,encoding='utf-8')
print(service_code)
```


<!-- Notebook original cell 196: markdown -->

### 18 · Create deterministic API fixtures and automated tests
Three synthetic profiles (not actual patients) are evaluated directly by the frozen model. The expected responses are frozen into a golden fixture; the API must match them before publication. Tests include invalid age, unknown symptom, missing field, extra field, wrong method, repeated requests and readiness. A validation status must be 400 or 422 and identify the relevant field. A 500 is never accepted as input rejection.


<!-- Notebook original cell 197: markdown -->

#### Step 073 — Build golden API fixtures

**What and why:** Generate three synthetic input profiles and expected responses from the registered model. These test serving parity, not diagnostic accuracy.

**Checkpoint:** Three golden cases and test_request.json saved.


**Executable Python notebook cell 198**

```python
profiles=[fixture_record.copy(),{'age':28.0,'gender':'Male',**{s:'No' for s in SYMPTOMS}},
          {'age':65.0,'gender':'Female',**{s:'Yes' for s in SYMPTOMS}}]
fixture_cases=[]
for i,record in enumerate(profiles,1):
    frame=pd.DataFrame([record],columns=FEATURES); frame.age=frame.age.astype('float64')
    p=int(registry_model.predict(frame)[0])
    fixture_cases.append({'name':f'synthetic-{i}','request':{'input_data':record},
                          'expected':{'prediction':p,'label':'Positive' if p==1 else 'Negative',
                                      'model':winner+'-Educational','registry_version':str(version.version)}})
write_json(DEPLOY/'test_request.json',fixture_cases[0]['request'])
write_json(DEPLOY/'api_fixtures.json',fixture_cases)
display(pd.DataFrame([{'case':x['name'],**x['expected']} for x in fixture_cases]))
```


<!-- Notebook original cell 199: markdown -->

#### Step 074 — Inspect an actual request and expected response

**What and why:** JSON is the wire contract. The nested input_data key matches the BentoML predict method argument.

**Checkpoint:** Request includes all features and response includes the model version.


**Executable Python notebook cell 200**

```python
print('POST /predict request:')
print(json.dumps(fixture_cases[0]['request'], indent=2))
print('Expected response:')
print(json.dumps(fixture_cases[0]['expected'], indent=2))
```


<!-- Notebook original cell 201: markdown -->

#### Step 075 — Write automated API regression tests

**What and why:** Read the full test script. It waits for readiness, checks golden outputs, rejects invalid fields, checks HTTP method, and repeats predictions.

**Checkpoint:** test_api.py compiles; it runs after server startup.


**Executable Python notebook cell 202**

```python
(DEPLOY/"test_api.py").write_text('''"""Standard-library API regression tests; run with API_BASE_URL=... python3 test_api.py."""
import copy
import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:3001").rstrip("/")


def request(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            return response.status, response.read().decode(), time.perf_counter()-start
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(), time.perf_counter()-start


def wait_ready(seconds=180):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        try:
            if request("GET", "/readyz")[0] == 200:
                return True
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    return False


def correct_result(status, body, expected):
    if status != 200:
        return False
    try:
        result = json.loads(body)
        return type(result.get("prediction")) is int and result == expected
    except (ValueError, AttributeError):
        return False


def rejects_field(status, body, field):
    if status not in (400, 422):
        return False
    try:
        details = json.loads(body).get("detail", [])
        return isinstance(details, list) and any(
            isinstance(item, dict) and field in item.get("loc", []) for item in details)
    except (ValueError, TypeError, AttributeError):
        return False


def main():
    fixtures = json.loads((HERE / "api_fixtures.json").read_text())
    results = []
    def check(name, passed, detail=""):
        results.append(bool(passed))
        print(f"{'PASS' if passed else 'FAIL'}: {name}" + (f" | {detail}" if detail else ""))
    print("Testing:", BASE_URL)
    check("API readiness", wait_ready())
    if not results[0]:
        print("API did not become ready")
        return 1
    for case in fixtures:
        status, body, elapsed = request("POST", "/predict", case["request"])
        check(case["name"] + " golden prediction", correct_result(status, body, case["expected"]),
              f"HTTP {status}, {elapsed*1000:.1f} ms" + (f" body={body[:300]}" if status != 200 else ""))
    sample = fixtures[0]["request"]
    changes = [("negative age", "age", -5), ("age above limit", "age", 121),
               ("unknown symptom", "polyuria", "Maybe"), ("unknown gender", "gender", "Other"),
               ("extra field", "not_a_feature", "unexpected")]
    for name, field, value in changes:
        payload = copy.deepcopy(sample); payload["input_data"][field] = value
        status, body, _ = request("POST", "/predict", payload)
        check(name, rejects_field(status, body, field), f"HTTP {status}: {body[:250]}")
    payload = copy.deepcopy(sample); del payload["input_data"]["age"]
    status, body, _ = request("POST", "/predict", payload)
    check("missing age", rejects_field(status, body, "age"), f"HTTP {status}")
    status, _, _ = request("GET", "/predict")
    check("GET prediction rejected", status == 405, f"HTTP {status}")
    repeated = [request("POST", "/predict", sample) for _ in range(5)]
    check("five repeat predictions consistent", all(correct_result(s,b,fixtures[0]["expected"]) for s,b,_ in repeated))
    print(f"RESULT: {sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
''',encoding="utf-8")
compile((DEPLOY/"test_api.py").read_text(),"test_api.py","exec")
```


<!-- Notebook original cell 203: markdown -->

### 19 · Start the BentoML server in Colab, send requests and analyze responses
The server runs as a child process bound to loopback, no external exposure. Wait for readiness rather than assuming startup is instantaneous. Inspect JSON and HTTP status, run regression tests, and compare a few sequential request durations. This is an API experiment, not a monitoring implementation. Always stop the child process when finished.


<!-- Notebook original cell 204: markdown -->

#### Step 076 — Start the BentoML server in a child process

**What and why:** Use localhost and port 3000, save stdout/stderr to a log, and start only this notebook’s service.

**Checkpoint:** Process starts; readiness is checked next.


**Executable Python notebook cell 205**

```python
import signal
API_PORT=3000
API_URL=f'http://127.0.0.1:{API_PORT}'
assert shutil.which('bentoml'), 'BentoML CLI not on PATH'
if 'server_proc' in globals() and server_proc.poll() is None:
    os.killpg(server_proc.pid,signal.SIGTERM); server_proc.wait(timeout=30)
server_log=open(LOCAL/'bento_server.log','w')
server_proc=subprocess.Popen(['bentoml','serve','service:DiabetesClassifier','--host','127.0.0.1','--port',str(API_PORT)],
                              cwd=DEPLOY,stdout=server_log,stderr=subprocess.STDOUT,start_new_session=True)
```


<!-- Notebook original cell 206: markdown -->

#### Step 077 — Wait for API readiness

**What and why:** Poll /readyz until HTTP 200, or show server logs if the child exits or the timeout expires.

**Checkpoint:** BentoML ready printed. If not, inspect logs before proceeding.


**Executable Python notebook cell 207**

```python
ready=False
for _ in range(90):
    if server_proc.poll() is not None: break
    try:
        if requests.get(API_URL+'/readyz',timeout=2).status_code==200: ready=True; break
    except requests.RequestException: pass
    time.sleep(2)
if not ready: print((LOCAL/'bento_server.log').read_text()[-6000:])
assert ready, 'BentoML failed readiness; inspect server log.'
print('BentoML ready:',API_URL)
```


<!-- Notebook original cell 208: markdown -->

#### Step 078 — Send one request manually from Colab

**What and why:** requests.get checks readiness and requests.post sends the JSON body. Inspect response headers and output before running the batch experiments.

**Checkpoint:** HTTP 200 and golden result match.


**Executable Python notebook cell 209**

```python
r = requests.get(API_URL + '/readyz', timeout=10)
print('Readiness:', r.status_code, repr(r.text))
r = requests.post(API_URL + '/predict', json=fixture_cases[0]['request'], timeout=20)
print('Prediction:', r.status_code, r.headers.get('Content-Type'), r.text)
assert r.status_code == 200 and r.json() == fixture_cases[0]['expected']
```


<!-- Notebook original cell 210: markdown -->

#### Step 079 — Experiment with valid API requests

**What and why:** Send three synthetic requests three times, record response status, model version, prediction and timing.

**Checkpoint:** Nine response rows and latency summary; latency is descriptive, not a load test.


**Executable Python notebook cell 211**

```python
import copy
observations=[]
for case in fixture_cases:
    for attempt in range(1,4):
        t=time.perf_counter(); r=requests.post(API_URL+'/predict',json=case['request'],timeout=20); elapsed=(time.perf_counter()-t)*1000
        r.raise_for_status(); body=r.json(); assert body==case['expected']
        observations.append({'case':case['name'],'attempt':attempt,'status':r.status_code,'latency_ms':elapsed,**body})
api_analysis=pd.DataFrame(observations); display(api_analysis)
display(api_analysis.groupby('case').latency_ms.agg(['min','median','max']))
api_analysis.to_csv(REPORTS/'api_experiment.csv',index=False)
```


<!-- Notebook original cell 212: markdown -->

#### Step 080 — Experiment with invalid input

**What and why:** Send age -5 and inspect Pydantic error body. BentoML may return 400 or 422.

**Checkpoint:** Client error, not HTTP 200 or 500.


**Executable Python notebook cell 213**

```python
bad=copy.deepcopy(fixture_cases[0]['request']); bad['input_data']['age']=-5
r=requests.post(API_URL+'/predict',json=bad,timeout=20); print('Invalid request:',r.status_code,r.text)
assert r.status_code in (400,422)
```


<!-- Notebook original cell 214: markdown -->

#### Step 081 — Run the entire regression suite

**What and why:** Invoke the generated test script against the actual Colab server and capture output.

**Checkpoint:** RESULT: 12/12 tests passed; nonzero exit stops packaging.


**Executable Python notebook cell 215**

```python
run_tests=subprocess.run([sys.executable,str(DEPLOY/'test_api.py')],cwd=DEPLOY,
                         env={**os.environ,'API_BASE_URL':API_URL},capture_output=True,text=True)
print(run_tests.stdout,run_tests.stderr)
( REPORTS/'colab_api_test.txt').write_text(run_tests.stdout+run_tests.stderr)
assert run_tests.returncode==0
API_TEST_PASS=True
```


<!-- Notebook original cell 216: markdown -->

#### Step 082 — Stop the notebook server safely

**What and why:** Terminate only the child process group, close logs, and back up experiment metadata.

**Checkpoint:** Local serving experiment finished.


**Executable Python notebook cell 217**

```python
# Stop only the process group launched by this notebook.
if server_proc.poll() is None:
    os.killpg(server_proc.pid,signal.SIGTERM)
    try: server_proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(server_proc.pid,signal.SIGKILL); server_proc.wait()
server_log.close()
shutil.copy2(LOCAL/'bento_server.log',REPORTS/'bento_server.log')
if 'ui_proc' in globals() and ui_proc.poll() is None:
    ui_proc.terminate(); ui_proc.wait(timeout=30); ui_log.close()
checkpoint_mlflow()
print('Local serving experiment finished.')
```


<!-- Notebook original cell 218: markdown -->

### 20 · Preserve the model and generate serving requirements
Export the exact verified Bento model, save the checksum and provenance. Build a minimal top-level dependency manifest from the training environment with the explicitly recorded model requirements. Transitive dependencies are not fully hash-locked: the Docker `pip check`, golden tests and image digest are the release checks. Do not substitute an untrusted model archive. The published image may contain learned information from this public dataset; do not use these steps with confidential datasets without review.


<!-- Notebook original cell 219: markdown -->

#### Step 083 — Export Bento model for Docker

**What and why:** Export the exact tested Bento tag, not a fresh model. Refuse silent overwrite.

**Checkpoint:** diabetes_model.bentomodel exists.


**Executable Python notebook cell 220**

```python
assert gate and API_TEST_PASS
archive=DEPLOY/'diabetes_model.bentomodel'
assert not archive.exists(), 'Archive already exists; do not silently replace an exported model.'
exported=bentoml.models.export_model(BENTO_TAG,str(archive))
assert archive.is_file(), exported
```


<!-- Notebook original cell 221: markdown -->

#### Step 084 — Write requirements and model manifest

**What and why:** Pin actual serving package versions and save model and data hashes, evaluation summary and source version.

**Checkpoint:** Requirements and JSON manifests saved.


**Executable Python notebook cell 222**

```python
(DEPLOY/'requirements.txt').write_text('\n'.join(SERVING_PINS)+'\n')
provenance={'dataset_sha256':manifest['csv_sha256'],'model_archive_sha256':sha256_file(archive),
            'registry':registry_meta,'bento_tag':BENTO_TAG,'algorithm':winner,'educational_gate_passed':gate,
            'holdout_metrics':holdout,'training_python':sys.version,'dependencies':VERSIONS,'run_id':RUN_ID}
write_json(DEPLOY/'model_manifest.json',provenance)
write_json(DEPLOY/'evaluation_summary.json',final_report)
print(exported, archive.stat().st_size, provenance['model_archive_sha256'])
```


<!-- Notebook original cell 223: markdown -->

#### Step 085 — Create model artifact verifier

**What and why:** Validate the artifact checksum, evaluation gate and fixture count before builds. This detects corruption, not malicious replacement of both model and manifest.

**Checkpoint:** Model manifest and fixture integrity verified.


**Executable Python notebook cell 224**

```python
(DEPLOY/"verify_artifact.py").write_text("""import hashlib, json
from pathlib import Path
p = Path(__file__).resolve().parent
m = json.loads((p / 'model_manifest.json').read_text())
assert m['educational_gate_passed'] is True
assert hashlib.sha256((p / 'diabetes_model.bentomodel').read_bytes()).hexdigest() == m['model_archive_sha256'], 'Model artifact checksum mismatch'
assert len(json.loads((p / 'api_fixtures.json').read_text())) == 3
print('Model manifest and fixture integrity verified:', m['bento_tag'])
""",encoding="utf-8")
subprocess.run([sys.executable,str(DEPLOY/"verify_artifact.py")],check=True)
```


<!-- Notebook original cell 225: markdown -->

### 21 · Docker — layers, image, container and Compose
A Dockerfile is a recipe; each filesystem-changing instruction creates/cache-reuses a layer. An image is the built artifact; a container is a runnable instance. `COPY requirements.txt` precedes `COPY service.py` so code changes usually reuse the dependency layer. We import the Bento archive during build and run as a non-root user. The system uses same Python major/minor as training. Compose publishes only loopback port 3001 and checks `/readyz`. A health check is a deployment gate, not a model quality metric. CI uses the same files you use locally.


<!-- Notebook original cell 226: markdown -->

#### Step 086 — Write Dockerfile layer by layer

**What and why:** FROM sets Python base; WORKDIR sets location; COPY requirements and RUN pip create a cacheable dependency layer; non-root USER imports the model; CMD starts BentoML.

**Checkpoint:** Dockerfile is generated; no image built yet.


**Executable Python notebook cell 227**

```python
docker_template="""FROM python:__PYTHON__-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \\
    BENTOML_HOME=/app/bentoml BENTOML_DO_NOT_TRACK=True
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt && python -m pip check
RUN useradd --create-home --uid 10001 appuser && chown appuser:appuser /app
COPY --chown=appuser:appuser diabetes_model.bentomodel ./
COPY --chown=appuser:appuser model_manifest.json api_fixtures.json verify_artifact.py ./
USER appuser
RUN python verify_artifact.py && bentoml models import ./diabetes_model.bentomodel && rm ./diabetes_model.bentomodel
COPY --chown=appuser:appuser service.py ./
EXPOSE 3000
CMD ["bentoml", "serve", "service:DiabetesClassifier", "--host", "0.0.0.0", "--port", "3000"]
"""
(DEPLOY/'Dockerfile').write_text(docker_template.replace('__PYTHON__',f'{sys.version_info.major}.{sys.version_info.minor}'))
```


<!-- Notebook original cell 228: markdown -->

#### Step 087 — Write Compose configuration

**What and why:** DIABETES_IMAGE selects registry or local image; host 3001 maps to container 3000, readiness probes are internal and resources are limited.

**Checkpoint:** compose.yaml saved; localhost-only exposure.


**Executable Python notebook cell 229**

```python
(DEPLOY/'compose.yaml').write_text("""name: diabetes-mlops
services:
  diabetes-api:
    image: ${DIABETES_IMAGE:-diabetes-bentoml:v1}
    pull_policy: never
    ports:
      - "127.0.0.1:${DIABETES_PORT:-3001}:3000"
    restart: unless-stopped
    mem_limit: 1g
    cpus: 1.0
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:3000/readyz', timeout=5)"]
      interval: 10s
      timeout: 8s
      retries: 6
      start_period: 60s
""")
```


<!-- Notebook original cell 230: markdown -->

#### Step 088 — Limit build context and Git contents

**What and why:** Allowlist only files required in the image; exclude temporary data, secrets and local experiment outputs from Git.

**Checkpoint:** Ignore files saved and Docker/Compose text displayed.


**Executable Python notebook cell 231**

```python
(DEPLOY/'.dockerignore').write_text("""*
!Dockerfile
!requirements.txt
!service.py
!diabetes_model.bentomodel
!model_manifest.json
!api_fixtures.json
!verify_artifact.py
""")
(DEPLOY/'.gitignore').write_text("""__pycache__/
*.py[cod]
.env
.env.*
!.env.example
.venv/
*.pem
*.key
*.sqlite
*.db
*.zip
raw/
processed/
extracted/
mlruns/
mlflow_artifacts/
reports/
.ipynb_checkpoints/
""")
print((DEPLOY/'Dockerfile').read_text()); print((DEPLOY/'compose.yaml').read_text())
```


<!-- Notebook original cell 232: markdown -->

### 22 · Optional actual Docker build/run in Colab
Most Colab runtimes do not have a usable Docker daemon. Detect capability instead of trying to install or start a privileged daemon. If Docker is available, explicitly set `RUN_DOCKER_HERE=True`, then build the image, inspect layers, create a container, check health and execute regression tests. Otherwise the exact same smoke script runs in GitHub Actions on an Ubuntu runner. Manual equivalent commands are shown below for learning, not required on your new VM after CI.


<!-- Notebook original cell 233: markdown -->

#### Step 089 — Create smoke test automation

**What and why:** Shell script validates the artifact, starts an isolated Compose project, runs tests and cleans up its test container even on failure.

**Checkpoint:** scripts/smoke.sh saved.


**Executable Python notebook cell 234**

```python
(DEPLOY/'scripts').mkdir(exist_ok=True)
(DEPLOY/'scripts/smoke.sh').write_text("""#!/usr/bin/env bash
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
""")
```


<!-- Notebook original cell 235: markdown -->

#### Step 090 — Create deployment and rollback automation

**What and why:** Accept an immutable GHCR digest, serialize changes with flock, pull before switching, check the exact running image, and restore the prior image on failure.

**Checkpoint:** deploy.sh saved; no remote deployment executed.


**Executable Python notebook cell 236**

```python
(DEPLOY/'scripts/deploy.sh').write_text("""#!/usr/bin/env bash
# Trusted lab host only. Accept an immutable GHCR manifest digest, never a mutable tag.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
image="${1:-}"
if [[ ! "$image" =~ ^ghcr\\.io/[a-z0-9._-]+/[a-z0-9._/-]+@sha256:[a-f0-9]{64}$ ]]; then
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
""")
```


<!-- Notebook original cell 237: markdown -->

#### Step 091 — Validate shell scripts

**What and why:** Use bash -n to check shell syntax before sending scripts to GitHub.

**Checkpoint:** No syntax error.


**Executable Python notebook cell 238**

```python
for script in (DEPLOY/'scripts').glob('*.sh'):
    script.chmod(0o755)
    subprocess.run(['bash','-n',str(script)],check=True)
```


<!-- Notebook original cell 239: markdown -->

#### Step 092 — Check Docker availability and optionally build locally

**What and why:** Colab often lacks a Docker daemon. Leave False for GitHub-hosted CI, or set True on a compatible trusted Docker environment.

**Checkpoint:** Either actual Docker commands pass or a clear deferred message appears.


**Executable Python notebook cell 240**

```python
RUN_DOCKER_HERE=False
available=bool(shutil.which('docker')) and subprocess.run(['docker','info'],capture_output=True).returncode==0
print('Docker daemon available:',available)
if RUN_DOCKER_HERE and available:
    subprocess.run(['docker','build','-t','diabetes-bentoml:v1','.'],cwd=DEPLOY,check=True)
    subprocess.run(['docker','history','diabetes-bentoml:v1'],check=True)
    subprocess.run(['bash','scripts/smoke.sh'],cwd=DEPLOY,check=True)
else:
    print('Docker creation deferred to GitHub Actions; no build success is claimed in Colab.')
```


<!-- Notebook original cell 241: markdown -->

#### Docker practice (Ubuntu terminal, optional)
```bash
## From generated repository root, only if you want to practice a manual build:
docker build -t diabetes-bentoml:v1 .
docker history diabetes-bentoml:v1
docker image ls
docker compose up -d --no-build --wait --wait-timeout 180
docker compose ps
API_BASE_URL=http://127.0.0.1:3001 python3 test_api.py
docker compose down
```
Do not run the smoke script against your live production Compose project. Its default project name is `diabetes-ci`, and it removes its own test container and network on exit. Use a different host port if 3001 is already in use.


<!-- Notebook original cell 242: markdown -->

#### Ubuntu practice — understand Docker before automating it
   Only after extracting the exported ZIP in section 25 on your own trusted Docker VM. These are optional manual exercises; use GitHub Actions in section 26 if you want automation immediately.

   **A. Inspect the build inputs**
   ```bash
   pwd
   ls -la
   cat Dockerfile
   cat requirements.txt
   python3 verify_artifact.py
   ```
   Checkpoint: files exist and artifact verifier succeeds.

   **B. Build an image and inspect its layers**
   ```bash
   docker build -t diabetes-bentoml:v1 .
   docker image ls diabetes-bentoml
   docker history diabetes-bentoml:v1
   ```
   `-t` assigns a tag; `.` is the build context. The build creates layers; it does not create a running container. Checkpoint: build exits successfully and history shows layer commands.

   **C. Create a standalone container**
   ```bash
   docker run -d --name diabetes-manual -p 127.0.0.1:3000:3000 diabetes-bentoml:v1
   docker ps
   docker logs --tail 30 diabetes-manual
   API_BASE_URL=http://127.0.0.1:3000 python3 test_api.py
   ```
   `-d` means detached; `--name` precedes the image; `-p` publishes ports. Checkpoint: 12/12 tests. If port/name exists, inspect existing resources; do not remove unrelated containers.

   **D. Remove only the practice container and use Compose instead**
   ```bash
   docker stop diabetes-manual
   docker rm diabetes-manual
   docker compose config -q
   docker compose up -d --no-build --wait --wait-timeout 180
   docker compose ps
   API_BASE_URL=http://127.0.0.1:3001 python3 test_api.py
   docker compose down
   ```
   Checkpoint: healthy and 12/12, then Compose removes its own resources. Docker image remains. After this, CI automates the build and smoke test.


<!-- Notebook original cell 243: markdown -->

### 23 · Continuous integration and continuous delivery
Two workflows: pull requests build and test with a read-only token; a push to `main` builds once, runs the same smoke tests, and publishes **that tested image** to GHCR only if every check succeeds. It publishes an immutable-to-reference SHA digest and a commit tag. Commit tags are intended for traceability but technically mutable; deploy by digest. The publish job prints the exact reference in the Actions summary. No PAT is needed to publish with GitHub Actions when package permissions allow the workflow's `GITHUB_TOKEN`.

GitHub-hosted Ubuntu is the clean build environment. All CI steps fail fast and leave a clear job status. The official checkout action is pinned to a known commit; review and update dependencies before any real deployment.


<!-- Notebook original cell 244: markdown -->

#### Step 093 — Generate pull request CI workflow

**What and why:** Untrusted PR validation runs on GitHub-hosted runners with read-only contents permission. Verify, build, start, test and clean up.

**Checkpoint:** ci.yml exists; not yet triggered.


**Executable Python notebook cell 245**

```python
(DEPLOY/'.github/workflows').mkdir(parents=True,exist_ok=True)
(DEPLOY/'.github/workflows/ci.yml').write_text("""name: Pull request CI
on:
  pull_request:
    branches: [main]
permissions:
  contents: read
jobs:
  test:
    runs-on: ubuntu-latest
    timeout-minutes: 25
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          persist-credentials: false
      - name: Verify model and source
        run: |
          python3 verify_artifact.py
          python3 -m py_compile service.py test_api.py verify_artifact.py
          bash -n scripts/smoke.sh scripts/deploy.sh
          docker compose config -q
      - name: Build image
        run: docker build -t diabetes-bentoml:v1 .
      - name: Smoke test container
        run: bash scripts/smoke.sh
""")
```


<!-- Notebook original cell 246: markdown -->

#### Step 094 — Generate main-branch delivery and optional deployment

**What and why:** Build once, test exact image, publish to GHCR using temporary GITHUB_TOKEN and return an immutable digest. Optional trusted runner deploys only when explicitly enabled.

**Checkpoint:** cd.yml exists; no runner or credentials needed in Colab.


**Executable Python notebook cell 247**

```python
(DEPLOY/'.github/workflows/cd.yml').write_text("""name: Main CI and GHCR delivery
on:
  push:
    branches: [main]
  workflow_dispatch:
permissions:
  contents: read
concurrency:
  group: diabetes-main-delivery
  cancel-in-progress: false
jobs:
  publish:
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      contents: read
      packages: write
    outputs:
      image: ${{ steps.publish.outputs.image }}
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          persist-credentials: false
      - name: Verify artifact and scripts
        run: |
          python3 verify_artifact.py
          python3 -m py_compile service.py test_api.py verify_artifact.py
          bash -n scripts/smoke.sh scripts/deploy.sh
          docker compose config -q
      - name: Build image once
        run: docker build --label "org.opencontainers.image.source=https://github.com/$GITHUB_REPOSITORY" --label "org.opencontainers.image.revision=$GITHUB_SHA" -t diabetes-bentoml:v1 .
      - name: Test the exact image
        run: bash scripts/smoke.sh
      - name: Authenticate and publish tested image
        id: publish
        env:
          GHCR_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        shell: bash
        run: |
          set -Eeuo pipefail
          repo="ghcr.io/${GITHUB_REPOSITORY,,}"
          tag="$repo:sha-$GITHUB_SHA"
          export DOCKER_CONFIG="$(mktemp -d)"
          trap 'docker logout ghcr.io >/dev/null 2>&1 || true; rm -rf "$DOCKER_CONFIG"' EXIT
          printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
          docker tag diabetes-bentoml:v1 "$tag"
          docker push "$tag" | tee push.log
          digest=$(sed -nE 's/.*digest: (sha256:[0-9a-f]{64}).*/\\1/p' push.log | tail -1)
          [[ "$digest" =~ ^sha256:[a-f0-9]{64}$ ]]
          ref="$repo@$digest"
          echo "image=$ref" >> "$GITHUB_OUTPUT"
          printf '## Published tested image\\n\\n`%s`\\n\\nCommit: `%s`\\n' "$ref" "$GITHUB_SHA" >> "$GITHUB_STEP_SUMMARY"
  # Enable only on a dedicated, persistent, trusted host. THM VMs are not suitable.
  deploy:
    needs: publish
    if: vars.ENABLE_LAB_AUTO_DEPLOY == 'true' && github.ref == 'refs/heads/main'
    runs-on: [self-hosted, linux, x64, diabetes-lab]
    timeout-minutes: 20
    environment: lab
    permissions:
      contents: read
      packages: read
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          persist-credentials: false
      - name: Deploy digest with readiness, regression tests and rollback
        env:
          GHCR_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          IMAGE_REF: ${{ needs.publish.outputs.image }}
        shell: bash
        run: |
          set -Eeuo pipefail
          export DOCKER_CONFIG="$(mktemp -d)"
          trap 'docker logout ghcr.io >/dev/null 2>&1 || true; rm -rf "$DOCKER_CONFIG"' EXIT
          printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
          bash scripts/deploy.sh "$IMAGE_REF"
""")
```


<!-- Notebook original cell 248: markdown -->

#### Step 095 — Validate workflow YAML

**What and why:** Parse YAML preserving on as a key and print job names. GitHub Actions itself must validate the workflow later.

**Checkpoint:** Both workflow files contain triggers and jobs.


**Executable Python notebook cell 249**

```python
import yaml
for file in (DEPLOY/'.github/workflows').glob('*.yml'):
    parsed=yaml.load(file.read_text(),Loader=yaml.BaseLoader)  # preserve the YAML key 'on'
    assert 'on' in parsed and 'jobs' in parsed
    print(file.name,':',list(parsed['jobs']))
print((DEPLOY/'.github/workflows/cd.yml').read_text())
```


<!-- Notebook original cell 250: markdown -->

### 24 · Deployment mode selection: ephemeral THM VM vs persistent host
**Recommended for this learning exercise:** automated CI and GHCR delivery; one command to pull and deploy on a fresh VM. A temporary TryHackMe machine is not a reliable continuous-deployment destination and may not be appropriate for storing credentials or registering a runner. Use only a VM you are authorized to use for this workload, prefer a personal Ubuntu VM, and do not expose the API externally. Public GHCR packages can be pulled anonymously (only make a package public if you are comfortable sharing its contents). For private packages, authenticate on a trusted personal machine using a minimal read-only package token and `docker login --password-stdin`; do not paste tokens into Colab or a temporary shared host.

**Optional full automatic deployment:** on a dedicated persistent private Ubuntu host, configure a restricted self-hosted GitHub Actions runner using repository Settings → Actions → Runners, with labels `linux`, `x64` and `diabetes-lab`; install Docker and Compose, verify runner account Docker access, create protected `lab` environment and require approval when supported. Protect `main`, restrict who can push workflow changes, and never allow untrusted PR jobs onto the runner. Only then set repository Actions variable `ENABLE_LAB_AUTO_DEPLOY=true`. With it unset, deployment is skipped; GitHub will not queue a nonexistent runner. The `publish` job's outputs are passed directly to the deploy job, not copied manually. Docker access is root-equivalent; do not use your everyday workstation as an untrusted runner. A private package must grant repository access.

The rollback script restores the prior image if readiness or golden tests fail. It does not implement zero downtime or database rollback; there is no application database in this lab. Keep the old version's repository checkout and fixtures for full rollback validation. Do not prune the last known good image before validation.


<!-- Notebook original cell 251: markdown -->

#### Step 096 — Generate repository README

**What and why:** Provide instructions for trusted local deployment, workflow triggers, GHCR and rollback, with limitations.

**Checkpoint:** README.md saved.


**Executable Python notebook cell 252**

```python
(DEPLOY/"README.md").write_text("""# Diabetes MLOps — educational deployment

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
""",encoding="utf-8")
```


<!-- Notebook original cell 253: markdown -->

### 25 · Final package, integrity checks and backup
Create the Git-ready deployment ZIP, containing hidden workflow files, without raw observations, logs with row-level errors, tracking DB, tokens or Colab environment secrets. Save the archive + checksum to Drive. Copy the file into an empty Git repository (avoid mixing with unrelated VM files). The notebook itself can be committed separately after clearing any sensitive outputs. The model is trusted because you trained and exported it in this run.


<!-- Notebook original cell 254: markdown -->

#### Step 097 — Validate all release files

**What and why:** Check required files, Python and shell syntax, artifact integrity, original data hashes and SQLite backup.

**Checkpoint:** All checks must pass before packaging.


**Executable Python notebook cell 255**

```python
assert gate and API_TEST_PASS
required=['service.py','Dockerfile','compose.yaml','requirements.txt','diabetes_model.bentomodel','model_manifest.json',
          'evaluation_summary.json','api_fixtures.json','test_request.json','test_api.py','verify_artifact.py','README.md',
          '.dockerignore','.gitignore','scripts/smoke.sh','scripts/deploy.sh','.github/workflows/ci.yml','.github/workflows/cd.yml']
assert all((DEPLOY/x).is_file() for x in required)
for x in ['service.py','test_api.py','verify_artifact.py']: compile((DEPLOY/x).read_text(),x,'exec')
subprocess.run([sys.executable,str(DEPLOY/'verify_artifact.py')],check=True)
for x in ['scripts/smoke.sh','scripts/deploy.sh']: subprocess.run(['bash','-n',str(DEPLOY/x)],check=True)
assert sha256_file(raw_csv)==manifest['csv_sha256'] and sha256_file(raw_zip)==raw_hash
checkpoint_mlflow()
```


<!-- Notebook original cell 256: markdown -->

#### Step 098 — Create the complete Git-ready ZIP

**What and why:** Include hidden workflows using an explicit allowlist. Save a SHA-256 sidecar and inspect the ZIP member list.

**Checkpoint:** Drive export path and checksum printed.


**Executable Python notebook cell 257**

```python
bundle=RUN/'diabetes-mlops-deployment.zip'
assert not bundle.exists(), 'Bundle already exported. Start a new session for a new release.'
with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for rel in required: z.write(DEPLOY/rel,rel)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None and set(z.namelist())==set(required)
    print('Included files:',*z.namelist(),sep='\n')
( RUN/'deployment.sha256').write_text(f'{sha256_file(bundle)}  {bundle.name}\n')
print('Drive export:',bundle,'\nSHA256:',sha256_file(bundle))
print('Ready for GitHub. CI and deployment checks must still run externally.')
```


<!-- Notebook original cell 258: markdown -->

### 26 · GitHub upload and CI/CD execution (external steps)
1. Download the ZIP from Drive; create a **new empty private repository** on GitHub (recommended). Do not initialize it with README if you want to push this existing folder straight away.
2. On your trusted computer, extract the ZIP into a clean folder. On Ubuntu use `python3 -m zipfile -e diabetes-mlops-deployment.zip ./diabetes-mlops` or unzip. Hidden files are included.
3. Inspect files and commit. Replace `YOUR_OWNER` and `YOUR_REPO` in the remote URL. Authenticate using your normal Git credential mechanism or GitHub CLI; never include tokens in the remote URL.

```bash
cd diabetes-mlops
ls -la
python3 verify_artifact.py
git init -b main
git add .
git diff --cached --stat
git commit -m "Complete diabetes MLOps training and deployment"
git remote add origin https://github.com/YOUR_OWNER/YOUR_REPO.git
git push -u origin main
```

4. GitHub → Actions → **Main CI and GHCR delivery**. Verify build, smoke tests and publishing all succeeded. A skipped optional deployment is expected. Copy the `ghcr.io/...@sha256:...` reference from the Actions job summary and note the exact commit SHA. If publishing is denied, inspect package access and permissions rather than broadening tokens unnecessarily.
5. For a public package, only change visibility if you intend to publish your model and are permitted to do so. For private packages, use a trusted machine with authorized package read access; avoid credentials on temporary/shared VMs.
6. Create a feature branch, change a nonfunctional README line, push, open a PR into `main` and verify **Pull request CI** runs. Merge only after successful checks. This demonstrates changes can trigger build and test without issuing Docker commands yourself.

#### What automation does now
`git push` → GitHub-hosted runner → model checksum verification → Docker build/layers → Compose container creation → readiness → golden regression tests → GHCR publication. On a dedicated opted-in runner, the tested digest is deployed automatically after publication and required environment approval. Otherwise deployment is one explicit command in the next section.


<!-- Notebook original cell 259: markdown -->

#### GitHub browser checklist — do this in order
   **1. Create the repository:** New repository → choose a name such as diabetes-mlops → Private → leave initialization unchecked. Copy the HTTPS URL. Use your own trusted computer to push, not a shared VM with your long-lived credentials.

   **2. Upload the files:** extract the Drive ZIP, including `.github/`, initialize Git and push using the commands above. Review `git diff --cached --stat` first; there should be no raw data or credentials.

   **3. Define the first automatic run:** open Actions. The `Main CI and GHCR delivery` workflow starts after the push to main. Expand the publish job, watch each step: checkout → artifact verification → Docker build → smoke → publish. A failure stops later dependent steps. Open the failed step to see logs. Fix source, commit and push again; don't manually publish an untested image.

   **4. Collect delivery evidence:** open the successful run summary, copy the exact digest and commit SHA. In repository Packages, verify the image exists. `sha-<commit>` is a tag; `@sha256:...` is the immutable deployment identity.

   **5. Practice pull request CI:** on your trusted checkout, run:
   ```bash
   git switch -c lab/readme-ci
   printf '\nCI practice change.\n' >> README.md
   git add README.md
   git commit -m "Practice pull request CI"
   git push -u origin lab/readme-ci
   ```
   In GitHub open a PR to main, inspect `Pull request CI`, merge only when green. Main workflow runs again and publishes a new tested digest. This shows the difference between PR CI and main delivery. The optional deployment job stays skipped until you explicitly configure a dedicated persistent runner.

   **Checkpoint:** record the actual Actions URL, SHA and digest in section 28. Do not mark the pipeline complete if it failed.


<!-- Notebook original cell 260: markdown -->

### 27 · Fresh VM deployment (no manual Docker build)
Use a machine you control and are authorized to use. Verify `docker version`, `docker compose version`, `git --version`, `python3 --version`, `df -h /`, `free -h`. It must have sufficient space for the old and new images during rollback (several GiB is prudent). Use a fresh checkout at the commit that produced the digest; for private repo use an authorized trusted machine.

```bash
git clone https://github.com/YOUR_OWNER/YOUR_REPO.git diabetes-mlops
cd diabetes-mlops
git checkout YOUR_PUBLISHED_COMMIT_SHA
## Copy real digest from GitHub Actions; do not run with placeholder.
bash scripts/deploy.sh 'ghcr.io/OWNER/REPO@sha256:REPLACE_WITH_REAL_DIGEST'
docker compose ps
curl -i http://127.0.0.1:3001/readyz
API_BASE_URL=http://127.0.0.1:3001 python3 test_api.py
```
Expected: healthy container, HTTP 200, all 12 contract tests pass. No `docker build` needed: your image came from the registry. The script internally runs Docker pull, Compose and tests. Package and repository can be public independently, so check both access settings.

#### Recovery exercise (only on a disposable lab service)
- Keep the previous successful digest and its Git revision. Avoid pruning the previous image.
- Deploy a second successfully published digest and verify its matching fixtures. Deployment script saves current and previous image references in `~/.local/state/diabetes-mlops/`.
- To manually roll back, check out the previous commit (old contract fixtures), and pass its exact digest to `scripts/deploy.sh`. Confirm readiness and old regression tests.
- If a deployment has failed, inspect `docker compose logs --tail 100`; script attempts to restore old image automatically. Invalid digest or authentication failures happen before replacement.
- For an intentional restart exercise: `docker compose restart` then verify readiness and tests. Be aware this causes short downtime.

Do not delete unrelated images or containers on shared machines. A new VM can always pull the published image again.


<!-- Notebook original cell 261: markdown -->

#### Deployment troubleshooting and verification
   On the target VM, `docker info` must succeed. If it doesn't, fix Docker access on a machine you control. Check `df -h /` before pulling. Use a public image only if you intend to share its contents and have permission; never paste credentials into shared or ephemeral machines. On trusted hosts, authenticate privately when needed.

   `scripts/deploy.sh` is the single deployment command: it pulls the exact image, creates/replaces the Compose container, waits for readiness, runs API tests and tries to restore the previous image if tests fail. You do not type `docker build` on the new VM.

   **Verify:** `docker compose ps` should show healthy; `curl` should give 200; `test_api.py` must report 12/12. If port 3001 is occupied, inspect `docker ps` and use another free localhost port via `DIABETES_PORT`; pass the same port to tests. If image pull fails, check repository/package visibility and actual digest. If tests fail, read `docker compose logs --tail 100` and use the previous commit's fixtures for rollback verification.

   **Full automation:** on a dedicated persistent trusted host, follow section 24 to register the labeled runner and enable the deployment variable only after approval and access restrictions. Push a new commit and inspect the optional deploy job. Temporary THM VMs are suitable only if authorized for manual pull/deploy practice, not persistent runners. No monitoring stack is introduced.


<!-- Notebook original cell 262: markdown -->

### 28 · Completion checklist and release decision
Record actual external outcomes rather than marking them complete in advance. The optional cells are explicit user-entered evidence. Monitoring is the next independent phase and is not included here.


<!-- Notebook original cell 263: markdown -->

#### Step 099 — Record actual CI/CD evidence

**What and why:** Set external flags only after observing results from GitHub and your VM. False is correct before executing external steps.

**Checkpoint:** completion_checklist.json saved; no external success assumed.


**Executable Python notebook cell 264**

```python
# Fill only AFTER observing each result. Leaving False is correct until you have evidence.
external_evidence={
 'github_ci_passed':False,
 'ghcr_image_published':False,
 'fresh_vm_deployed':False,
 'fresh_vm_api_tests_passed':False,
 'rollback_exercise_passed':False,
 'published_digest':'',
 'github_commit':'',
 'github_actions_run_url':'',
}
progress={'raw_preserved':sha256_file(raw_zip)==raw_hash,'group_split_safe':not bool(set(g_dev)&set(g_test)),
          'five_baselines_logged':len(baseline_refs)==5,'five_metrics_audits':len(baseline_audit)==5,
          'five_retrained_logged':len(tuned_refs)==5,'holdout_gate':gate,'registry_alias_tested':True,
          'bento_import_verified':True,'colab_api_tests_passed':API_TEST_PASS,'deployment_bundle_created':bundle.exists(),
          **external_evidence}
display(pd.Series(progress,name='evidence'))
write_json(RUN/'completion_checklist.json',progress)
```


<!-- Notebook original cell 265: markdown -->

### References and boundaries
- [UCI Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)
- [MLflow 3 migration and model URIs](https://mlflow.org/docs/latest/ml/mlflow-3)
- [MLflow tracking](https://mlflow.org/docs/latest/tracking/)
- [BentoML MLflow integration](https://docs.bentoml.org/en/latest/examples/mlflow.html)
- [BentoML model import/export](https://docs.bentoml.org/en/latest/build-with-bentoml/model-loading-and-management.html)
- [GitHub Actions publishing Docker images](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images)

No claims of clinical readiness, external test execution or successful deployment are made before observing actual output. Do not reuse the holdout to tune models. Next separate lab: logging/monitoring, drift, Prometheus/Grafana, infrastructure and retraining schedules.




---

# Part IX — Visual proof from the completed monitoring lab

The following screenshots were provided during the practical session. They are snapshots, not simulations or examples with fabricated values. Read them alongside Chapter 41 for metric meanings.

## 46. First successful Grafana dashboard

![Screenshot of the first Grafana dashboard: eight successful predictions, six rejected requests, target up one](image(8).png){width=6.4in}

This screenshot confirms that Grafana rendered measurements from Prometheus. `8` and `6` are cumulative test-generated counts, while `1` is Prometheus target scrape availability.

## 47. Dashboard after one regression test run

![Screenshot of the later Grafana dashboard: sixteen successes, twelve rejections, and a time-series requests panel](image(9).png){width=6.4in}

The second screenshot confirms an increase from eight to sixteen successful prediction responses and from six to twelve rejected requests after a second API test run. The traffic graph uses `sum by (http_response_code)` over the **cumulative** request counter. Its stair steps show counter updates; they are not raw instantaneous request rates. The fourth panel is still titled “New panel” in this screenshot; renaming was recommended but not verified. A latency panel was discussed, but there is no supplied screenshot proving that it was saved.

## 48. Reproducibility checklist

| Verification | Status supported by supplied evidence |
|---|---|
| Notebook source exists (266 cells, 115 code) | Yes, attached and in GitHub. |
| Selected SVM's evaluation report archived | Yes, released `evaluation_summary.json`. |
| Bento model archive hash and manifest | Yes, released in Git repository. |
| PR validation and merge | Yes, PR #1 merged. |
| CD publish after merge | Yes, successful GitHub Actions run. |
| v1.1.0 running healthily on Ubuntu | Yes, container and HTTP 200 logs supplied. |
| Regression tests after deployment | Yes, 13/13 test groups passed. |
| Prometheus collecting | Yes, `up=1`, API queried successfully. |
| Grafana dashboard | Yes, user screenshots with updated counters. |
| Cumulative latency panel | **Not evidenced as saved**; instructed and explained. |
| Prometheus durable TSDB volume | **Not configured** in supplied lab commands. |
| Alert notification delivery | **Not implemented**. |
| Production auto-deploy self-hosted runner | **Skipped intentionally**. |
| Kubernetes | **Not yet started**. |

---

# Part X — Final source-of-truth and verification statement

This manual includes the source content of the available 266-cell guided notebook, original lab documentation, observed deployments and Grafana screenshots, and links to immutable GitHub repository files. It cannot faithfully include transient notebook displays not exported with the notebook, private Google Drive MLflow tracking state, hidden user secrets or undocumented server-side configuration. Those are deliberately described as unavailable or unverified rather than invented.

**Recommended artifact hierarchy:** (1) original Colab `.ipynb` for executable training and plots; (2) immutable Git revision for source code; (3) GHCR digest for deployed binaries; (4) manifest and evaluation summary for lineage and holdout evidence; (5) this manual for comprehensive explanation; (6) Ubuntu monitoring config/dashboard export for reproducing observability on a persistent host.
