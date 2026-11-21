# Diabetes MLOps — Complete End-to-End Project Handbook

**Version:** 1.0 of this handbook  
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
