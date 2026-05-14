# Applied Computer Vision Systems Engineering

Hands-on course repository for transforming computer vision algorithms into reliable, testable, observable production systems.

The main case study is a small visual-localization system. Students start with image processing and geometry, then build a state estimator, inject failures, evaluate behavior, log results with MLflow, and finish with local AI-assisted engineering workflows.

The existing Pascal VOC semantic-segmentation workflow is preserved as an optional PyTorch appendix. It is no longer part of the default install path.

## Teaching Arc

```text
image processing prototype
  -> geometry and camera transforms
  -> localization estimator
  -> configurable production architecture
  -> reliability and failure scenarios
  -> data/replay workflows
  -> MLflow evaluation observability
  -> local agent-assisted engineering reports
```

## Project Layout

```text
lessons/                     # Quarto lesson pages for the semester arc
nbs/                         # optional nbdev/PyTorch segmentation appendix
ipcv/                        # reusable package code
tests/                       # fast deterministic tests
configs/                     # localization and failure scenario configs
data/                        # local datasets, ignored by Git except .gitkeep
models/                      # local checkpoints, ignored by Git except .gitkeep
outputs/                     # generated replay artifacts, ignored by Git
reports/                     # generated reports, ignored by Git
skills/                      # local AI-assisted engineering skills/templates
.github/workflows/           # CI and optional site deployment
localize.py                  # config-driven visual-localization runner
train.py                     # optional PyTorch segmentation runner
```

## Setup

Core course install:

```bash
uv sync --dev
```

Optional segmentation appendix:

```bash
uv sync --dev --group segmentation
```

Optional FastAPI handoff:

```bash
uv sync --dev --group service
```

## Run The Core Localization Workflow

Baseline replay without MLflow:

```bash
uv run python localize.py --config configs/localization.yaml --no-mlflow
```

Failure scenario examples:

```bash
uv run python localize.py --config configs/localization.yaml --scenario gps_drift --no-mlflow
uv run python localize.py --config configs/localization.yaml --scenario calibration_error --no-mlflow
uv run python localize.py --config configs/localization.yaml --scenario delayed_observations --no-mlflow
```

With MLflow enabled:

```bash
uv run python localize.py --config configs/localization.yaml --scenario gps_drift
```

Generated artifacts include:

- `truth.csv`
- `observations.csv`
- `estimates.csv`
- `errors.csv`
- `metrics.json`
- `report.md`
- `trajectory.png`
- `error_over_time.png`

These are written under `outputs/localization/<scenario>/` and ignored by Git.

## Inspect MLflow Runs

```bash
uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000`.

The localization experiment is `ipcv-localization`. It logs configuration, scenario metadata, trajectory metrics, plots, CSV artifacts, and Markdown reports.

## Run Tests

Core tests are deterministic and use synthetic data:

```bash
uv run pytest
```

Verify the nbdev export wrapper used by the optional appendix:

```bash
uv run nbdev-export
```

Segmentation tests are marked `segmentation` and skip automatically when PyTorch is not installed.

## Optional FastAPI Handoff

Install the service group, then run:

```bash
uv run --group service uvicorn ipcv.service:app --reload
```

Endpoints:

- `GET /health`
- `GET /scenarios`
- `POST /replay`
- `POST /evaluate`

The service is intentionally late-course material. The core workflow remains package code, configs, CLI, tests, artifacts, and MLflow.

## Optional PyTorch Segmentation Appendix

The original Pascal VOC segmentation workflow is kept in `nbs/` and `train.py`.

Install optional dependencies:

```bash
uv sync --dev --group segmentation
```

Run a fast synthetic segmentation smoke check:

```bash
uv run --group segmentation python train.py \
  --config params.yaml \
  --synthetic \
  --model tiny \
  --no-mlflow
```

Run the advanced MLflow segmentation tutorial from `nbs/04_advanced_mlflow_training.ipynb` when teaching learned perception or model registry concepts.

## DVC

Git tracks code, tests, docs, configs, and small metadata.

DVC is reserved for generated replay outputs, downloaded datasets, and model artifacts:

```bash
uv run dvc repro localize
```

The optional segmentation DVC stage requires the `segmentation` dependency group.

## Version-Control Boundaries

Track:

- package code
- tests
- lessons and notebooks
- configs
- workflow definitions
- small templates and skill files

Do not track:

- downloaded datasets
- checkpoints
- MLflow state
- replay outputs
- generated reports
- caches
- secrets
