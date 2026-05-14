# ML Specialization

Use this reference when the course teaches ML, AI, computer vision, NLP, tabular modeling, or experiment workflows.

## Mapping

Generic course arc to ML:

```text
prototype              -> notebook or script experiment
reusable structure     -> package code
configuration          -> params.yaml and CLI overrides
tests                  -> synthetic/tiny data tests
automation             -> CI for tests/imports only
observability          -> MLflow metrics, artifacts, previews
artifact boundaries    -> DVC or ignored local data/model folders
documentation          -> nbdev, Quarto, MkDocs, or notebooks
handoff                -> model registry, docs site, or scheduled job
```

## Defaults

- Keep training optional or bounded.
- Use real data only in local/demo commands.
- Use synthetic or tiny fixtures in tests.
- Log parameters, metrics, checkpoints, and domain-specific previews.
- Do not optimize for accuracy unless the course is explicitly about modeling.
- Do optimize for reproducibility, inspection, and clear command paths.

## Recommended Files

```text
nbs/ or docs/
package_name/
tests/
data/.gitkeep
models/.gitkeep
params.yaml
dvc.yaml
train.py
README.md
.github/workflows/ci.yml
```

## MLflow Pattern

Log at least:

- config params
- train metrics
- validation metrics when available
- checkpoint artifact
- visual examples for vision/audio/text tasks where useful

Avoid committing local MLflow state:

```text
mlruns/
mlflow.db
mlflow*.db
```

## DVC Pattern

Use DVC to explain the boundary between:

```text
Git: code, config, tests, notebooks, docs
DVC: data, model checkpoints, large artifacts
```

Do not require a remote DVC setup for beginner courses unless remote collaboration is part of the lesson.

## CI Pattern

CI should:

- install dependencies
- export generated code if using nbdev
- verify imports
- run pytest

CI should not:

- download large datasets
- train full models
- require GPUs
- depend on local MLflow DBs
