# From Computer Vision Student to Machine Learning Engineer

Minimal nbdev project for a 20-minute teaching arc on moving from a research notebook to a production-oriented ML workflow.

The demo uses Pascal VOC semantic segmentation, PyTorch, `segmentation_models_pytorch`, nbdev, MLflow, pytest, DVC, and GitHub Actions. The goal is not model accuracy. The goal is a workflow another researcher or engineer can understand, rerun, and change.

## Teaching Arc

```text
nbs/00_research_experiment.ipynb
  Research-style VOC training setup in a notebook.

nbs/01_collaboration_config_runner.ipynb
  A collaborator joins, so code is factored into reusable parts and config-driven train.py.

nbs/02_mlflow_tracking_registry.ipynb
  MLflow tracking records params, metrics, checkpoints, and optional model registry hooks.

nbs/03_ci_quality_gate.ipynb
  Fast synthetic tests and CI protect the workflow contract.

nbs/04_advanced_mlflow_training.ipynb
  Optional advanced tutorial: actual MLflow tracking, artifacts, model logging, and registration.
```

The intended story is:

```text
my experiment works
  -> someone else can change the config
  -> train.py runs the workflow without notebook state
  -> MLflow records what happened
  -> tests and CI protect the contract
```

## Project Layout

```text
nbs/                         # staged nbdev notebooks / presentation pages
ipcv/                        # nbdev package code used by notebooks, tests, and train.py
tests/                       # lightweight tests, no VOC download
data/                        # local VOC data, ignored by Git except .gitkeep
models/                      # local checkpoints, ignored by Git except .gitkeep
.github/workflows/           # CI and optional publishing infrastructure
params.yaml                  # training configuration
dvc.yaml                     # DVC training stage
train.py                     # config-driven training runner
```

## Install

This project uses `uv` and targets Python 3.12.

```bash
uv sync --dev
```

## Run Tests

Tests use synthetic data so they are fast and do not download VOC.

```bash
uv run pytest
```

Verify the nbdev export wrapper:

```bash
uv run nbdev-export
```

## Run The Real VOC Training Demo

The default config downloads Pascal VOC and trains a tiny CPU run on a small subset.

```bash
uv run python train.py --config params.yaml
```

Useful quick variants:

```bash
uv run python train.py --config params.yaml --max-samples 4 --max-batches 1
uv run python train.py --config params.yaml --synthetic --model tiny --no-mlflow
uv run python train.py --config params.yaml --synthetic --model tiny --max-samples 4 --max-batches 1 --log-model
```

For a pretrained visual sanity check, use torchvision's lightweight LR-ASPP model. This uses pretrained VOC-style segmentation weights, so the preview masks should look more reasonable than a randomly initialized U-Net:

```bash
uv run python train.py \
  --config params.yaml \
  --model lraspp_mobilenet_v3_large \
  --pretrained \
  --epochs 0 \
  --max-samples 4 \
  --device cpu \
  --checkpoint-path models/voc-lraspp-pretrained-preview.pt \
  --tracking-uri sqlite:///mlflow.db
```

Outputs:

- VOC data under `data/`
- MLflow local tracking under `mlruns/` and `mlflow.db`
- checkpoint under `models/voc-tiny-unet.pt`
- segmentation preview images in the MLflow run

## Inspect MLflow Runs

After running training with MLflow enabled, start the local MLflow UI:

```bash
uv run mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

The experiment name is `ipcv-voc-demo` by default. The UI shows run parameters, metrics, checkpoint artifacts, and segmentation preview images.

For a small MacBook VOC run with preview images:

```bash
uv run python train.py \
  --config params.yaml \
  --model unet \
  --max-samples 64 \
  --max-batches 20 \
  --epochs 3 \
  --device mps \
  --checkpoint-path models/voc-unet-macbook.pt \
  --tracking-uri sqlite:///mlflow.db \
  --log-model
```

Use `--device cpu` if `mps` is unavailable. In the MLflow run, check:

- `Metrics`: `train_loss`, `train_pixel_accuracy`, and `train_mean_iou` by epoch
- `Metrics`: `val_loss`, `val_pixel_accuracy`, and `val_mean_iou` by epoch
- `Images`: `segmentation_preview` by epoch
- `Artifacts`: `predictions/epoch_000.png`, `predictions/epoch_001.png`, and checkpoint files

If you want better-looking masks immediately, use the pretrained torchvision model instead of the randomly initialized U-Net:

```bash
uv run python train.py \
  --config params.yaml \
  --model lraspp_mobilenet_v3_large \
  --pretrained \
  --max-samples 32 \
  --max-batches 5 \
  --val-max-samples 32 \
  --val-max-batches 5 \
  --epochs 1 \
  --device mps \
  --checkpoint-path models/voc-lraspp-pretrained-macbook.pt \
  --tracking-uri sqlite:///mlflow.db \
  --log-every-n-batches 5
```

For a longer laptop run that still stays bounded:

```bash
uv run python train.py \
  --config params.yaml \
  --model lraspp_mobilenet_v3_large \
  --pretrained \
  --max-samples 128 \
  --max-batches 40 \
  --val-max-samples 128 \
  --val-max-batches 40 \
  --epochs 8 \
  --device mps \
  --checkpoint-path models/voc-lraspp-pretrained-longer.pt \
  --tracking-uri sqlite:///mlflow.db \
  --log-every-n-batches 5
```

The terminal prints run settings, the MLflow run ID, batch losses every `N` batches, validation summaries, preview artifact paths, and checkpoint logging state.

To train on the whole Pascal VOC train split, remove both teaching limits with `--full-train-split`:

```bash
uv run python train.py \
  --config params.yaml \
  --model lraspp_mobilenet_v3_large \
  --pretrained \
  --full-train-split \
  --val-max-samples 256 \
  --val-max-batches 128 \
  --epochs 5 \
  --device mps \
  --checkpoint-path models/voc-lraspp-pretrained-full-train.pt \
  --tracking-uri sqlite:///mlflow.db \
  --log-every-n-batches 25
```

This uses all images from `image_set: train` and does not cap training batches per epoch. Validation still runs every epoch on a capped subset so you can track learning without doubling the runtime. Add `--full-val-split` if you also want the whole VOC validation split.

## MLflow Registry Hook

Tracking is enabled by default. Model logging/registration is available but off by default:

```yaml
tracking:
  mlflow_experiment: ipcv-voc-demo
  checkpoint_path: models/voc-tiny-unet.pt
  log_model: false
  registered_model_name:
  log_prediction_images: true
  prediction_samples: 2
  prediction_image_key: segmentation_preview
```

Set `log_model: true` and provide `registered_model_name` when the tracking backend is ready for model registration.

The advanced MLflow tutorial is in `nbs/04_advanced_mlflow_training.ipynb`. It includes:

- a synthetic MLflow smoke run,
- a real VOC MLflow command,
- checkpoint artifact logging,
- segmentation preview image logging,
- PyTorch model logging,
- optional model registration,
- MLflow UI inspection steps.

Fast registered-model smoke command:

```bash
uv run python train.py \
  --config params.yaml \
  --synthetic \
  --model tiny \
  --max-samples 4 \
  --max-batches 1 \
  --checkpoint-path models/registered-smoke.pt \
  --log-model \
  --registered-model-name ipcv-voc-segmentation
```

## DVC

Git tracks small text artifacts: code, config, tests, and notebooks.

DVC is for large or generated artifacts: downloaded VOC data, checkpoints, and future experiment outputs.

The included stage runs the same agnostic training entry point:

```bash
uv run dvc repro
```

No DVC remote is required for the teaching setup.

## Work Like The nbdev Tutorial

The primary workflow is the same style as the nbdev tutorial:

- edit and run notebooks in JupyterLab
- keep code cells visible in the generated docs
- use `nbdev-preview` for a local docs website that updates while you work
- use `nbdev-export` when notebook code should update the package

Important distinction:

- The nbdev/Quarto docs view is a static reading view. Code blocks have copy buttons.
- The JupyterLab notebook or RISE view is the live execution view. Code cells have kernels and can run.

```bash
uv run jupyter lab
```

For live executable slides, open a notebook such as `nbs/00_research_experiment.ipynb` in JupyterLab and start the RISE slideshow with `Ctrl+R` (`Option+R` on macOS). The notebooks include slideshow metadata so markdown sections become slides and code cells appear as runnable fragments.

Live presentation flow:

1. Start JupyterLab:

   ```bash
   uv run jupyter lab
   ```

2. Open `nbs/00_research_experiment.ipynb`.
3. Wait for the Python kernel to attach.
4. Press `Option+R` on macOS, or `Ctrl+R`, to enter the RISE slideshow.
5. Move through slides with the arrow keys.
6. Run visible code cells with `Shift+Enter`.

If the shortcut does not work, open the command palette with `Cmd+Shift+C` and search for `RISE`.

In another terminal, preview the nbdev docs:

```bash
uv run nbdev-preview
```

If Quarto is not installed on the machine, install it once:

```bash
uv run nbdev-install-quarto
```

Notebook code is shown but not executed during docs rendering, so the preview remains quick and deterministic. Run the code from Jupyter when presenting or developing.

To render the docs site manually:

```bash
quarto render
```

The rendered files go to `_site/`.

If local `nbdev-preview` cannot find Quarto, either install Quarto locally or render through Docker:

```bash
docker run --rm -it \
  -v "$PWD":/project \
  -w /project \
  ghcr.io/quarto-dev/quarto:latest \
  quarto render
```

## Why These Practices Matter

- **Config-based training** moves choices out of hidden notebook state.
- **`train.py`** gives laptops, CI, DVC, and schedulers one stable entry point.
- **MLflow** records run parameters, metrics, and artifacts.
- **pytest** protects the workflow contract.
- **CI** gives the team a shared minimum quality gate.
- **DVC** separates code history from large data and model artifacts.
