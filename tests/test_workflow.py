import os
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn

from ipcv.workflow import (
    TrainingConfig,
    TorchvisionSegmentationModel,
    build_dataloader,
    build_segmentation_preview_image,
    build_tiny_model,
    build_validation_dataloader,
    evaluate_one_epoch,
    load_config,
    log_segmentation_preview,
    mlflow_safe_params,
    run_training,
)


def test_load_config_reads_nested_yaml(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text(
        """
project:
  run_name: test-run
data:
  root: example-data
  use_synthetic_data: true
  image_size: 32
  max_samples: 3
  validation_image_set: val
  validation_max_samples: 2
model:
  model_name: tiny
  use_pretrained_weights: true
  num_classes: 5
training:
  batch_size: 1
  epochs: 1
  max_batches: 1
  validation_max_batches: 1
  run_validation: true
  log_every_n_batches: 3
tracking:
  mlflow_tracking_uri: sqlite:///test.db
  checkpoint_path: example-model.pt
  log_model: true
  model_artifact_name: segmentation-model
  registered_model_name: ipcv-demo
  log_prediction_images: false
  prediction_samples: 1
  prediction_image_key: preview-test
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.run_name == "test-run"
    assert config.data_root == "example-data"
    assert config.use_synthetic_data is True
    assert config.image_size == 32
    assert config.validation_image_set == "val"
    assert config.validation_max_samples == 2
    assert config.model_name == "tiny"
    assert config.use_pretrained_weights is True
    assert config.num_classes == 5
    assert config.validation_max_batches == 1
    assert config.run_validation is True
    assert config.log_every_n_batches == 3
    assert config.mlflow_tracking_uri == "sqlite:///test.db"
    assert config.log_model is True
    assert config.model_artifact_name == "segmentation-model"
    assert config.registered_model_name == "ipcv-demo"
    assert config.log_prediction_images is False
    assert config.prediction_samples == 1
    assert config.prediction_image_key == "preview-test"


def test_synthetic_dataloader_shapes():
    config = TrainingConfig(
        use_synthetic_data=True,
        image_size=32,
        max_samples=4,
        batch_size=2,
        num_classes=7,
    )

    images, masks = next(iter(build_dataloader(config)))

    assert images.shape == (2, 3, 32, 32)
    assert masks.shape == (2, 32, 32)
    assert masks.max() < 7


def test_tiny_model_output_shape():
    config = TrainingConfig(model_name="tiny", image_size=32, num_classes=7)
    model = build_tiny_model(config)

    output = model(torch.randn(2, 3, 32, 32))

    assert output.shape == (2, 7, 32, 32)


def test_evaluate_one_epoch_returns_validation_metrics():
    config = TrainingConfig(
        use_synthetic_data=True,
        model_name="tiny",
        image_size=16,
        validation_max_samples=2,
        batch_size=2,
        num_classes=3,
    )
    model = build_tiny_model(config)

    metrics = evaluate_one_epoch(
        model=model,
        dataloader=build_validation_dataloader(config),
        device=torch.device("cpu"),
        num_classes=config.num_classes,
        max_batches=1,
    )

    assert metrics["val_batches"] == 1
    assert metrics["val_loss"] > 0
    assert 0 <= metrics["val_pixel_accuracy"] <= 1
    assert 0 <= metrics["val_mean_iou"] <= 1


def test_torchvision_wrapper_returns_logits_from_dict_output():
    class DictOutputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=1)

        def forward(self, images):
            return {"out": self.conv(images)}

    model = TorchvisionSegmentationModel(DictOutputModel(), normalize=True)

    output = model(torch.rand(2, 3, 16, 16))

    assert output.shape == (2, 4, 16, 16)


def test_segmentation_preview_image_has_three_columns():
    config = TrainingConfig(image_size=16, num_classes=3, prediction_samples=2)
    images = torch.rand(2, 3, 16, 16)
    masks = torch.randint(low=0, high=3, size=(2, 16, 16))
    logits = torch.randn(2, 3, 16, 16)

    preview = build_segmentation_preview_image(
        images=images,
        masks=masks,
        logits=logits,
        config=config,
    )

    assert preview.mode == "RGB"
    assert preview.size == (72, 86)


def test_log_segmentation_preview_logs_image_series_and_artifact():
    class MlflowStub:
        def __init__(self):
            self.logged_images = []

        def log_image(self, image, **kwargs):
            self.logged_images.append((image, kwargs))

    config = TrainingConfig(
        use_synthetic_data=True,
        model_name="tiny",
        image_size=16,
        max_samples=2,
        batch_size=2,
        num_classes=3,
        prediction_samples=1,
    )
    mlflow_stub = MlflowStub()

    artifact_file = log_segmentation_preview(
        mlflow_module=mlflow_stub,
        model=build_tiny_model(config),
        dataloader=build_dataloader(config),
        config=config,
        device=torch.device("cpu"),
        step=0,
    )

    assert artifact_file == "predictions/epoch_000.png"
    assert mlflow_stub.logged_images[0][1] == {
        "key": "segmentation_preview",
        "step": 0,
        "synchronous": True,
    }
    assert mlflow_stub.logged_images[1][1] == {
        "artifact_file": "predictions/epoch_000.png",
        "synchronous": True,
    }


def test_run_training_writes_checkpoint_without_mlflow(tmp_path):
    checkpoint = tmp_path / "model.pt"
    config = TrainingConfig(
        use_synthetic_data=True,
        model_name="tiny",
        image_size=32,
        max_samples=4,
        batch_size=2,
        epochs=1,
        max_batches=1,
        num_classes=3,
        validation_max_samples=2,
        validation_max_batches=1,
        checkpoint_path=str(checkpoint),
    )

    result = run_training(config, enable_mlflow=False)

    assert Path(result["checkpoint_path"]).exists()
    assert result["metrics"]["batches"] == 1
    assert result["metrics"]["train_batches"] == 1
    assert result["metrics"]["train_loss"] > 0
    assert 0 <= result["metrics"]["train_pixel_accuracy"] <= 1
    assert 0 <= result["metrics"]["train_mean_iou"] <= 1
    assert result["metrics"]["val_batches"] == 1
    assert result["metrics"]["val_loss"] > 0
    assert 0 <= result["metrics"]["val_pixel_accuracy"] <= 1
    assert 0 <= result["metrics"]["val_mean_iou"] <= 1


def test_mlflow_safe_params_serializes_none_values():
    params = mlflow_safe_params(
        TrainingConfig(log_model=True, registered_model_name=None)
    )

    assert params["log_model"] is True
    assert params["registered_model_name"] == ""


def test_train_py_runner_accepts_config_overrides(tmp_path):
    config_path = tmp_path / "params.yaml"
    checkpoint = tmp_path / "runner.pt"
    config_path.write_text(
        f"""
data:
  use_synthetic_data: true
  image_size: 32
  max_samples: 2
model:
  model_name: tiny
  num_classes: 3
training:
  batch_size: 1
  epochs: 1
  max_batches: 1
  validation_max_batches: 1
tracking:
  checkpoint_path: {checkpoint}
""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "train.py"),
            "--config",
            str(config_path),
            "--synthetic",
            "--model",
            "tiny",
            "--log-model",
            "--no-mlflow",
            "--log-every-n-batches",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert checkpoint.exists()
    assert "checkpoint_path" in completed.stdout
    assert "[ipcv] epoch 1/1 batch 1/1" in completed.stdout
    assert "train_pixel_accuracy=" in completed.stdout
    assert "train_mean_iou=" in completed.stdout
    assert "[ipcv] epoch 1/1 validation finished" in completed.stdout


def test_train_py_runner_accepts_full_train_split_flag(tmp_path):
    config_path = tmp_path / "params.yaml"
    checkpoint = tmp_path / "full-split-runner.pt"
    config_path.write_text(
        f"""
data:
  use_synthetic_data: true
  image_size: 16
  max_samples: 2
model:
  model_name: tiny
  num_classes: 3
training:
  batch_size: 1
  epochs: 1
  max_batches: 1
  run_validation: false
tracking:
  checkpoint_path: {checkpoint}
""",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "train.py"),
            "--config",
            str(config_path),
            "--synthetic",
            "--model",
            "tiny",
            "--full-train-split",
            "--no-mlflow",
            "--log-every-n-batches",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert checkpoint.exists()
    assert "samples=2" in completed.stdout
    assert "batches_per_epoch=2" in completed.stdout
