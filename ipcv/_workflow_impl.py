"""Minimal semantic segmentation workflow used by the nbdev slides and runner."""

from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


@dataclass(frozen=True)
class TrainingConfig:
    """Small config object that is friendly to notebooks, tests, and scripts."""

    run_name: str = "voc-tiny-cpu"
    data_root: str = "data"
    dataset_year: str = "2012"
    image_set: str = "train"
    validation_image_set: str = "val"
    download: bool = True
    image_size: int = 128
    max_samples: int = 12
    validation_max_samples: int = 64
    num_workers: int = 0
    use_synthetic_data: bool = False
    model_name: str = "unet"
    encoder_name: str = "resnet18"
    use_pretrained_weights: bool = False
    num_classes: int = 21
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 1e-3
    max_batches: int | None = 4
    validation_max_batches: int | None = 16
    run_validation: bool = True
    log_every_n_batches: int = 5
    seed: int = 42
    device: str = "cpu"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "ipcv-voc-demo"
    checkpoint_path: str = "models/voc-tiny-unet.pt"
    log_model: bool = False
    model_artifact_name: str = "model"
    registered_model_name: str | None = None
    log_prediction_images: bool = True
    prediction_samples: int = 2
    prediction_image_key: str = "segmentation_preview"

    def with_overrides(self, **overrides: Any) -> "TrainingConfig":
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path = "params.yaml") -> TrainingConfig:
    """Load the teaching config from nested YAML into a flat dataclass."""

    raw = _read_yaml(path)
    project = raw.get("project", {})
    data = raw.get("data", {})
    model = raw.get("model", {})
    training = raw.get("training", {})
    tracking = raw.get("tracking", {})
    return TrainingConfig(
        run_name=project.get("run_name", TrainingConfig.run_name),
        data_root=data.get("root", TrainingConfig.data_root),
        dataset_year=str(data.get("dataset_year", TrainingConfig.dataset_year)),
        image_set=data.get("image_set", TrainingConfig.image_set),
        validation_image_set=data.get(
            "validation_image_set", TrainingConfig.validation_image_set
        ),
        download=bool(data.get("download", TrainingConfig.download)),
        image_size=int(data.get("image_size", TrainingConfig.image_size)),
        max_samples=int(data.get("max_samples", TrainingConfig.max_samples)),
        validation_max_samples=int(
            data.get(
                "validation_max_samples", TrainingConfig.validation_max_samples
            )
        ),
        num_workers=int(data.get("num_workers", TrainingConfig.num_workers)),
        use_synthetic_data=bool(
            data.get("use_synthetic_data", TrainingConfig.use_synthetic_data)
        ),
        model_name=model.get("model_name", TrainingConfig.model_name),
        encoder_name=model.get("encoder_name", TrainingConfig.encoder_name),
        use_pretrained_weights=bool(
            model.get(
                "use_pretrained_weights", TrainingConfig.use_pretrained_weights
            )
        ),
        num_classes=int(model.get("num_classes", TrainingConfig.num_classes)),
        batch_size=int(training.get("batch_size", TrainingConfig.batch_size)),
        epochs=int(training.get("epochs", TrainingConfig.epochs)),
        learning_rate=float(
            training.get("learning_rate", TrainingConfig.learning_rate)
        ),
        max_batches=training.get("max_batches", TrainingConfig.max_batches),
        validation_max_batches=training.get(
            "validation_max_batches", TrainingConfig.validation_max_batches
        ),
        run_validation=bool(
            training.get("run_validation", TrainingConfig.run_validation)
        ),
        log_every_n_batches=int(
            training.get("log_every_n_batches", TrainingConfig.log_every_n_batches)
        ),
        seed=int(training.get("seed", TrainingConfig.seed)),
        device=training.get("device", TrainingConfig.device),
        mlflow_experiment=tracking.get(
            "mlflow_experiment", TrainingConfig.mlflow_experiment
        ),
        mlflow_tracking_uri=tracking.get(
            "mlflow_tracking_uri", TrainingConfig.mlflow_tracking_uri
        ),
        checkpoint_path=tracking.get(
            "checkpoint_path", TrainingConfig.checkpoint_path
        ),
        log_model=bool(tracking.get("log_model", TrainingConfig.log_model)),
        model_artifact_name=tracking.get(
            "model_artifact_name", TrainingConfig.model_artifact_name
        ),
        registered_model_name=tracking.get(
            "registered_model_name", TrainingConfig.registered_model_name
        ),
        log_prediction_images=bool(
            tracking.get(
                "log_prediction_images", TrainingConfig.log_prediction_images
            )
        ),
        prediction_samples=int(
            tracking.get("prediction_samples", TrainingConfig.prediction_samples)
        ),
        prediction_image_key=tracking.get(
            "prediction_image_key", TrainingConfig.prediction_image_key
        ),
    )


class VOCSegmentationTransform:
    """Resize VOC images and masks into tensors that a segmentation model expects."""

    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, image: Any, mask: Any) -> tuple[torch.Tensor, torch.Tensor]:
        image = image.convert("RGB")
        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )
        mask = TF.resize(
            mask,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.NEAREST,
        )
        return TF.to_tensor(image), torch.as_tensor(np.array(mask), dtype=torch.long)


class SyntheticSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tiny deterministic dataset used by tests and CI instead of downloading VOC."""

    def __init__(
        self,
        length: int = 4,
        image_size: int = 64,
        num_classes: int = 21,
        seed: int = 42,
    ):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed + index)
        image = torch.rand(3, self.image_size, self.image_size, generator=generator)
        mask = torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.image_size, self.image_size),
            generator=generator,
            dtype=torch.long,
        )
        return image, mask


def _build_segmentation_dataset(
    config: TrainingConfig,
    *,
    image_set: str,
    max_samples: int,
    seed: int,
) -> Dataset[Any]:
    """Build either a VOC split or a deterministic synthetic stand-in."""

    if config.use_synthetic_data:
        return SyntheticSegmentationDataset(
            length=max(1, max_samples),
            image_size=config.image_size,
            num_classes=config.num_classes,
            seed=seed,
        )

    dataset = VOCSegmentation(
        root=config.data_root,
        year=config.dataset_year,
        image_set=image_set,
        download=config.download,
        transforms=VOCSegmentationTransform(config.image_size),
    )
    if max_samples:
        return Subset(dataset, range(min(max_samples, len(dataset))))
    return dataset


def build_dataset(config: TrainingConfig) -> Dataset[Any]:
    """Build the configured training dataset."""

    return _build_segmentation_dataset(
        config,
        image_set=config.image_set,
        max_samples=config.max_samples,
        seed=config.seed,
    )


def build_validation_dataset(config: TrainingConfig) -> Dataset[Any]:
    """Build the configured validation dataset."""

    return _build_segmentation_dataset(
        config,
        image_set=config.validation_image_set,
        max_samples=config.validation_max_samples,
        seed=config.seed + 10_000,
    )


def build_dataloader(config: TrainingConfig) -> DataLoader[Any]:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )


def build_validation_dataloader(config: TrainingConfig) -> DataLoader[Any]:
    dataset = build_validation_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )


class TinySegmentationModel(nn.Module):
    """Small model for fast tests; the default demo uses segmentation_models_pytorch."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def build_tiny_model(config: TrainingConfig) -> nn.Module:
    return TinySegmentationModel(num_classes=config.num_classes)


class TorchvisionSegmentationModel(nn.Module):
    """Wrap torchvision segmentation models so the runner always receives logits."""

    def __init__(self, model: nn.Module, normalize: bool = False):
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            images = (images - self.mean) / self.std
        output = self.model(images)
        if isinstance(output, Mapping):
            return output["out"]
        return output


def build_torchvision_model(config: TrainingConfig) -> nn.Module:
    """Build a torchvision segmentation model, optionally with VOC-style weights."""

    if config.use_pretrained_weights and config.num_classes != 21:
        raise ValueError("Torchvision pretrained segmentation weights expect 21 classes.")

    from torchvision.models.segmentation import (
        DeepLabV3_ResNet50_Weights,
        FCN_ResNet50_Weights,
        LRASPP_MobileNet_V3_Large_Weights,
        deeplabv3_resnet50,
        fcn_resnet50,
        lraspp_mobilenet_v3_large,
    )

    if config.model_name == "deeplabv3_resnet50":
        weights = (
            DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            if config.use_pretrained_weights
            else None
        )
        model = deeplabv3_resnet50(
            weights=weights,
            weights_backbone=None,
            num_classes=None if weights else config.num_classes,
        )
    elif config.model_name == "fcn_resnet50":
        weights = (
            FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
            if config.use_pretrained_weights
            else None
        )
        model = fcn_resnet50(
            weights=weights,
            weights_backbone=None,
            num_classes=None if weights else config.num_classes,
        )
    elif config.model_name == "lraspp_mobilenet_v3_large":
        weights = (
            LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
            if config.use_pretrained_weights
            else None
        )
        model = lraspp_mobilenet_v3_large(
            weights=weights,
            weights_backbone=None,
            num_classes=None if weights else config.num_classes,
        )
    else:
        raise ValueError(f"Unsupported torchvision segmentation model: {config.model_name}")

    return TorchvisionSegmentationModel(
        model=model,
        normalize=config.use_pretrained_weights,
    )


def build_model(config: TrainingConfig) -> nn.Module:
    """Build the real segmentation model used in the local VOC demo."""

    if config.model_name == "tiny":
        return build_tiny_model(config)
    if config.model_name in {
        "deeplabv3_resnet50",
        "fcn_resnet50",
        "lraspp_mobilenet_v3_large",
    }:
        return build_torchvision_model(config)

    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=config.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=config.num_classes,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def console_log(message: str) -> None:
    """Print a concise training progress message for CLI demos."""

    print(f"[ipcv] {message}", flush=True)


def dataloader_batch_count(
    dataloader: DataLoader[Any],
    max_batches: int | None = None,
) -> int | None:
    """Return the expected number of batches when the dataloader can report it."""

    try:
        batches = len(dataloader)
    except TypeError:
        return max_batches
    if max_batches is None:
        return batches
    return min(max_batches, batches)


def dataloader_sample_count(dataloader: DataLoader[Any]) -> int | None:
    """Return the number of samples when the dataloader has a sized dataset."""

    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        return None
    try:
        return len(dataset)
    except TypeError:
        return None


def sanitize_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Map unknown labels to VOC's ignore label before cross entropy."""

    mask = mask.clone()
    invalid = (mask < 0) | ((mask >= num_classes) & (mask != 255))
    mask[invalid] = 255
    return mask


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
    progress_callback: Callable[[int, int | None, float], None] | None = None,
) -> dict[str, float | int]:
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model.train()
    total_loss = 0.0
    batches = 0
    expected_batches = dataloader_batch_count(dataloader, max_batches)
    correct_pixels = 0
    total_pixels = 0
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)

    for batch_index, (images, masks) in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        masks = sanitize_mask(masks.to(device), num_classes=num_classes)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        predictions = logits.argmax(dim=1)

        total_loss += float(loss.detach().cpu())
        batches += 1
        correct, total = update_segmentation_stats(
            intersections=intersections,
            unions=unions,
            predictions=predictions.detach().cpu(),
            masks=masks.detach().cpu(),
            num_classes=num_classes,
        )
        correct_pixels += correct
        total_pixels += total
        if progress_callback is not None:
            progress_callback(
                batches,
                expected_batches,
                float(loss.detach().cpu()),
            )

    if batches == 0:
        raise ValueError("No batches were seen. Check max_samples, batch_size, and max_batches.")
    metrics = segmentation_stats_to_metrics(
        prefix="train",
        total_loss=total_loss,
        batches=batches,
        correct_pixels=correct_pixels,
        total_pixels=total_pixels,
        intersections=intersections,
        unions=unions,
    )
    metrics["batches"] = batches
    return metrics


def update_segmentation_stats(
    intersections: torch.Tensor,
    unions: torch.Tensor,
    predictions: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
) -> tuple[int, int]:
    """Accumulate segmentation accuracy and IoU counts for one batch."""

    valid = masks != 255
    valid_predictions = predictions[valid]
    valid_masks = masks[valid]
    correct = int((valid_predictions == valid_masks).sum().item())
    total = int(valid_masks.numel())

    for class_id in range(num_classes):
        predicted_class = valid_predictions == class_id
        target_class = valid_masks == class_id
        intersections[class_id] += torch.logical_and(
            predicted_class, target_class
        ).sum()
        unions[class_id] += torch.logical_or(predicted_class, target_class).sum()

    return correct, total


def segmentation_stats_to_metrics(
    *,
    prefix: str,
    total_loss: float,
    batches: int,
    correct_pixels: int,
    total_pixels: int,
    intersections: torch.Tensor,
    unions: torch.Tensor,
) -> dict[str, float | int]:
    """Convert accumulated segmentation stats into metric names for MLflow."""

    present_classes = unions > 0
    mean_iou = (
        float((intersections[present_classes] / unions[present_classes]).mean())
        if bool(present_classes.any())
        else 0.0
    )
    pixel_accuracy = correct_pixels / total_pixels if total_pixels else 0.0
    return {
        f"{prefix}_loss": total_loss / batches,
        f"{prefix}_batches": batches,
        f"{prefix}_pixel_accuracy": pixel_accuracy,
        f"{prefix}_mean_iou": mean_iou,
    }


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: torch.device,
    num_classes: int,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate segmentation loss, pixel accuracy, and mean IoU."""

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    model.eval()
    total_loss = 0.0
    batches = 0
    correct_pixels = 0
    total_pixels = 0
    intersections = torch.zeros(num_classes, dtype=torch.float64)
    unions = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for batch_index, (images, masks) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            masks = sanitize_mask(masks.to(device), num_classes=num_classes)
            logits = model(images)
            loss = criterion(logits, masks)
            predictions = logits.argmax(dim=1)

            total_loss += float(loss.detach().cpu())
            batches += 1
            correct, total = update_segmentation_stats(
                intersections=intersections,
                unions=unions,
                predictions=predictions.detach().cpu(),
                masks=masks.detach().cpu(),
                num_classes=num_classes,
            )
            correct_pixels += correct
            total_pixels += total

    if batches == 0:
        raise ValueError(
            "No validation batches were seen. Check validation_max_samples "
            "and validation_max_batches."
        )

    return segmentation_stats_to_metrics(
        prefix="val",
        total_loss=total_loss,
        batches=batches,
        correct_pixels=correct_pixels,
        total_pixels=total_pixels,
        intersections=intersections,
        unions=unions,
    )


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: TrainingConfig,
    metrics: Mapping[str, Any],
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(config),
            "metrics": dict(metrics),
        },
        checkpoint_path,
    )
    return checkpoint_path


def mlflow_safe_params(config: TrainingConfig) -> dict[str, Any]:
    """Return scalar MLflow params with stable names for a nested training config."""

    return {key: "" if value is None else value for key, value in asdict(config).items()}


def mlflow_input_example(config: TrainingConfig) -> torch.Tensor:
    """Small example tensor for MLflow model signature/artifact previews."""

    return torch.zeros(1, 3, config.image_size, config.image_size, dtype=torch.float32)


def voc_color_palette(num_classes: int) -> np.ndarray:
    """Return the standard Pascal VOC-style class color palette."""

    palette = np.zeros((max(1, num_classes), 3), dtype=np.uint8)
    for class_id in range(num_classes):
        label = class_id
        for bit in range(8):
            palette[class_id, 0] |= ((label >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((label >> 1) & 1) << (7 - bit)
            palette[class_id, 2] |= ((label >> 2) & 1) << (7 - bit)
            label >>= 3
    return palette


def tensor_to_rgb_image(image: torch.Tensor) -> Any:
    """Convert a normalized CHW image tensor into a PIL RGB image."""

    from PIL import Image

    array = (
        image.detach()
        .cpu()
        .float()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray((array * 255).round().astype(np.uint8))


def mask_to_rgb_image(mask: torch.Tensor, num_classes: int) -> Any:
    """Colorize a class-index segmentation mask for visual inspection."""

    from PIL import Image

    mask_array = mask.detach().cpu().numpy()
    palette = voc_color_palette(num_classes)
    colorized = np.full((*mask_array.shape, 3), 224, dtype=np.uint8)
    valid = (mask_array >= 0) & (mask_array < num_classes)
    colorized[valid] = palette[mask_array[valid]]
    return Image.fromarray(colorized)


def build_segmentation_preview_image(
    images: torch.Tensor,
    masks: torch.Tensor,
    logits: torch.Tensor,
    config: TrainingConfig,
) -> Any:
    """Build an input / ground-truth / prediction panel for MLflow image logging."""

    from PIL import Image, ImageDraw

    rows = min(config.prediction_samples, images.shape[0])
    if rows <= 0:
        raise ValueError("prediction_samples must be positive to build a preview image.")

    predictions = logits.argmax(dim=1)
    cell_width = int(images.shape[-1])
    cell_height = int(images.shape[-2])
    caption_height = 18
    padding = 6
    columns = ("input", "target", "prediction")
    panel_width = len(columns) * cell_width + (len(columns) + 1) * padding
    row_height = caption_height + cell_height + padding
    panel_height = rows * row_height + padding
    panel = Image.new("RGB", (panel_width, panel_height), color="white")
    draw = ImageDraw.Draw(panel)

    for row in range(rows):
        y = padding + row * row_height
        cells = (
            tensor_to_rgb_image(images[row]),
            mask_to_rgb_image(masks[row], config.num_classes),
            mask_to_rgb_image(predictions[row], config.num_classes),
        )
        for column, (label, cell) in enumerate(zip(columns, cells, strict=True)):
            x = padding + column * (cell_width + padding)
            draw.text((x, y), label, fill=(30, 30, 30))
            panel.paste(cell, (x, y + caption_height))

    return panel


def log_segmentation_preview(
    mlflow_module: Any,
    model: nn.Module,
    dataloader: DataLoader[Any],
    config: TrainingConfig,
    device: torch.device,
    step: int,
) -> str | None:
    """Log a small segmentation preview as both an MLflow image and artifact."""

    if config.prediction_samples <= 0:
        return None

    was_training = model.training
    model.eval()
    try:
        try:
            images, masks = next(iter(dataloader))
        except StopIteration:
            return None

        images = images[: config.prediction_samples]
        masks = masks[: config.prediction_samples]
        with torch.no_grad():
            logits = model(images.to(device)).detach().cpu()

        panel = build_segmentation_preview_image(
            images=images,
            masks=masks,
            logits=logits,
            config=config,
        )
        artifact_file = f"predictions/epoch_{step:03d}.png"
        mlflow_module.log_image(
            panel,
            key=config.prediction_image_key,
            step=step,
            synchronous=True,
        )
        mlflow_module.log_image(
            panel,
            artifact_file=artifact_file,
            synchronous=True,
        )
        return artifact_file
    finally:
        model.train(was_training)


def log_pytorch_model(
    mlflow_module: Any,
    model: nn.Module,
    config: TrainingConfig,
) -> Any:
    """Log and optionally register the trained PyTorch model with MLflow."""

    import mlflow.pytorch

    was_training = model.training
    original_device = next(model.parameters()).device
    model.eval()
    model.to("cpu")
    try:
        return mlflow.pytorch.log_model(
            pytorch_model=model,
            name=config.model_artifact_name,
            input_example=mlflow_input_example(config).numpy(),
            registered_model_name=config.registered_model_name,
        )
    finally:
        model.to(original_device)
        model.train(was_training)


def run_training(
    config: TrainingConfig,
    *,
    model_builder: Callable[[TrainingConfig], nn.Module] = build_model,
    dataloader_builder: Callable[[TrainingConfig], DataLoader[Any]] = build_dataloader,
    validation_dataloader_builder: Callable[
        [TrainingConfig], DataLoader[Any]
    ] = build_validation_dataloader,
    enable_mlflow: bool = True,
) -> dict[str, Any]:
    """Run a tiny, reproducible segmentation training job from configuration."""

    set_seed(config.seed)
    device = torch.device(config.device)
    dataloader = dataloader_builder(config)
    validation_dataloader = (
        validation_dataloader_builder(config) if config.run_validation else None
    )
    model = model_builder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    expected_batches = dataloader_batch_count(dataloader, config.max_batches)
    sample_count = dataloader_sample_count(dataloader)
    validation_batches = (
        dataloader_batch_count(validation_dataloader, config.validation_max_batches)
        if validation_dataloader is not None
        else None
    )
    validation_samples = (
        dataloader_sample_count(validation_dataloader)
        if validation_dataloader is not None
        else None
    )
    preview_dataloader = validation_dataloader if validation_dataloader is not None else dataloader

    console_log(
        "starting training "
        f"run_name={config.run_name} model={config.model_name} "
        f"pretrained={config.use_pretrained_weights} "
        f"dataset={'synthetic' if config.use_synthetic_data else 'voc'} "
        f"samples={sample_count if sample_count is not None else 'unknown'}"
    )
    console_log(
        "settings "
        f"device={config.device} epochs={config.epochs} "
        f"batch_size={config.batch_size} batches_per_epoch={expected_batches} "
        f"lr={config.learning_rate} checkpoint={config.checkpoint_path}"
    )
    if validation_dataloader is not None:
        console_log(
            "validation "
            f"split={config.validation_image_set} "
            f"samples={validation_samples if validation_samples is not None else 'unknown'} "
            f"batches_per_epoch={validation_batches}"
        )
    else:
        console_log("validation disabled")

    mlflow_module = None
    run_context = nullcontext()
    if enable_mlflow:
        import mlflow

        mlflow_module = mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        run_context = mlflow.start_run(run_name=config.run_name)

    metrics: dict[str, Any] = {}
    prediction_artifacts: list[str] = []
    started_at = time.perf_counter()
    with run_context as active_run:
        if mlflow_module is not None:
            mlflow_module.log_params(mlflow_safe_params(config))
            run_id = active_run.info.run_id if active_run is not None else "unknown"
            console_log(
                "mlflow active "
                f"experiment={config.mlflow_experiment} run_id={run_id} "
                f"tracking_uri={config.mlflow_tracking_uri}"
            )
        else:
            console_log("mlflow disabled")

        if config.epochs == 0 and mlflow_module is not None and config.log_prediction_images:
            console_log("epochs=0, logging pretrained/current model preview only")
            prediction_artifact = log_segmentation_preview(
                mlflow_module=mlflow_module,
                model=model,
                dataloader=preview_dataloader,
                config=config,
                device=device,
                step=0,
            )
            if prediction_artifact is not None:
                prediction_artifacts.append(prediction_artifact)
                console_log(f"mlflow prediction preview logged artifact={prediction_artifact}")

        for epoch in range(config.epochs):
            epoch_started_at = time.perf_counter()
            console_log(f"epoch {epoch + 1}/{config.epochs} started")

            def progress_callback(
                batch_number: int,
                total_batches: int | None,
                loss_value: float,
            ) -> None:
                interval = max(config.log_every_n_batches, 0)
                is_last = total_batches is not None and batch_number == total_batches
                should_log = (
                    interval > 0
                    and (batch_number == 1 or batch_number % interval == 0 or is_last)
                )
                if not should_log:
                    return
                denominator = total_batches if total_batches is not None else "?"
                console_log(
                    f"epoch {epoch + 1}/{config.epochs} "
                    f"batch {batch_number}/{denominator} "
                    f"loss={loss_value:.4f}"
                )

            epoch_metrics = train_one_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                device=device,
                num_classes=config.num_classes,
                max_batches=config.max_batches,
                progress_callback=progress_callback,
            )
            validation_metrics: dict[str, float | int] = {}
            if validation_dataloader is not None:
                console_log(f"epoch {epoch + 1}/{config.epochs} validation started")
                validation_metrics = evaluate_one_epoch(
                    model=model,
                    dataloader=validation_dataloader,
                    device=device,
                    num_classes=config.num_classes,
                    max_batches=config.validation_max_batches,
                )
                console_log(
                    f"epoch {epoch + 1}/{config.epochs} validation finished "
                    f"val_loss={float(validation_metrics['val_loss']):.4f} "
                    f"val_pixel_accuracy={float(validation_metrics['val_pixel_accuracy']):.4f} "
                    f"val_mean_iou={float(validation_metrics['val_mean_iou']):.4f}"
                )
            metrics = {"epoch": epoch, **epoch_metrics, **validation_metrics}
            epoch_seconds = time.perf_counter() - epoch_started_at
            console_log(
                f"epoch {epoch + 1}/{config.epochs} finished "
                f"train_loss={float(epoch_metrics['train_loss']):.4f} "
                f"train_pixel_accuracy={float(epoch_metrics['train_pixel_accuracy']):.4f} "
                f"train_mean_iou={float(epoch_metrics['train_mean_iou']):.4f} "
                f"batches={int(epoch_metrics['batches'])} "
                f"seconds={epoch_seconds:.1f}"
            )
            if mlflow_module is not None:
                for metric_name, metric_value in {
                    **epoch_metrics,
                    **validation_metrics,
                }.items():
                    if metric_name == "epoch":
                        continue
                    mlflow_module.log_metric(metric_name, float(metric_value), step=epoch)
                console_log(
                    f"mlflow metrics logged step={epoch} "
                    f"train_loss={float(epoch_metrics['train_loss']):.4f}"
                    f" train_pixel_accuracy={float(epoch_metrics['train_pixel_accuracy']):.4f}"
                    f" train_mean_iou={float(epoch_metrics['train_mean_iou']):.4f}"
                    + (
                        f" val_loss={float(validation_metrics['val_loss']):.4f}"
                        f" val_pixel_accuracy={float(validation_metrics['val_pixel_accuracy']):.4f}"
                        f" val_mean_iou={float(validation_metrics['val_mean_iou']):.4f}"
                        if validation_metrics
                        else ""
                    )
                )
                if config.log_prediction_images:
                    prediction_artifact = log_segmentation_preview(
                        mlflow_module=mlflow_module,
                        model=model,
                        dataloader=preview_dataloader,
                        config=config,
                        device=device,
                        step=epoch,
                    )
                    if prediction_artifact is not None:
                        prediction_artifacts.append(prediction_artifact)
                        console_log(
                            "mlflow prediction preview logged "
                            f"artifact={prediction_artifact}"
                        )

        checkpoint = save_checkpoint(
            model=model,
            path=config.checkpoint_path,
            config=config,
            metrics=metrics,
        )
        console_log(f"checkpoint saved path={checkpoint}")
        if mlflow_module is not None:
            mlflow_module.log_artifact(str(checkpoint))
            console_log(f"mlflow checkpoint artifact logged path={checkpoint}")
            mlflow_module.set_tag("task", "semantic-segmentation")
            mlflow_module.set_tag("dataset", "voc-synthetic" if config.use_synthetic_data else "voc")
            mlflow_module.set_tag("runner", "train.py")
            if config.log_model:
                console_log("mlflow model logging started")
                log_pytorch_model(mlflow_module, model=model, config=config)
                console_log(f"mlflow model logged name={config.model_artifact_name}")

    console_log(f"finished training seconds={time.perf_counter() - started_at:.1f}")

    return {
        "checkpoint_path": str(checkpoint),
        "metrics": metrics,
        "prediction_artifacts": prediction_artifacts,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IPCV segmentation training demo.")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml.")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of VOC.")
    parser.add_argument(
        "--full-train-split",
        action="store_true",
        help="Use the full VOC train split and remove the per-epoch batch cap.",
    )
    parser.add_argument("--no-download", action="store_true", help="Do not download VOC.")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging.")
    parser.add_argument("--tracking-uri", default=None, help="Override MLflow tracking URI.")
    parser.add_argument("--log-model", action="store_true", help="Log the trained PyTorch model to MLflow.")
    parser.add_argument("--no-log-predictions", action="store_true", help="Disable MLflow prediction image logging.")
    parser.add_argument("--prediction-samples", type=int, default=None, help="Examples to show in MLflow prediction previews.")
    parser.add_argument("--registered-model-name", default=None, help="Optional MLflow registered model name.")
    parser.add_argument("--max-samples", type=int, default=None, help="Override sample count.")
    parser.add_argument("--val-max-samples", type=int, default=None, help="Override validation sample count.")
    parser.add_argument(
        "--full-val-split",
        action="store_true",
        help="Use the full validation split and remove the validation batch cap.",
    )
    parser.add_argument("--no-validation", action="store_true", help="Disable per-epoch validation.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--max-batches", type=int, default=None, help="Override batch limit.")
    parser.add_argument("--val-max-batches", type=int, default=None, help="Override validation batch limit.")
    parser.add_argument("--log-every-n-batches", type=int, default=None, help="Print batch progress every N batches; use 0 to disable batch logs.")
    parser.add_argument("--device", default=None, help="Override training device.")
    parser.add_argument(
        "--model",
        choices=[
            "unet",
            "tiny",
            "deeplabv3_resnet50",
            "fcn_resnet50",
            "lraspp_mobilenet_v3_large",
        ],
        default=None,
        help="Override model.",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights when the selected model supports them.")
    parser.add_argument("--checkpoint-path", default=None, help="Override checkpoint path.")
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config).with_overrides(
        use_synthetic_data=True if args.synthetic else None,
        download=False if args.no_download else None,
        max_samples=args.max_samples,
        validation_max_samples=args.val_max_samples,
        epochs=args.epochs,
        max_batches=args.max_batches,
        validation_max_batches=args.val_max_batches,
        run_validation=False if args.no_validation else None,
        log_every_n_batches=args.log_every_n_batches,
        device=args.device,
        model_name=args.model,
        use_pretrained_weights=True if args.pretrained else None,
        checkpoint_path=args.checkpoint_path,
        mlflow_tracking_uri=args.tracking_uri,
        log_model=True if args.log_model else None,
        log_prediction_images=False if args.no_log_predictions else None,
        prediction_samples=args.prediction_samples,
        registered_model_name=args.registered_model_name,
    )
    if args.full_train_split:
        config = replace(
            config,
            max_samples=0 if not config.use_synthetic_data else config.max_samples,
            max_batches=None,
        )
    if args.full_val_split:
        config = replace(
            config,
            validation_max_samples=(
                0 if not config.use_synthetic_data else config.validation_max_samples
            ),
            validation_max_batches=None,
        )
    result = run_training(config, enable_mlflow=not args.no_mlflow)
    print(json.dumps(result, indent=2))
    return 0
