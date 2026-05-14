"""MLflow and artifact helpers for localization observability."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

_matplotlib_cache = Path(tempfile.gettempdir()) / "ipcv-matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def plot_trajectory(
    truth: pd.DataFrame,
    estimates: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save a truth-vs-estimate trajectory plot."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(truth["x"], truth["y"], label="truth", linewidth=2)
    axis.plot(estimates["estimate_x"], estimates["estimate_y"], label="estimate")
    axis.set_title("Localization trajectory")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.axis("equal")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def plot_error_over_time(errors: pd.DataFrame, path: str | Path) -> Path:
    """Save an error-over-time plot."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(errors["timestamp_truth"], errors["position_error"], color="tab:red")
    axis.set_title("Localization error")
    axis.set_xlabel("time")
    axis.set_ylabel("position error")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    return output


def write_metrics(metrics: Mapping[str, float], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(metrics), indent=2), encoding="utf-8")
    return output


def write_markdown_report(
    *,
    metrics: Mapping[str, float],
    config: Any,
    scenario: Any,
    path: str | Path,
) -> Path:
    """Write a compact run report for review sessions and local skills."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metric_lines = "\n".join(
        f"- `{name}`: {value:.4f}" for name, value in sorted(metrics.items())
    )
    content = (
        "# Localization Evaluation Report\n\n"
        f"## Scenario\n\n```json\n{json.dumps(_json_ready(scenario), indent=2)}\n```\n\n"
        f"## Metrics\n\n{metric_lines}\n\n"
        f"## Config\n\n```json\n{json.dumps(_json_ready(config), indent=2)}\n```\n"
    )
    output.write_text(content, encoding="utf-8")
    return output


def log_localization_artifacts(
    mlflow_module: Any,
    *,
    metrics: Mapping[str, float],
    artifact_paths: list[Path],
    params: Mapping[str, Any],
) -> None:
    """Log localization metrics, params, and artifacts to MLflow."""

    safe_params = {
        key: json.dumps(_json_ready(value))
        if isinstance(value, (dict, list, tuple))
        else value
        for key, value in params.items()
    }
    mlflow_module.log_params(safe_params)
    for name, value in metrics.items():
        mlflow_module.log_metric(name, float(value))
    for artifact_path in artifact_paths:
        mlflow_module.log_artifact(str(artifact_path))
