"""Evaluation metrics for visual-localization runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics and per-step errors produced by an evaluation run."""

    metrics: dict[str, float]
    errors: pd.DataFrame


def trajectory_errors(truth: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    """Join truth and estimates and compute per-step localization errors."""

    merged = truth.merge(estimates, on="step", how="inner", suffixes=("_truth", "_estimate"))
    if merged.empty:
        raise ValueError("truth and estimates do not share any steps")
    dx = merged["estimate_x"] - merged["x"]
    dy = merged["estimate_y"] - merged["y"]
    merged["error_x"] = dx
    merged["error_y"] = dy
    merged["position_error"] = np.hypot(dx, dy)
    return merged


def evaluate_trajectory(truth: pd.DataFrame, estimates: pd.DataFrame) -> EvaluationResult:
    """Compute run-level localization quality metrics."""

    errors = trajectory_errors(truth, estimates)
    position_error = errors["position_error"].to_numpy(dtype=np.float64)
    innovation = errors["innovation_norm"].dropna().to_numpy(dtype=np.float64)
    corrected = errors["corrected"].astype(bool)
    metrics = {
        "trajectory_rmse": float(np.sqrt(np.mean(position_error**2))),
        "trajectory_mae": float(np.mean(position_error)),
        "trajectory_max_error": float(np.max(position_error)),
        "trajectory_p95_error": float(np.percentile(position_error, 95)),
        "final_drift": float(position_error[-1]),
        "correction_rate": float(corrected.mean()),
        "mean_innovation_norm": float(np.mean(innovation)) if innovation.size else 0.0,
    }
    return EvaluationResult(metrics=metrics, errors=errors)


def check_regression_thresholds(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[str]:
    """Return threshold failures as human-readable messages."""

    failures: list[str] = []
    for metric_name, threshold in thresholds.items():
        value = metrics.get(metric_name)
        if value is None:
            failures.append(f"missing metric {metric_name}")
        elif value > threshold:
            failures.append(
                f"{metric_name}={value:.4f} exceeds threshold {threshold:.4f}"
            )
    return failures
