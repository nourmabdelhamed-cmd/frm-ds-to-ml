"""Synthetic visual-localization replay and Kalman estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SensorObservation:
    """A timestamped position observation from a simulated vision/GPS sensor."""

    step: int
    timestamp: float
    position: tuple[float, float]
    source: str = "visual"
    valid: bool = True


@dataclass(frozen=True)
class LocalizationConfig:
    """Configuration for the deterministic localization replay."""

    steps: int = 50
    dt: float = 0.2
    seed: int = 42
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.25)
    process_noise: float = 0.02
    observation_noise: float = 0.08
    estimator_process_noise: float = 0.03
    estimator_measurement_noise: float = 0.08
    initial_covariance: float = 0.5
    landmarks: tuple[tuple[float, float], ...] = field(
        default_factory=lambda: ((0.0, 0.0), (5.0, 1.0), (8.0, 4.0), (3.0, 7.0))
    )


def generate_ground_truth(config: LocalizationConfig) -> pd.DataFrame:
    """Generate a smooth deterministic 2D trajectory."""

    if config.steps <= 0:
        raise ValueError("steps must be positive")
    state = np.asarray(config.initial_state, dtype=np.float64).copy()
    rows: list[dict[str, float | int]] = []
    rng = np.random.default_rng(config.seed)
    for step in range(config.steps):
        timestamp = step * config.dt
        rows.append(
            {
                "step": step,
                "timestamp": timestamp,
                "x": float(state[0]),
                "y": float(state[1]),
                "vx": float(state[2]),
                "vy": float(state[3]),
            }
        )
        turn = 0.025 * np.array(
            [
                np.sin(step * 0.17),
                np.cos(step * 0.13),
            ],
            dtype=np.float64,
        )
        process = rng.normal(0.0, config.process_noise, size=2)
        state[2:] = state[2:] + turn * config.dt + process * config.dt
        state[:2] = state[:2] + state[2:] * config.dt
    return pd.DataFrame(rows)


def simulate_observations(
    truth: pd.DataFrame,
    config: LocalizationConfig,
    *,
    source: str = "visual",
) -> list[SensorObservation]:
    """Create noisy position observations from a ground-truth trajectory."""

    rng = np.random.default_rng(config.seed + 1_000)
    observations: list[SensorObservation] = []
    for row in truth.itertuples(index=False):
        noise = rng.normal(0.0, config.observation_noise, size=2)
        position = (float(row.x + noise[0]), float(row.y + noise[1]))
        observations.append(
            SensorObservation(
                step=int(row.step),
                timestamp=float(row.timestamp),
                position=position,
                source=source,
            )
        )
    return observations


def observations_to_frame(observations: list[SensorObservation]) -> pd.DataFrame:
    """Convert observations into a telemetry table."""

    return pd.DataFrame(
        [
            {
                "step": obs.step,
                "timestamp": obs.timestamp,
                "observed_x": obs.position[0],
                "observed_y": obs.position[1],
                "source": obs.source,
                "valid": obs.valid,
            }
            for obs in observations
        ]
    )


class KalmanLocalizationEstimator:
    """Constant-velocity Kalman estimator for the localization labs."""

    def __init__(self, config: LocalizationConfig):
        self.config = config
        self.state = np.asarray(config.initial_state, dtype=np.float64).copy()
        self.covariance = np.eye(4, dtype=np.float64) * config.initial_covariance

    def predict(self) -> None:
        dt = self.config.dt
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = self.config.estimator_process_noise
        process_covariance = np.eye(4, dtype=np.float64) * q
        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process_covariance

    def correct(self, observation: SensorObservation) -> np.ndarray:
        measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        measurement_covariance = (
            np.eye(2, dtype=np.float64) * self.config.estimator_measurement_noise
        )
        measurement = np.asarray(observation.position, dtype=np.float64)
        innovation = measurement - measurement_matrix @ self.state
        innovation_covariance = (
            measurement_matrix @ self.covariance @ measurement_matrix.T
            + measurement_covariance
        )
        kalman_gain = (
            self.covariance
            @ measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - kalman_gain @ measurement_matrix) @ self.covariance
        return innovation

    def run(self, observations: list[SensorObservation]) -> pd.DataFrame:
        rows: list[dict[str, float | int | bool]] = []
        for observation in sorted(observations, key=lambda item: (item.step, item.timestamp)):
            self.predict()
            innovation = np.array([np.nan, np.nan], dtype=np.float64)
            corrected = False
            if observation.valid:
                innovation = self.correct(observation)
                corrected = True
            rows.append(
                {
                    "step": observation.step,
                    "timestamp": observation.timestamp,
                    "estimate_x": float(self.state[0]),
                    "estimate_y": float(self.state[1]),
                    "estimate_vx": float(self.state[2]),
                    "estimate_vy": float(self.state[3]),
                    "innovation_x": float(innovation[0]),
                    "innovation_y": float(innovation[1]),
                    "innovation_norm": float(np.linalg.norm(innovation))
                    if corrected
                    else np.nan,
                    "corrected": corrected,
                }
            )
        return pd.DataFrame(rows)
