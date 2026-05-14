"""Failure scenario injection for localization replays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ipcv.localization import SensorObservation


@dataclass(frozen=True)
class FailureScenario:
    """A deterministic perturbation applied to sensor observations."""

    name: str = "baseline"
    description: str = "No injected failure."
    drift_per_step: tuple[float, float] = (0.0, 0.0)
    observation_delay_steps: int = 0
    drop_rate: float = 0.0
    corruption_rate: float = 0.0
    corruption_scale: float = 1.0
    calibration_scale: float = 1.0
    seed: int = 42


BASELINE_SCENARIO = FailureScenario()


def apply_failure_scenario(
    observations: list[SensorObservation],
    scenario: FailureScenario = BASELINE_SCENARIO,
    *,
    dt: float,
) -> list[SensorObservation]:
    """Return observations after applying deterministic production failures."""

    rng = np.random.default_rng(scenario.seed)
    drift = np.asarray(scenario.drift_per_step, dtype=np.float64)
    failed: list[SensorObservation] = []
    for index, observation in enumerate(observations):
        if scenario.drop_rate > 0.0 and rng.random() < scenario.drop_rate:
            continue

        position = np.asarray(observation.position, dtype=np.float64)
        position = position * scenario.calibration_scale
        position = position + drift * index
        valid = observation.valid
        if scenario.corruption_rate > 0.0 and rng.random() < scenario.corruption_rate:
            position = position + rng.normal(0.0, scenario.corruption_scale, size=2)
            valid = False

        failed.append(
            SensorObservation(
                step=observation.step + scenario.observation_delay_steps,
                timestamp=observation.timestamp + scenario.observation_delay_steps * dt,
                position=(float(position[0]), float(position[1])),
                source=f"{observation.source}:{scenario.name}",
                valid=valid,
            )
        )
    return failed
