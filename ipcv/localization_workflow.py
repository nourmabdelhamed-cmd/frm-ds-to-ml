"""Config-driven localization workflow used by the course runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ipcv.evaluation import EvaluationResult, evaluate_trajectory
from ipcv.failures import BASELINE_SCENARIO, FailureScenario, apply_failure_scenario
from ipcv.localization import (
    KalmanLocalizationEstimator,
    LocalizationConfig,
    observations_to_frame,
    generate_ground_truth,
    simulate_observations,
)
from ipcv.observability import (
    log_localization_artifacts,
    plot_error_over_time,
    plot_trajectory,
    write_markdown_report,
    write_metrics,
)


@dataclass(frozen=True)
class LocalizationWorkflowConfig:
    """Top-level config for the localization replay command."""

    run_name: str = "localization-baseline"
    output_dir: str = "outputs/localization"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment: str = "ipcv-localization"
    log_artifacts: bool = True
    localization: LocalizationConfig = LocalizationConfig()

    def with_overrides(self, **overrides: Any) -> "LocalizationWorkflowConfig":
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean)


@dataclass(frozen=True)
class LocalizationRun:
    """Complete output from a localization replay."""

    truth: pd.DataFrame
    observations: pd.DataFrame
    estimates: pd.DataFrame
    evaluation: EvaluationResult
    artifacts: list[Path]


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _tuple2(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    if len(value) != 2:
        raise ValueError("expected a two-value sequence")
    return (float(value[0]), float(value[1]))


def _tuple4(value: Any, default: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if value is None:
        return default
    if len(value) != 4:
        raise ValueError("expected a four-value sequence")
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def load_localization_config(path: str | Path = "configs/localization.yaml") -> LocalizationWorkflowConfig:
    """Load nested YAML into workflow and localization dataclasses."""

    raw = _read_yaml(path)
    project = raw.get("project", {})
    simulation = raw.get("simulation", {})
    estimator = raw.get("estimator", {})
    tracking = raw.get("tracking", {})
    localization = LocalizationConfig(
        steps=int(simulation.get("steps", LocalizationConfig.steps)),
        dt=float(simulation.get("dt", LocalizationConfig.dt)),
        seed=int(simulation.get("seed", LocalizationConfig.seed)),
        initial_state=_tuple4(
            simulation.get("initial_state"),
            LocalizationConfig.initial_state,
        ),
        process_noise=float(
            simulation.get("process_noise", LocalizationConfig.process_noise)
        ),
        observation_noise=float(
            simulation.get("observation_noise", LocalizationConfig.observation_noise)
        ),
        estimator_process_noise=float(
            estimator.get(
                "process_noise",
                LocalizationConfig.estimator_process_noise,
            )
        ),
        estimator_measurement_noise=float(
            estimator.get(
                "measurement_noise",
                LocalizationConfig.estimator_measurement_noise,
            )
        ),
        initial_covariance=float(
            estimator.get("initial_covariance", LocalizationConfig.initial_covariance)
        ),
    )
    return LocalizationWorkflowConfig(
        run_name=project.get("run_name", LocalizationWorkflowConfig.run_name),
        output_dir=project.get("output_dir", LocalizationWorkflowConfig.output_dir),
        mlflow_tracking_uri=tracking.get(
            "mlflow_tracking_uri",
            LocalizationWorkflowConfig.mlflow_tracking_uri,
        ),
        mlflow_experiment=tracking.get(
            "mlflow_experiment",
            LocalizationWorkflowConfig.mlflow_experiment,
        ),
        log_artifacts=bool(
            tracking.get("log_artifacts", LocalizationWorkflowConfig.log_artifacts)
        ),
        localization=localization,
    )


def load_failure_scenario(path: str | Path | None = None) -> FailureScenario:
    """Load a failure scenario YAML file."""

    if path is None:
        return BASELINE_SCENARIO
    raw = _read_yaml(path)
    return FailureScenario(
        name=raw.get("name", Path(path).stem),
        description=raw.get("description", ""),
        drift_per_step=_tuple2(raw.get("drift_per_step"), (0.0, 0.0)),
        observation_delay_steps=int(raw.get("observation_delay_steps", 0)),
        drop_rate=float(raw.get("drop_rate", 0.0)),
        corruption_rate=float(raw.get("corruption_rate", 0.0)),
        corruption_scale=float(raw.get("corruption_scale", 1.0)),
        calibration_scale=float(raw.get("calibration_scale", 1.0)),
        seed=int(raw.get("seed", 42)),
    )


def resolve_scenario_path(scenario: str | None) -> Path | None:
    """Resolve a scenario name or path to a YAML file."""

    if scenario is None or scenario == "baseline":
        return None
    candidate = Path(scenario)
    if candidate.exists():
        return candidate
    named = Path("configs") / "failures" / f"{scenario}.yaml"
    if named.exists():
        return named
    raise FileNotFoundError(f"could not find scenario {scenario!r}")


def run_localization_workflow(
    config: LocalizationWorkflowConfig,
    *,
    scenario: FailureScenario = BASELINE_SCENARIO,
    enable_mlflow: bool = True,
) -> LocalizationRun:
    """Run simulation, failure injection, estimation, evaluation, and logging."""

    truth = generate_ground_truth(config.localization)
    raw_observations = simulate_observations(truth, config.localization)
    observations = apply_failure_scenario(
        raw_observations,
        scenario,
        dt=config.localization.dt,
    )
    estimator = KalmanLocalizationEstimator(config.localization)
    estimates = estimator.run(observations)
    observation_frame = observations_to_frame(observations)
    evaluation = evaluate_trajectory(truth, estimates)

    run_dir = Path(config.output_dir) / scenario.name
    run_dir.mkdir(parents=True, exist_ok=True)
    truth_path = run_dir / "truth.csv"
    observations_path = run_dir / "observations.csv"
    estimates_path = run_dir / "estimates.csv"
    errors_path = run_dir / "errors.csv"
    truth.to_csv(truth_path, index=False)
    observation_frame.to_csv(observations_path, index=False)
    estimates.to_csv(estimates_path, index=False)
    evaluation.errors.to_csv(errors_path, index=False)
    artifact_paths = [
        truth_path,
        observations_path,
        estimates_path,
        errors_path,
        write_metrics(evaluation.metrics, run_dir / "metrics.json"),
        write_markdown_report(
            metrics=evaluation.metrics,
            config=config,
            scenario=scenario,
            path=run_dir / "report.md",
        ),
        plot_trajectory(truth, estimates, run_dir / "trajectory.png"),
        plot_error_over_time(evaluation.errors, run_dir / "error_over_time.png"),
    ]

    if enable_mlflow:
        import mlflow

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        with mlflow.start_run(run_name=f"{config.run_name}-{scenario.name}"):
            log_localization_artifacts(
                mlflow,
                metrics=evaluation.metrics,
                artifact_paths=artifact_paths if config.log_artifacts else [],
                params={
                    "run_name": config.run_name,
                    "scenario": scenario.name,
                    "scenario_config": asdict(scenario),
                    "localization_config": asdict(config.localization),
                },
            )
            mlflow.set_tag("task", "visual-localization")
            mlflow.set_tag("runner", "localize.py")

    return LocalizationRun(
        truth=truth,
        observations=observation_frame,
        estimates=estimates,
        evaluation=evaluation,
        artifacts=artifact_paths,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the IPCV localization replay.")
    parser.add_argument("--config", default="configs/localization.yaml")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_localization_config(args.config).with_overrides(
        output_dir=args.output_dir,
        mlflow_tracking_uri=args.tracking_uri,
        run_name=args.run_name,
    )
    scenario = load_failure_scenario(resolve_scenario_path(args.scenario))
    run = run_localization_workflow(
        config,
        scenario=scenario,
        enable_mlflow=not args.no_mlflow,
    )
    print(
        json.dumps(
            {
                "metrics": run.evaluation.metrics,
                "artifacts": [str(path) for path in run.artifacts],
            },
            indent=2,
        )
    )
    return 0
