from pathlib import Path

import numpy as np

from ipcv.evaluation import check_regression_thresholds, evaluate_trajectory
from ipcv.failures import FailureScenario, apply_failure_scenario
from ipcv.localization import (
    KalmanLocalizationEstimator,
    LocalizationConfig,
    generate_ground_truth,
    simulate_observations,
)
from ipcv.localization_workflow import (
    load_failure_scenario,
    load_localization_config,
    run_localization_workflow,
)


def test_localization_replay_produces_metrics():
    config = LocalizationConfig(steps=12, observation_noise=0.01, seed=7)
    truth = generate_ground_truth(config)
    observations = simulate_observations(truth, config)
    estimates = KalmanLocalizationEstimator(config).run(observations)

    result = evaluate_trajectory(truth, estimates)

    assert result.metrics["trajectory_rmse"] >= 0
    assert result.metrics["correction_rate"] == 1.0
    assert len(result.errors) == 12


def test_failure_scenario_adds_drift():
    config = LocalizationConfig(steps=3, seed=1)
    truth = generate_ground_truth(config)
    observations = simulate_observations(truth, config)
    scenario = FailureScenario(name="drift", drift_per_step=(1.0, 0.0))

    failed = apply_failure_scenario(observations, scenario, dt=config.dt)

    assert failed[2].position[0] > observations[2].position[0] + 1.9


def test_regression_thresholds_return_failures():
    failures = check_regression_thresholds(
        {"trajectory_rmse": 2.0},
        {"trajectory_rmse": 1.0},
    )

    assert failures == ["trajectory_rmse=2.0000 exceeds threshold 1.0000"]


def test_localization_workflow_writes_artifacts(tmp_path):
    config_path = tmp_path / "localization.yaml"
    config_path.write_text(
        """
project:
  run_name: test-localization
  output_dir: {output_dir}
simulation:
  steps: 8
  dt: 0.2
  seed: 5
  observation_noise: 0.01
tracking:
  mlflow_tracking_uri: sqlite:///{mlflow_db}
  mlflow_experiment: ipcv-test
""".format(
            output_dir=tmp_path / "outputs",
            mlflow_db=tmp_path / "mlflow.db",
        ),
        encoding="utf-8",
    )
    workflow_config = load_localization_config(config_path)
    scenario = load_failure_scenario(None)

    run = run_localization_workflow(
        workflow_config,
        scenario=scenario,
        enable_mlflow=False,
    )

    artifact_names = {Path(path).name for path in run.artifacts}
    assert {"metrics.json", "report.md", "trajectory.png", "error_over_time.png"} <= artifact_names
    assert np.isfinite(run.evaluation.metrics["trajectory_rmse"])
