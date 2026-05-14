---
name: regression-analyzer
description: Compare IPCV localization metrics across baseline and failure runs to flag regressions.
---

# Regression Analyzer

Use this skill before accepting a change to estimator, geometry, failure, or evaluation code.

## Inputs

- baseline `metrics.json`
- candidate `metrics.json`
- optional threshold dictionary from the reviewer

## Default Thresholds

- `trajectory_rmse <= 0.35`
- `final_drift <= 0.60`
- `trajectory_p95_error <= 0.70`

## Workflow

1. Load both metric files.
2. Report absolute and percentage changes.
3. Flag any threshold failures.
4. State whether the change is acceptable for the current lesson goal.

## Output

Return `pass`, `warn`, or `fail`, followed by the metric evidence.
