---
name: calibration-checker
description: Review calibration-related localization failures from IPCV scenario configs and run artifacts.
---

# Calibration Checker

Use this skill when a localization run has high final drift or systematic position error.

## Inputs

- `configs/failures/calibration_error.yaml`
- `outputs/localization/<scenario>/errors.csv`
- `outputs/localization/<scenario>/trajectory.png`
- `outputs/localization/<scenario>/report.md`

## Workflow

1. Check whether `calibration_scale` differs from `1.0`.
2. Compare early and late position error to distinguish scale bias from random noise.
3. Inspect whether error direction is consistent across the replay.
4. Recommend one recalibration or validation check.

## Output

Return a concise calibration review with metric evidence and the next replay command.
