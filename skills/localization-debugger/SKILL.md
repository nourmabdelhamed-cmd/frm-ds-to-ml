---
name: localization-debugger
description: Diagnose visual-localization replay failures from local IPCV configs, metrics, reports, and telemetry artifacts.
---

# Localization Debugger

Use this skill after running `localize.py`.

## Inputs

- `outputs/localization/<scenario>/metrics.json`
- `outputs/localization/<scenario>/report.md`
- `outputs/localization/<scenario>/errors.csv`
- `outputs/localization/<scenario>/estimates.csv`
- scenario YAML under `configs/failures/`

## Workflow

1. Compare the scenario metrics against `outputs/localization/baseline/metrics.json`.
2. Inspect `trajectory_rmse`, `final_drift`, `trajectory_p95_error`, and `mean_innovation_norm`.
3. Use `errors.csv` to identify when error starts growing.
4. Check whether the scenario changes drift, delay, corruption, drops, or calibration scale.
5. Produce a short diagnosis with evidence and one bounded fix to test next.

## Output

Write the answer as:

- Summary
- Evidence
- Likely cause
- Next experiment
