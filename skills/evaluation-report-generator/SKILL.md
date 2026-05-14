---
name: evaluation-report-generator
description: Generate a human-readable IPCV localization evaluation report from local run artifacts.
---

# Evaluation Report Generator

Use this skill to turn generated localization artifacts into a review-ready summary.

## Inputs

- `metrics.json`
- `report.md`
- `trajectory.png`
- `error_over_time.png`
- scenario YAML

## Workflow

1. Summarize the run goal and scenario.
2. List the most important metrics.
3. Explain what the plots should show.
4. Identify one risk or open question.
5. Recommend one follow-up replay.

## Output

Return a Markdown report suitable for a project review session.
