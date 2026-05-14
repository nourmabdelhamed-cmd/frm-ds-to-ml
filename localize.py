"""Agnostic visual-localization replay runner.

The runner only loads configuration and starts the workflow. Simulation,
failure injection, estimation, evaluation, artifacts, and MLflow logging live
inside the package.
"""

from ipcv.localization_workflow import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
