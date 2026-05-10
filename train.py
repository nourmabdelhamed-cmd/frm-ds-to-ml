"""Agnostic training runner for the demo.

This file intentionally knows only how to load configuration and start a run.
Dataset, model, training, tracking, and artifact behavior live in the package.
"""

from ipcv.workflow import cli_main


if __name__ == "__main__":
    raise SystemExit(cli_main())
